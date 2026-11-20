"""Refresh a folder graph: re-extract only what changed, retire what a file stopped saying.

See ``docs/PHASE2_INCREMENTAL_REBUILDS.md``. A refresh compares the folder
against the *current* source version of each file the graph knows, and:

* leaves unchanged files alone — their claims, and everything built on them,
  stand;
* extracts a changed or added file as a new source version, keeping claims
  for sentences it still says and creating claims for sentences that are new;
* retires sentences a file stopped saying, and every claim of a deleted file,
  by rejecting each claim's "file still says" assumption with one evidence
  node per file (GRAPH.md rule 7). Exactly what was built on those claims
  falls with them (rule 2), and nothing is deleted (rule 3);
* gives a sentence that returns after being retired a *new* claim that
  ``supersedes`` the dead one (rule 6: nothing is revived).

A file the folder still has but that is skipped this time (over a size limit,
undecodable) is left as it was, never treated as deleted.

Everything is decided from the graph itself: no side manifest exists to drift
from it. Paths are processed in sorted order and claims in graph order, so the
same sequence of folder states gives the same graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .assumptions import AssumptionLedger
from .graph import ContextGraph
from .ingest import (
    CLAIM_ASSUMPTION_AUTHOR,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_ENTITIES,
    DEFAULT_MAX_FILES,
    DEFAULT_MAX_TOTAL_BYTES,
    DEFAULT_SUFFIXES,
    ParsedFile,
    _annotator,
    _utc_iso,
    assume_claims,
    discover_entities,
    documents_from,
    grade_entities,
    label_sources,
    read_folder,
)
from .model import AssumptionStatus, EdgeType, Node, NodeType, Provenance, slug
from .pipeline import Pipeline, claim_id_for, is_claim
from .resolve import Resolver
from .temporal import parse_date

REFRESH_EXTRACTOR = "folder-refresh"


class RefreshError(Exception):
    """The graph cannot be refreshed from a folder (not a folder graph, too old)."""


@dataclass
class RefreshReport:
    """What a refresh changed, or with ``dry_run``, would change."""

    root: str
    dry_run: bool
    changed: bool
    build: int
    ingested_at: str
    files: Dict[str, List[str]] = field(default_factory=dict)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    claims_kept: int = 0
    claims_added: List[str] = field(default_factory=list)
    claims_retired: List[str] = field(default_factory=list)
    claims_restored: List[str] = field(default_factory=list)
    #: node id -> readable chain, for what fell *beyond* the retired claims
    #: themselves: decisions and anything else built on them.
    downstream: Dict[str, List[str]] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    entities_added: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "dry_run": self.dry_run,
            "changed": self.changed,
            "build": self.build,
            "ingested_at": self.ingested_at,
            "files": {k: list(v) for k, v in self.files.items()},
            "skipped": list(self.skipped),
            "claims": {
                "kept": self.claims_kept,
                "added": len(self.claims_added),
                "retired": len(self.claims_retired),
                "restored": len(self.claims_restored),
            },
            "claims_retired": list(self.claims_retired),
            "downstream": {k: list(v) for k, v in self.downstream.items()},
            "evidence": list(self.evidence),
            "entities_added": list(self.entities_added),
        }


# ── reading the graph's own record of the folder ─────────────────────────
def _current_sources(graph: ContextGraph) -> Dict[str, Node]:
    """path -> the source version a refresh compares against.

    Refuses graphs it cannot refresh rather than guessing: a graph with no
    folder sources, or one built before sources carried a content hash and a
    current-version marker.
    """
    current: Dict[str, Node] = {}
    folder_sources = 0
    for node in graph.nodes.values():
        if node.type is not NodeType.SOURCE:
            continue
        if node.attrs.get("source_time_basis") != "file-modified" and not node.attrs.get("deleted"):
            continue
        folder_sources += 1
        if "sha256" not in node.attrs or "current" not in node.attrs:
            raise RefreshError(
                "this graph was built before incremental refresh; rebuild it with "
                "--folder ... --save, then refresh that"
            )
        if node.attrs["current"]:
            path = node.attrs["path"]
            if path in current:
                raise RefreshError(f"two current versions of {path!r} in the graph")
            current[path] = node
    if not folder_sources:
        raise RefreshError("this graph was not built from a folder; there is nothing to refresh")
    return current


def _live_claims_by_path(graph: ContextGraph) -> Dict[str, Dict[str, Node]]:
    """path -> sentence -> the live claim saying it there."""
    out: Dict[str, Dict[str, Node]] = {}
    for node in graph.nodes.values():
        if node.type is NodeType.CLAIM and node.live and "path" in node.attrs:
            out.setdefault(node.attrs["path"], {})[node.label] = node
    return out


def _claim_assumption(graph: ContextGraph, claim_id: str) -> Optional[str]:
    """The active "file still says" assumption a folder claim rests on."""
    for edge in graph.out_edges(claim_id, [EdgeType.DEPENDS_ON]):
        assumption = graph.assumptions.get(edge.dst)
        if (
            assumption is not None
            and assumption.created_by == CLAIM_ASSUMPTION_AUTHOR
            and assumption.status is AssumptionStatus.ACTIVE
        ):
            return assumption.id
    return None


def _sentences(pipeline: Pipeline, doc: Any) -> List[Tuple[int, int, str]]:
    """The sentences EXTRACT would lift from ``doc``, in order, with spans."""
    return [(a, b, s) for a, b, s in pipeline.chunk(doc) if is_claim(s)]


# ── the refresh ──────────────────────────────────────────────────────────
def refresh_folder(
    graph: ContextGraph,
    resolver: Resolver,
    ledger: AssumptionLedger,
    root: Any,
    *,
    dry_run: bool = False,
    as_of: Optional[str] = None,
    suffixes: Iterable[str] = DEFAULT_SUFFIXES,
    max_files: int = DEFAULT_MAX_FILES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_entities: int = DEFAULT_MAX_ENTITIES,
) -> RefreshReport:
    """Bring ``graph`` up to date with the folder at ``root``; see the module doc.

    ``as_of`` (an ISO date) dates the evidence for deleted files, which the
    folder cannot date itself; without it that evidence is ``undated``, which
    ``temporal.py`` reports as uncertainty rather than guessing.

    With ``dry_run`` the graph is not touched: the report says what would
    change and what would fall. Otherwise the graph is changed in memory; the
    caller saves it (as a session's next generation) or discards it.
    """
    if as_of is not None:
        parse_date(as_of, where="as_of")
    ingested_at = _utc_iso(datetime.now(timezone.utc).timestamp())
    current = _current_sources(graph)
    base, files, skipped = read_folder(
        root, suffixes=suffixes, max_files=max_files, max_bytes=max_bytes,
        max_total_bytes=max_total_bytes, allow_empty=True,
    )
    skipped_paths = {s.path for s in skipped}

    by_path = {f.path: f for f in files}
    known = {p: f for p, f in by_path.items() if p in current}
    unchanged = sorted(p for p, f in known.items() if current[p].attrs["sha256"] == f.sha256)
    changed = sorted(p for p, f in known.items() if current[p].attrs["sha256"] != f.sha256)
    added = sorted(p for p in by_path if p not in current)
    # A file still in the folder but skipped this time is not deleted.
    deleted = sorted(p for p in current if p not in by_path and p not in skipped_paths)
    left_as_is = sorted(p for p in current if p not in by_path and p in skipped_paths)

    report = RefreshReport(
        root=str(base),
        dry_run=dry_run,
        changed=bool(changed or added or deleted),
        build=graph.build,
        ingested_at=ingested_at,
        files={
            "unchanged": unchanged,
            "changed": changed,
            "added": added,
            "deleted": deleted,
            "left_as_is": left_as_is,
        },
        skipped=[{"path": s.path, "reason": s.reason} for s in skipped],
    )
    if not report.changed:
        return report

    live = _live_claims_by_path(graph)
    entities = discover_entities(files, max_entities=max_entities)
    to_extract: List[ParsedFile] = [by_path[p] for p in sorted(changed + added)]
    docs = documents_from(to_extract, entities)
    probe = Pipeline(ContextGraph(), Resolver())

    # Decide, per file, which live claims are kept and which are retired.
    kept: Dict[str, Dict[str, Tuple[int, int]]] = {}
    retiring: Dict[str, List[Node]] = {}
    for doc, parsed in zip(docs, to_extract):
        now = {s: (a, b) for a, b, s in _sentences(probe, doc)}
        before = live.get(parsed.path, {})
        kept[parsed.path] = {s: now[s] for s in before if s in now}
        gone = [before[s] for s in before if s not in now]
        if gone:
            retiring[parsed.path] = gone
        report.claims_kept += len(kept[parsed.path])
    for path in deleted:
        gone = list(live.get(path, {}).values())
        if gone:
            retiring[path] = gone

    if dry_run:
        return _dry_run(graph, ledger, report, retiring, docs, to_extract, kept)

    build_before = graph.build
    restored: Dict[str, str] = {}  # new claim id -> the dead claim it supersedes

    def claim_ids_for(path: str, still: Set[str]):
        def claim_id_of(sentence: str) -> Optional[str]:
            if sentence in still:
                return None
            generation = 0
            while claim_id_for(sentence, path, generation) in graph.nodes:
                generation += 1
            claim_id = claim_id_for(sentence, path, generation)
            if generation:
                restored[claim_id] = claim_id_for(sentence, path, generation - 1)
            return claim_id

        return claim_id_of

    for doc, parsed in zip(docs, to_extract):
        doc.claim_id_of = claim_ids_for(parsed.path, set(kept[parsed.path]))

    entities_before = set(resolver.canonical)
    if docs:
        by_doc = {doc.id: f for doc, f in zip(docs, to_extract)}
        Pipeline(graph, resolver, annotate=_annotator(by_doc)).build(docs)
    else:
        graph.build += 1
    build = graph.build
    assert build == build_before + 1

    label_sources(graph, docs, to_extract)
    for path in changed:
        old = current[path]
        old.attrs["current"] = False
        old.attrs["replaced_at_build"] = build
    report.claims_added = assume_claims(graph, ledger, docs, to_extract)

    for new_id, dead_id in sorted(restored.items()):
        if new_id in graph.nodes and dead_id in graph.nodes:
            graph.add_edge(new_id, EdgeType.SUPERSEDES, dead_id)
            report.claims_restored.append(new_id)

    # Kept claims stay as they are, but say where they now sit. Provenance
    # keeps the first sighting: a fact about where the text was lifted.
    for doc, parsed in zip(docs, to_extract):
        annotate = _annotator({doc.id: parsed})
        for sentence, (start, end) in kept[parsed.path].items():
            claim = live[parsed.path][sentence]
            attrs = annotate(doc, start, end, sentence) or {}
            claim.attrs.update(attrs)
            claim.attrs["location_sha256"] = parsed.sha256

    grade_entities(graph, resolver, entities)
    report.entities_added = sorted(set(resolver.canonical) - entities_before)

    # Retire, one evidence node per file, in path order.
    tombstones = {path: _tombstone(graph, current[path], build, as_of) for path in deleted}
    for path in sorted(retiring):
        if path in deleted:
            source_id = tombstones[path]
            label = (
                f"{path} not found when the folder was re-read at build {build}; "
                f"its {len(retiring[path])} claim(s) no longer stand"
            )
            kind, new_sha = "file-missing", None
        else:
            parsed = by_path[path]
            source_id = next(d.id for d, f in zip(docs, to_extract) if f.path == path)
            label = (
                f"{path} re-read at build {build}: content {parsed.sha256[:12]} no longer "
                f"says {len(retiring[path])} sentence(s) that "
                f"{current[path].attrs['sha256'][:12]} said"
            )
            kind, new_sha = "file-reread", parsed.sha256
        evidence = graph.add_node(
            NodeType.EVIDENCE,
            label,
            attrs={
                "kind": kind,
                "path": path,
                "sha256": new_sha,
                "previous_sha256": current[path].attrs["sha256"],
            },
            provenance=Provenance(
                source_id=source_id,
                extractor=REFRESH_EXTRACTOR,
                checks=["sha256"],
                recorded_at_build=build,
            ),
        )
        report.evidence.append(evidence.id)
        for claim in retiring[path]:
            assumption_id = _claim_assumption(graph, claim.id)
            if assumption_id is None:
                continue
            outcome = ledger.reject(assumption_id, evidence_id=evidence.id)
            report.claims_retired.append(claim.id)
            for node_id, chain in outcome.invalidated.items():
                if node_id != claim.id and node_id not in report.downstream:
                    report.downstream[node_id] = chain

    for node_id in report.claims_retired:
        report.downstream.pop(node_id, None)
    report.build = build
    return report


def _tombstone(graph: ContextGraph, last: Node, build: int, as_of: Optional[str]) -> str:
    """A source recording that a file was found missing, dated if we can.

    Evidence for a deletion needs a source to date it from, and the last
    version's modification date would be wrong: it would date the fall to
    before the file was deleted. ``as_of`` if given, else ``undated``.
    """
    path = last.attrs["path"]
    last.attrs["current"] = False
    last.attrs["deleted_at_build"] = build
    node = graph.add_node(
        NodeType.SOURCE,
        f"{path} (not found at build {build})",
        id=slug(f"{path}@missing@{build}", NodeType.SOURCE.value),
        attrs={
            "origin": path,
            "retrieved_at": as_of if as_of is not None else "undated",
            "source_time_basis": "refresh-as-of" if as_of is not None else "undated",
            "path": path,
            "sha256": last.attrs["sha256"],
            "deleted": True,
            "current": False,
        },
    )
    return node.id


def _dry_run(
    graph: ContextGraph,
    ledger: AssumptionLedger,
    report: RefreshReport,
    retiring: Dict[str, List[Node]],
    docs: List[Any],
    to_extract: List[ParsedFile],
    kept: Dict[str, Dict[str, Tuple[int, int]]],
) -> RefreshReport:
    """Fill the report from read-only queries; the graph is not touched."""
    probe = Pipeline(ContextGraph(), Resolver())
    for doc, parsed in zip(docs, to_extract):
        for _, _, sentence in _sentences(probe, doc):
            if sentence in kept[parsed.path]:
                continue
            generation = 0
            while claim_id_for(sentence, parsed.path, generation) in graph.nodes:
                generation += 1
            report.claims_added.append(claim_id_for(sentence, parsed.path, generation))
            if generation:
                report.claims_restored.append(report.claims_added[-1])
    report.claims_added = sorted(set(report.claims_added))
    report.claims_restored = sorted(set(report.claims_restored))
    retired: Set[str] = set()
    for path in sorted(retiring):
        for claim in retiring[path]:
            assumption_id = _claim_assumption(graph, claim.id)
            if assumption_id is None:
                continue
            retired.add(claim.id)
            report.claims_retired.append(claim.id)
            for node_id, chain in ledger.blast_radius(assumption_id).items():
                report.downstream.setdefault(node_id, chain)
    for node_id in retired:
        report.downstream.pop(node_id, None)
    return report
