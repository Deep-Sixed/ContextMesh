"""CHUNK → EXTRACT → RESOLVE → LINK → EMBED → PRUNE.

What earns a node in the graph. Each stage reports what it admitted and what it
dropped; the two counters under the dashboard's build path — DROPPED AT RESOLVE
and COMMITTED → WALKABLE — are the RESOLVE and PRUNE lines of this report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .embed import embed
from .graph import ContextGraph
from .model import EdgeType, NodeType, Provenance, slug
from .resolve import Resolver
from .traverse import Walker

#: Node types the second half of PRUNE never drops, walked or not. `source`
#: was already exempt from the build-time half; `assumption` and `evidence`
#: join it here because GRAPH.md's rules give nothing walk telemetry the
#: authority to override -- an assumption nothing has questioned yet is not
#: thereby irrelevant, and evidence is what rule 7 requires an audit be able
#: to find.
_PRESERVED_FROM_WALK_PRUNE = (NodeType.SOURCE, NodeType.ASSUMPTION, NodeType.EVIDENCE)

#: Fewer distinct walks than this in the observation window and "nothing
#: touched it" is not a fact about the corpus yet -- it is one question that
#: happened to ask about something else. Two is the smallest count that is
#: not "one query."
MIN_OBSERVATION_WALKS = 2

STAGES: Tuple[Tuple[str, str], ...] = (
    ("CHUNK", "spans in"),
    ("EXTRACT", "entities + claims"),
    ("RESOLVE", "one id per thing"),
    ("LINK", "typed edges only"),
    ("EMBED", "vector on the node"),
    ("PRUNE", "nobody walked it"),
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
#: A sentence and its trailing whitespace, for splitting with offsets intact.
_SENTENCE_PIECE = re.compile(r".+?(?:[.!?](?=\s)|$)\s*", re.S)

# EXTRACT keeps a span only if it asserts something checkable. The test is
# lexical and therefore explicit: this list is the extractor's contract, and a
# span that does not match it is dropped and counted, never silently admitted.
_COPULA = (
    "is are was were be been being has have had does do did "
    "will would shall should must can cannot could may might"
).split()
_IRREGULAR = (
    "took grew built made ran fell sent held meant kept found became rose "
    "gave began chose left lost paid put set spent won wrote"
).split()
_STEMS = (
    "store build agree answer keep reach kill pass serve assume fit grow propose "
    "split measure increase reduce require accept upgrade take degrade improve add "
    "exceed supply receive look discard record replace consume ask dominate carri "
    "carry fall refer mark remain support detect force make continue provide size "
    "run account prefer cause break contradict observe fail return show find use "
    "need allow prevent raise drop hold mean cost spend index rebuild resolve link "
    "prune walk cite depend produce supersede justify serve rank score embed match "
    "split shard split scale train evaluate report exceed track trace"
).split()

_ASSERTIVE = re.compile(
    r"\b(?:"
    + "|".join(_COPULA + _IRREGULAR)
    + r"|(?:"
    + "|".join(sorted(set(_STEMS), key=len, reverse=True))
    + r")(?:s|es|ed|d|ing)?"
    + r")\b",
    re.I,
)


@dataclass
class Document:
    """One thing that text came from."""

    id: str
    title: str
    origin: str
    text: str
    retrieved_at: str = ""
    entities: List[str] = field(default_factory=list)
    #: (subject mention, relation, object mention) hints the extractor honours
    relations: List[Tuple[str, str, str]] = field(default_factory=list)
    #: Optional (start, end) blocks of ``text`` to sentence-split separately.
    #: Empty means "the whole text is one block", which is how the bundled
    #: corpus is built. A loader that reads real files sets these so a heading,
    #: a list item or a paragraph never runs into the next one, and so markup
    #: it has blanked out is never part of a claim.
    segments: List[Tuple[int, int]] = field(default_factory=list)


#: ``annotate(doc, start, end, sentence)`` -> extra attrs for the claim lifted
#: from ``doc.text[start:end]``, or None. Lets a loader record where in a real
#: file a claim sits and how much to trust the extraction, without the
#: pipeline knowing anything about files.
ClaimAnnotator = Callable[["Document", int, int, str], Optional[Dict[str, Any]]]


@dataclass
class StageReport:
    name: str
    caption: str
    admitted: int = 0
    dropped: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "caption": self.caption,
            "admitted": self.admitted,
            "dropped": self.dropped,
            "notes": list(self.notes),
        }


@dataclass
class BuildReport:
    build: int
    stages: List[StageReport]
    spans_in: int = 0
    dropped_at_resolve: int = 0
    committed_walkable: int = 0
    collapsed_aliases: int = 0
    pruned_nodes: int = 0

    def stage(self, name: str) -> StageReport:
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "build": self.build,
            "spans_in": self.spans_in,
            "dropped_at_resolve": self.dropped_at_resolve,
            "committed_walkable": self.committed_walkable,
            "collapsed_aliases": self.collapsed_aliases,
            "pruned_nodes": self.pruned_nodes,
            "stages": [s.to_dict() for s in self.stages],
        }


class Pipeline:
    """Builds a ContextGraph from documents, one stage at a time."""

    def __init__(
        self,
        graph: Optional[ContextGraph] = None,
        resolver: Optional[Resolver] = None,
        *,
        prune_unwalked: bool = True,
        annotate: Optional[ClaimAnnotator] = None,
    ) -> None:
        self.graph = ContextGraph() if graph is None else graph
        self.resolver = Resolver() if resolver is None else resolver
        self.prune_unwalked = prune_unwalked
        self.annotate = annotate

    # ── 01 CHUNK ─────────────────────────────────────────────────────────
    def chunk(self, doc: Document) -> List[Tuple[int, int, str]]:
        if doc.segments:
            return self._chunk_segments(doc)
        spans: List[Tuple[int, int, str]] = []
        cursor = 0
        for sentence in _SENTENCE.split(doc.text.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            start = doc.text.find(sentence, cursor)
            if start < 0:
                start = cursor
            end = start + len(sentence)
            cursor = end
            spans.append((start, end, sentence))
        return spans

    def _chunk_segments(self, doc: Document) -> List[Tuple[int, int, str]]:
        """Sentence-split each declared block, keeping offsets into ``doc.text``.

        The returned sentence has its whitespace collapsed, because a loader
        that blanks markup in place (to keep offsets true to the file) leaves
        runs of spaces behind; the span still points at the original bytes.
        """
        spans: List[Tuple[int, int, str]] = []
        for seg_start, seg_end in doc.segments:
            block = doc.text[seg_start:seg_end]
            for match in _SENTENCE_PIECE.finditer(block):
                piece = match.group(0)
                body = piece.strip()
                if not body:
                    continue
                start = seg_start + match.start() + len(piece) - len(piece.lstrip())
                spans.append((start, start + len(body), " ".join(body.split())))
        return spans

    # ── the whole build ──────────────────────────────────────────────────
    def build(self, docs: Sequence[Document]) -> BuildReport:
        graph = self.graph
        graph.build += 1
        report = BuildReport(
            build=graph.build,
            stages=[StageReport(name, caption) for name, caption in STAGES],
        )

        chunk = report.stage("CHUNK")
        extract = report.stage("EXTRACT")
        resolve = report.stage("RESOLVE")
        link = report.stage("LINK")
        embed_stage = report.stage("EMBED")
        prune = report.stage("PRUNE")

        # Seed the resolver with the entities each document declares. A name
        # that already resolves to a known entity is registered as an alias of
        # it — otherwise "PG Vector" would quietly become a second pgvector.
        for doc in docs:
            for name in doc.entities:
                existing = self.resolver.match(name)
                if existing is not None:
                    entity_id = existing[0]
                    self.resolver.register(
                        entity_id, self.resolver.canonical[entity_id], aliases=[name]
                    )
                    continue
                self.resolver.register(slug(name, NodeType.ENTITY.value), name)

        claim_spans: List[Tuple[Document, int, int, str, str]] = []

        for doc in docs:
            graph.add_node(
                NodeType.SOURCE,
                doc.title,
                id=doc.id,
                attrs={"origin": doc.origin, "retrieved_at": doc.retrieved_at},
            )
            chunk.admitted += 1

            spans = self.chunk(doc)
            report.spans_in += len(spans)

            # 02 EXTRACT — a span becomes a claim only if it asserts something.
            for start, end, sentence in spans:
                if not _ASSERTIVE.search(sentence) or len(sentence.split()) < 4:
                    extract.dropped += 1
                    continue
                claim_id = slug(sentence, NodeType.CLAIM.value)
                claim_spans.append((doc, start, end, sentence, claim_id))
                extract.admitted += 1

        # 03 RESOLVE — one id per real-world thing.
        entity_ids: Dict[str, str] = {}
        for doc in docs:
            for name in doc.entities:
                record = self.resolver.resolve(name)
                if record.resolved:
                    entity_ids[name] = record.canonical_id
                    resolve.admitted += 1
                else:
                    resolve.dropped += 1

        for _, _, _, sentence, _ in claim_spans:
            for mention in _candidate_mentions(sentence):
                record = self.resolver.resolve(mention)
                if record.resolved:
                    resolve.admitted += 1
                else:
                    resolve.dropped += 1

        report.dropped_at_resolve = resolve.dropped
        collapsed = self.resolver.collapsed()
        report.collapsed_aliases = sum(
            len(forms) - 1 for forms in collapsed.values() if len(forms) > 1
        )
        resolve.notes.append(
            f"{report.collapsed_aliases} surface forms folded into canonical ids"
        )

        for entity_id, label in self.resolver.canonical.items():
            # `canonical` and `aliases` are Must carry for Entity, so both go
            # in at creation rather than being patched onto the node after —
            # `add_node` now refuses a node that doesn't have them yet.
            aliases = [form for form in collapsed.get(entity_id, []) if form != label]
            graph.add_node(
                NodeType.ENTITY,
                label,
                id=entity_id,
                attrs={"canonical": label, "aliases": aliases},
            )

        # 04 LINK — typed edges only.
        # A document that declares an entity is itself a mention of it.
        for doc in docs:
            for name in doc.entities:
                record = self.resolver.resolve(name)
                if not record.resolved or record.canonical_id not in graph.nodes:
                    link.dropped += 1
                    continue
                graph.add_edge(doc.id, EdgeType.MENTIONS, record.canonical_id)
                graph.add_edge(record.canonical_id, EdgeType.DERIVED_FROM, doc.id)
                link.admitted += 2

        for doc, start, end, sentence, claim_id in claim_spans:
            # A sentence repeated in a second document is the same claim seen
            # again; its location attrs stay those of the first sighting, so
            # they never disagree with the provenance that sighting recorded.
            claim_attrs = (
                self.annotate(doc, start, end, sentence)
                if self.annotate is not None and claim_id not in graph.nodes
                else None
            )
            claim = graph.add_node(
                NodeType.CLAIM,
                sentence,
                id=claim_id,
                attrs=claim_attrs,
                provenance=Provenance(
                    source_id=doc.id,
                    span=(start, end),
                    extractor="assertive-span",
                    checks=["assertive", "span-bounded"],
                    recorded_at_build=graph.build,
                ),
            )
            graph.add_edge(claim.id, EdgeType.DERIVED_FROM, doc.id)
            link.admitted += 1

            for mention in _candidate_mentions(sentence):
                record = self.resolver.resolve(mention)
                if not record.resolved:
                    link.dropped += 1
                    continue
                target = record.canonical_id
                if target not in graph.nodes:
                    link.dropped += 1
                    continue
                graph.add_edge(claim.id, EdgeType.MENTIONS, target)
                graph.add_edge(doc.id, EdgeType.MENTIONS, target)
                graph.add_edge(target, EdgeType.DERIVED_FROM, doc.id)
                link.admitted += 3

        for doc in docs:
            for subject, relation, obj in doc.relations:
                left = self._hint(subject, doc)
                right = self._hint(obj, doc)
                if not (left and right) or left == right:
                    link.dropped += 1
                    continue
                try:
                    graph.add_edge(left, EdgeType(relation), right)
                    link.admitted += 1
                except Exception:
                    # An illegal pair is a dropped edge, never an untyped one.
                    link.dropped += 1

        link.notes.append(f"untyped edges: {graph.untyped_edges}")

        # 05 EMBED — vector on the node.
        for node in graph.nodes.values():
            if node.embedding is None:
                node.embedding = embed(f"{node.label} {node.attrs.get('rationale', '')}")
                embed_stage.admitted += 1

        # 06 PRUNE — nobody walked it.
        for node in graph.nodes.values():
            if node.type is NodeType.SOURCE:
                continue
            if graph.degree(node.id) == 0:
                node.pruned = True
                prune.dropped += 1
        report.pruned_nodes = prune.dropped
        prune.admitted = sum(1 for n in graph.nodes.values() if n.live)
        report.committed_walkable = sum(1 for e in graph.edges.values() if e.live)

        return report

    def _hint(self, hint: str, doc: Document) -> Optional[str]:
        """Resolve a relation hint to a node id.

        Hints are written the way a person would write them: a node id, a
        fragment of the claim they mean, or an entity name.
        """
        graph = self.graph
        if hint in graph.nodes:
            return hint
        needle = hint.lower()
        best: Optional[str] = None
        for node in graph.nodes.values():
            if node.type is not NodeType.CLAIM or needle not in node.label.lower():
                continue
            if node.provenance and node.provenance.source_id == doc.id:
                return node.id
            best = best or node.id
        if best:
            return best
        record = self.resolver.resolve(hint)
        if not record.resolved or record.canonical_id not in graph.nodes:
            return None
        # No claim quotes the hint verbatim; take a claim from this document
        # that mentions the entity instead.
        for edge in graph.in_edges(record.canonical_id, (EdgeType.MENTIONS,)):
            node = graph.node(edge.src)
            if (
                node.type is NodeType.CLAIM
                and node.provenance
                and node.provenance.source_id == doc.id
            ):
                return node.id
        return record.canonical_id

    def prune_unwalked_nodes(
        self,
        walker: Walker,
        *,
        min_walks: int = 1,
        min_observation_walks: int = MIN_OBSERVATION_WALKS,
    ) -> int:
        """The second half of PRUNE: drop what survived the build but nothing walked.

        ``walker`` is the observation window: the caller declares the run
        boundary by handing over the ``Walker`` that produced the telemetry,
        rather than this method inferring one from graph state it cannot see.
        Below ``min_observation_walks`` walks in that window this is a no-op --
        GRAPH.md rule 5 requires the window to have closed before "unwalked"
        means anything, since one query dead-ending is not evidence a node is
        irrelevant, only that this particular question did not need it.

        ``Node.walks`` -- incremented once per node a walk actually visits, in
        ``Walker.walk`` -- is the only signal a node's own eligibility is
        judged on. Degree is also read, inherited unchanged from the
        build-time half this method has always shared it with; nothing about
        a node's label, age, or type enters the decision beyond the type
        exemptions in ``_PRESERVED_FROM_WALK_PRUNE``. A node already not live
        -- pruned by an earlier call, or invalidated by a rejected assumption
        -- is skipped rather than re-decided, so this pass never creates a
        second, redundant state change on top of one that already happened.

        Deterministic and idempotent: the same graph and the same walk
        history produce the same dropped set every time, and calling this
        again with no new walks in between changes nothing.
        """
        if not self.prune_unwalked:
            return 0
        if walker.graph is not self.graph:
            raise ValueError(
                "walker was not run against this pipeline's graph; its "
                "telemetry does not describe what this graph's nodes saw"
            )
        if len(walker.walks) < min_observation_walks:
            return 0
        dropped = 0
        for node in self.graph.nodes.values():
            if node.type in _PRESERVED_FROM_WALK_PRUNE or not node.live:
                continue
            if node.walks < min_walks and self.graph.degree(node.id) <= 1:
                node.pruned = True
                dropped += 1
        return dropped


_CAPITALISED = re.compile(r"\b([A-Z][A-Za-z0-9+.-]*(?:\s+[A-Z][A-Za-z0-9+.-]*)*)\b")
_LOWER_TECH = re.compile(r"\b([a-z][a-z0-9]*(?:[-_.][a-z0-9]+)+)\b")


def _candidate_mentions(sentence: str) -> List[str]:
    """Surface forms worth asking the resolver about."""
    found: List[str] = []
    for match in _CAPITALISED.finditer(sentence):
        text = match.group(1).strip()
        if match.start() == 0 and " " not in text:
            continue  # sentence-initial single word is usually not a name
        if text not in found:
            found.append(text)
    for match in _LOWER_TECH.finditer(sentence):
        text = match.group(1)
        if text not in found:
            found.append(text)
    words = re.findall(r"\b[a-z][a-z0-9-]{2,}\b", sentence)
    for word in words:
        if len(word) >= 4 and word not in found:
            found.append(word)
    # Most real entity names are phrases, not single words: "build time",
    # "resident memory", "recall at ten". Without n-grams those never reach the
    # resolver and the entities sit in the graph with nothing said about them.
    for size in (2, 3):
        for i in range(len(words) - size + 1):
            phrase = " ".join(words[i : i + size])
            if phrase not in found:
                found.append(phrase)
    return found
