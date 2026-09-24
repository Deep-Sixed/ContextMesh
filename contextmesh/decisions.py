"""Decision history: append-only, with the reasoning attached.

A decision node records why it was made, what it stood on, and what it replaced.
Superseding never deletes — the old decision stays walkable so "why did we
change our mind" is answerable by walking, not by reading a changelog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .graph import ContextGraph
from .model import (
    DECISION_AUTO_MINT_PREFIX,
    EdgeType,
    Node,
    NodeType,
    Provenance,
    decision_fingerprint,
    is_auto_minted_decision_id,
    slug,
)
from .ontology import OntologyError


@dataclass
class DecisionRecord:
    decision_id: str
    title: str
    rationale: str
    at_build: int
    supersedes: Optional[str] = None
    assumptions: List[str] = field(default_factory=list)
    supported_by: List[str] = field(default_factory=list)
    cites: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "title": self.title,
            "rationale": self.rationale,
            "at_build": self.at_build,
            "supersedes": self.supersedes,
            "assumptions": list(self.assumptions),
            "supported_by": list(self.supported_by),
            "cites": list(self.cites),
        }


class DecisionLog:
    """Append-only log mirrored into the graph as `decision` nodes."""

    def __init__(self, graph: ContextGraph) -> None:
        self.graph = graph
        self.records: List[DecisionRecord] = []

    def _mint_fresh_id(self, title: str) -> str:
        """The next auto id for `title`, guaranteed unused in the graph.

        Deliberately reads `self.graph.nodes`, not `self.records`: the
        latter is this `DecisionLog` object's own history and resets to
        empty for a fresh instance over the same graph (a restored Runner
        with no `decisions=` given constructs exactly this). Counting from
        durable graph state instead means the discriminator picks up where
        a *previous* `DecisionLog` over this graph left off, so "always a
        fresh id" holds across that reconstruction too, not just within one
        object's lifetime.

        Minted under `DECISION_AUTO_MINT_PREFIX`, a namespace `decide()`
        never lets an explicit id claim -- see `is_auto_minted_decision_id`.
        """
        n = 1
        while True:
            candidate = slug(f"{title}|{n}", DECISION_AUTO_MINT_PREFIX)
            if candidate not in self.graph.nodes:
                return candidate
            n += 1

    def decide(
        self,
        title: str,
        rationale: str,
        *,
        source_id: str,
        id: Optional[str] = None,
        supported_by: Iterable[str] = (),
        cites: Iterable[str] = (),
        assumptions: Iterable[str] = (),
        produces: Iterable[str] = (),
        supersedes: Optional[str] = None,
    ) -> Node:
        """Record a decision. Every call without an explicit id mints a new,
        never-reused decision, even if the content is byte-identical to a
        prior one -- a decision is an immutable event, not content-addressed,
        so "the same title again" is a second decision, not an update. Use
        `supersedes` to link it to the one it replaces. The id this mints is
        guaranteed unused in the graph, not merely unused in this
        `DecisionLog` object's own history, so it stays fresh across a new
        `DecisionLog` constructed over the same (possibly restored) graph.

        An explicit `id` is purely an idempotency key, not a content
        address: calling again with the same `id` and the exact same
        immutable content (title, rationale, source, and the sets of
        additional citations, claims, assumptions, entities and predecessor
        it references) is a true no-op that returns the existing node
        untouched; calling again with the same `id` and any different
        content is refused with `OntologyError` before anything is written,
        rather than silently rewriting what that id already means. This
        check compares against the *existing node's own provenance and
        edges* -- not a digest trusted from its attrs -- so a decision node
        that did not genuinely go through `decide()` (which `add_node`
        refuses to create in the first place) or a snapshot whose stored
        digest disagrees with the node's real content (which loading a
        snapshot independently rejects) cannot be mistaken for a legitimate
        retry target. An explicit `id` that names a decision minted
        *without* one (or any non-decision node) is refused the same way:
        it is not this call's event to retry, and this holds even if only
        the node's `decision_identity` attr, not its id, claims otherwise --
        an id `_mint_fresh_id` could have produced for this title is never
        accepted as an explicit target, no matter what its attrs say.

        The write is atomic: if any referenced id is invalid partway through
        (a bad source, claim, assumption, entity, or supersedes target), the
        decision and edges this call would have created are rolled back
        rather than left standing incomplete.
        """
        supported_by = tuple(supported_by)
        cites = tuple(cites)
        assumptions = tuple(assumptions)
        produces = tuple(produces)
        for target in assumptions:
            # depends_on also legally points decision->decision, but that is a
            # task ordering wired after decide(), which the fingerprint leaves
            # out. Accepting one here would store a digest the decision's own
            # edges can never reproduce: unloadable, and never retriable.
            found = self.graph.get(target)
            if found is not None and found.type is not NodeType.ASSUMPTION:
                raise OntologyError(
                    f"assumptions= takes assumption ids; {target!r} is a "
                    f"{found.type.value}"
                )
        fingerprint = decision_fingerprint(
            title=title,
            rationale=rationale,
            source_id=source_id,
            cites=set(cites) - {source_id},
            derived_from=supported_by,
            depends_on=assumptions,
            produces=produces,
            supersedes=supersedes,
        )

        if id is not None:
            existing = self.graph.get(id)
            if existing is not None:
                if (
                    existing.type is not NodeType.DECISION
                    or existing.attrs.get("decision_identity") != "explicit"
                    or is_auto_minted_decision_id(existing.id)
                ):
                    raise OntologyError(
                        f"decision id {id!r} already names a decision this "
                        "call did not mint (or a non-decision node); "
                        "refusing to treat this as an idempotent retry"
                    )
                existing_fingerprint = decision_fingerprint(
                    **self.graph._decision_structure(existing.id)
                )
                if existing_fingerprint != fingerprint:
                    raise OntologyError(
                        f"decision id {id!r} is already recorded with "
                        "different content; refusing to treat this call as "
                        "the same decision"
                    )
                return existing
            if is_auto_minted_decision_id(id):
                raise OntologyError(
                    f"explicit decision id {id!r} lies in the auto-minted "
                    f"namespace ({DECISION_AUTO_MINT_PREFIX!r}); choose a "
                    "different id -- this namespace is reserved so an "
                    "auto-minted decision's id always structurally proves "
                    "it, regardless of what its attrs claim"
                )
            node_id = id
            identity_mode = "explicit"
        else:
            node_id = self._mint_fresh_id(title)
            identity_mode = "auto"

        node = self.graph.add_node(
            NodeType.DECISION,
            title,
            id=node_id,
            attrs={
                "rationale": rationale,
                "decision_identity": identity_mode,
                "decision_payload_digest": fingerprint,
            },
            provenance=Provenance(
                source_id=source_id,
                extractor="decision-log",
                checks=["rationale-present", "provenance-present"],
                recorded_at_build=self.graph.build,
            ),
            _decision_mint_authorized=True,
        )

        created_edge_ids: List[str] = []

        def _tracked_edge(src: str, etype: EdgeType, dst: str) -> None:
            key = (src, etype.value, dst)
            already = key in self.graph._edge_key
            edge = self.graph.add_edge(src, etype, dst, _decision_edge_authorized=True)
            if not already:
                created_edge_ids.append(edge.id)

        try:
            _tracked_edge(node.id, EdgeType.CITES, source_id)
            for claim_id in supported_by:
                _tracked_edge(claim_id, EdgeType.SUPPORTS, node.id)
                _tracked_edge(node.id, EdgeType.DERIVED_FROM, claim_id)
            for src in cites:
                _tracked_edge(node.id, EdgeType.CITES, src)
            for assumption_id in assumptions:
                _tracked_edge(node.id, EdgeType.DEPENDS_ON, assumption_id)
            for entity_id in produces:
                _tracked_edge(node.id, EdgeType.PRODUCES, entity_id)
            if supersedes:
                _tracked_edge(node.id, EdgeType.SUPERSEDES, supersedes)
                self.graph.node(supersedes).attrs["superseded_by"] = node.id
        except Exception:
            # `node` was always freshly created just above -- both branches
            # above either returned early on an existing id or resolved one
            # verified absent from the graph -- so it is always safe to
            # discard here.
            for edge_id in created_edge_ids:
                self.graph._discard_edge(edge_id)
            self.graph._discard_node(node.id)
            raise

        self.records.append(
            DecisionRecord(
                decision_id=node.id,
                title=title,
                rationale=rationale,
                at_build=self.graph.build,
                supersedes=supersedes,
                assumptions=list(assumptions),
                supported_by=list(supported_by),
                cites=list(cites),
            )
        )
        return node

    def history_of(self, decision_id: str) -> List[DecisionRecord]:
        """Every version of a decision, oldest first."""
        by_id = {r.decision_id: r for r in self.records}
        chain: List[DecisionRecord] = []
        current: Optional[str] = decision_id
        while current and current in by_id:
            record = by_id[current]
            chain.append(record)
            current = record.supersedes
        return list(reversed(chain))

    def current(self) -> List[DecisionRecord]:
        """Decisions nothing has superseded yet."""
        replaced = {r.supersedes for r in self.records if r.supersedes}
        return [r for r in self.records if r.decision_id not in replaced]

    def to_dict(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.records]
