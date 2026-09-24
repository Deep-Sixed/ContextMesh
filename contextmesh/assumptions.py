"""Assumptions as first-class, versioned graph objects — and what breaks when
one of them turns out to be false.

Selective invalidation is the load-bearing idea: rejecting an assumption must
invalidate *exactly* the work that stood on it, and must leave everything else
in place. The report proves both halves, because "we invalidated something" is
not a useful answer without "and here is what we deliberately kept".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from .graph import ContextGraph
from .model import (
    Assumption,
    AssumptionStatus,
    EdgeType,
    NodeType,
    slug,
)
from .ontology import ONTOLOGY, OntologyError

# Failure travels along these edges and no others (GRAPH.md rule 2), and the
# direction matters:
#
#   X -[depends_on]->  Y   Y failing invalidates X   → follow backwards
#   X -[derived_from]->Y   Y failing invalidates X   → follow backwards
#   X -[produces]->    Y   X failing invalidates Y   → follow forwards
#
# Nothing else propagates. `mentions`, `cites`, `supports` and `supersedes`
# survive an invalidation, which is what makes it selective rather than a purge.
#
# The direction itself is not restated here: `ontology.py` parses it from
# GRAPH.md's edge-table `Invalidation` column, and this module reads it from
# there, so it cannot hold a copy that disagrees with the file.
BACKWARD: FrozenSet[str] = ONTOLOGY.backward
FORWARD: FrozenSet[str] = ONTOLOGY.forward
PROPAGATING: FrozenSet[str] = ONTOLOGY.propagating


class AssumptionError(Exception):
    """Raised when an assumption operation violates graph invariants."""


@dataclass
class InvalidationReport:
    assumption_id: str
    statement: str
    #: node id -> the readable chain that made it depend on the assumption
    invalidated: Dict[str, List[str]] = field(default_factory=dict)
    invalidated_edges: List[str] = field(default_factory=list)
    preserved: List[str] = field(default_factory=list)
    replacement_id: Optional[str] = None

    @property
    def blast_radius(self) -> int:
        return len(self.invalidated)

    def why(self, node_id: str) -> str:
        chain = self.invalidated.get(node_id)
        if not chain:
            return f"{node_id} was not invalidated"
        return " → ".join(chain)

    def to_dict(self) -> Dict[str, object]:
        return {
            "assumption_id": self.assumption_id,
            "statement": self.statement,
            "blast_radius": self.blast_radius,
            "invalidated": {k: list(v) for k, v in self.invalidated.items()},
            "invalidated_edges": list(self.invalidated_edges),
            "preserved_count": len(self.preserved),
            "preserved": list(self.preserved),
            "replacement_id": self.replacement_id,
        }


def apply_invalidation(
    graph: "ContextGraph", assumption_id: str, radius: Dict[str, List[str]]
) -> List[str]:
    """Mark what a fallen assumption takes down with it. Returns the edge ids.

    One writer for one rule. :meth:`AssumptionLedger.reject` calls this when an
    assumption falls, and :func:`contextmesh.temporal.as_of_graph` calls it to
    derive the invalidation a past projection had — which has to be *derived*
    rather than copied forward, or the past inherits today's casualties.

    An edge falls for either of two independent reasons: it is bound to this
    assumption via `justifies` (``edge.assumption_id == assumption_id`` —
    the relationship itself no longer holds), or both its endpoints are in
    `radius` (rule 2 already reached them some other way). A bound edge's
    endpoints do not fall *because* the edge is bound; only `radius`, which
    `blast_radius` derives purely from rule 2's propagating edges, decides that.
    """
    graph.node(assumption_id).invalidated = True
    invalidated_edges: List[str] = []
    for edge in graph.edges.values():
        if edge.assumption_id == assumption_id or (edge.src in radius and edge.dst in radius):
            edge.invalidated = True
            invalidated_edges.append(edge.id)
    for node_id in radius:
        graph.node(node_id).invalidated = True
    return invalidated_edges


class AssumptionLedger:
    """Creates, supersedes and rejects assumptions against a ContextGraph."""

    def __init__(self, graph: ContextGraph) -> None:
        self.graph = graph
        self.history: List[Dict[str, object]] = []

    # ── lifecycle ────────────────────────────────────────────────────────
    def assume(
        self,
        statement: str,
        *,
        created_by: str = "agent",
        evidence_ids: Iterable[str] = (),
    ) -> Assumption:
        """Ground new work in a statement, or reuse the assumption this
        statement already names.

        Same statement, called again, is a true no-op: it returns the
        existing record exactly as it stands -- active, rejected, or
        superseded -- without touching its lifecycle. Re-asking is not a
        second way to reset an assumption back to active; only
        :meth:`supersede` puts new ground under a *different* statement.
        A different statement that happens to derive the same id (slug's
        digest is truncated, GRAPH.md rule 8) is a collision, not a reuse,
        and is refused the same way ``add_assumption`` refuses one.

        A reuse that asks for justification the existing record does not
        already carry is refused too. Returning the record unchanged would
        tell the caller its evidence was recorded when no ``justified_by``
        edge was written -- the no-op would silently drop part of the call.

        The write is atomic: if any justifying evidence id is invalid, the
        assumption this call would have created is rolled back rather than
        left standing without the justification the caller asked for.
        """
        existing = self.graph.assumptions.get(slug(statement, "assumption"))
        if existing is not None:
            if existing.statement != statement:
                raise OntologyError(
                    f"assumption id {existing.id!r} is already "
                    f"{existing.statement!r}; refusing to treat {statement!r} "
                    "as the same assumption"
                )
            missing = [ev for ev in evidence_ids if ev not in existing.evidence_ids]
            if missing:
                raise AssumptionError(
                    f"assumption {existing.id!r} already exists "
                    f"({existing.status.value}) without evidence {missing!r}; "
                    "re-asking the same statement does not add justification"
                )
            return existing

        assumption = Assumption(
            id=slug(statement, "assumption"),
            statement=statement,
            created_by=created_by,
            created_at_build=self.graph.build,
            evidence_ids=list(evidence_ids),
        )
        self.graph.add_assumption(assumption)
        created_edges: List[str] = []
        try:
            for ev in assumption.evidence_ids:
                edge = self.graph.add_edge(assumption.id, EdgeType.JUSTIFIED_BY, ev)
                created_edges.append(edge.id)
        except Exception:
            for edge_id in created_edges:
                self.graph._discard_edge(edge_id)
            del self.graph.assumptions[assumption.id]
            self.graph._discard_node(assumption.id)
            raise
        self._record("assumed", assumption.id, statement)
        return assumption

    def supersede(self, old_id: str, statement: str, *, created_by: str = "agent") -> Assumption:
        """Replace an assumption without deleting it. History is append-only."""
        old = self.graph.assumptions[old_id]
        new = Assumption(
            id=slug(f"{statement}|v{old.version + 1}", "assumption"),
            statement=statement,
            version=old.version + 1,
            created_by=created_by,
            created_at_build=self.graph.build,
            supersedes=old_id,
        )
        self.graph.add_assumption(new)
        old.status = AssumptionStatus.SUPERSEDED
        old.superseded_by = new.id
        self.graph.node(old.id).attrs["status"] = old.status.value
        self.graph.add_edge(new.id, EdgeType.SUPERSEDES, old_id)
        self._record("superseded", new.id, statement, replaces=old_id)
        return new

    def justifies(self, assumption_id: str, edge_id: str) -> None:
        """Bind an edge to an assumption: the relationship holds only while
        the assumption stands (GRAPH.md, "What edge-level assumption binding
        means"). Rejecting the assumption invalidates this edge directly; it
        does not, by itself, invalidate ``edge.src`` or ``edge.dst`` — a node
        that must fall with the assumption needs its own `depends_on`.

        Refuses a dangling assumption id, a dangling or dead edge id, and
        rebinding an edge already bound to a *different* assumption.
        Binding the same (assumption, edge) pair again is a no-op.
        """
        graph = self.graph
        if assumption_id not in graph.assumptions:
            raise AssumptionError(f"no assumption {assumption_id!r} in this graph")
        edge = graph.edges.get(edge_id)
        if edge is None:
            raise AssumptionError(f"no edge {edge_id!r} in this graph")
        if not edge.live:
            raise AssumptionError(f"edge {edge_id!r} is not live")
        if edge.assumption_id == assumption_id:
            return
        if edge.assumption_id is not None:
            raise AssumptionError(
                f"edge {edge_id!r} is already bound to assumption "
                f"{edge.assumption_id!r}; refusing to silently rebind it "
                f"to {assumption_id!r}"
            )
        assumption = graph.assumptions[assumption_id]
        if assumption.status is not AssumptionStatus.ACTIVE:
            raise AssumptionError(
                f"cannot bind edge {edge_id!r} to assumption {assumption_id!r}: "
                f"status is {assumption.status.value}, not active"
            )
        edge.assumption_id = assumption_id

    def depends(self, node_id: str, assumption_id: str) -> None:
        """Wire a decision or claim to the assumption it rests on.

        An explicit-id decision takes its assumptions through
        ``DecisionLog.decide(assumptions=...)`` instead; adding one here
        afterwards is refused, because it would change the content that
        decision's digest vouches for.
        """
        self.graph.add_edge(node_id, EdgeType.DEPENDS_ON, assumption_id)

    # ── the interesting part ─────────────────────────────────────────────
    def blast_radius(self, assumption_id: str) -> Dict[str, List[str]]:
        """Nodes whose validity depends on this assumption, with the chain why.

        Walks *backwards* along the propagating edge types from the assumption:
        a decision `depends_on` an assumption, a claim is `derived_from` that
        decision, an entity is `produces`d by it, and so on down the line.
        Rule 2 is the only mechanism this follows: an edge bound to this
        assumption via `justifies` does not seed its endpoints in here on its
        own, however that edge is invalidated too, in `apply_invalidation`,
        which does not depend on `blast_radius` to reach it.

        Direction comes from ``graph.ontology``, not the module-level
        ``BACKWARD``/``FORWARD``: ``ContextGraph`` accepts a custom
        ``Ontology`` (live, or restored through ``from_dict``), and a graph
        built against one has to invalidate under that same one, not
        whichever ontology happened to be the process-global default when
        this module was imported.
        """
        graph = self.graph
        backward = graph.ontology.backward
        forward = graph.ontology.forward
        assumption = graph.assumptions[assumption_id]
        start_label = f"assumption({assumption.statement[:48]})"

        reached: Dict[str, List[str]] = {}
        frontier: List[Tuple[str, List[str]]] = [(assumption_id, [start_label])]

        while frontier:
            node_id, chain = frontier.pop(0)
            if node_id != assumption_id:
                if node_id in reached and len(reached[node_id]) <= len(chain):
                    continue
                reached[node_id] = chain
            # Whoever depends on this node, or derives from it, falls with it.
            for edge in graph.in_edges(node_id, backward):
                parent = graph.node(edge.src)
                if parent.type is NodeType.ASSUMPTION:
                    continue
                frontier.append(
                    (parent.id, [*chain, f"<-{edge.type.value}-", parent.label])
                )
            # Whatever this node produced falls with it too.
            if node_id != assumption_id:
                for edge in graph.out_edges(node_id, forward):
                    child = graph.node(edge.dst)
                    frontier.append(
                        (child.id, [*chain, f"-{edge.type.value}->", child.label])
                    )
        return reached

    def reject(
        self,
        assumption_id: str,
        *,
        evidence_id: str,
        replacement: Optional[str] = None,
    ) -> InvalidationReport:
        """Disprove an assumption and invalidate exactly what stood on it.

        GRAPH.md Rule 7: An assumption is only ever rejected by evidence that
        contradicts it. A valid, live NodeType.EVIDENCE node is required.

        ``replacement`` must be new ground. A statement that already names an
        assumption -- the one being rejected, or any other, in any state --
        is refused before anything is written: reusing it would rewrite that
        record's version and lineage in place, and re-grounding on the
        rejected statement itself would make it supersede itself.
        """
        if not evidence_id or not isinstance(evidence_id, str):
            raise AssumptionError("reject() requires a non-empty evidence_id string")

        graph = self.graph
        if assumption_id not in graph.assumptions or assumption_id not in graph.nodes:
            raise AssumptionError(f"no assumption {assumption_id!r} in this graph")

        assumption = graph.assumptions[assumption_id]
        if assumption.status is not AssumptionStatus.ACTIVE:
            raise AssumptionError(
                f"cannot reject assumption {assumption_id!r}: "
                f"status is {assumption.status.value}, not active"
            )

        evidence_node = graph.nodes.get(evidence_id)
        if evidence_node is None:
            raise AssumptionError(
                f"evidence_id {evidence_id!r} is not in this graph"
            )
        if evidence_node.type is not NodeType.EVIDENCE:
            raise AssumptionError(
                f"evidence_id {evidence_id!r} is a {evidence_node.type.value}, not evidence"
            )
        if evidence_node.invalidated:
            raise AssumptionError(
                f"evidence_id {evidence_id!r} is invalidated"
            )

        if replacement:
            replacement_id = slug(replacement, "assumption")
            clash = graph.assumptions.get(replacement_id)
            if clash is not None or replacement_id in graph.nodes:
                named = clash.statement if clash is not None else replacement
                raise AssumptionError(
                    f"replacement {replacement!r} is not new ground: id "
                    f"{replacement_id!r} already names {named!r}"
                    + (f" ({clash.status.value})" if clash is not None else "")
                )

        radius = self.blast_radius(assumption_id)
        node = graph.node(assumption_id)

        # Pre-mutation state snapshot for atomic rollback
        original_status = assumption.status
        original_rejected_build = assumption.rejected_at_build
        original_node_status = node.attrs.get("status")
        original_node_invalidated = node.invalidated
        original_evidence_ids = list(assumption.evidence_ids)
        key = (evidence_id, EdgeType.CONTRADICTS.value, assumption_id)
        edge_already_existed = key in graph._edge_key
        created_edge_id: Optional[str] = None

        try:
            # 1. Establish the typed contradiction edge
            edge = graph.add_edge(evidence_id, EdgeType.CONTRADICTS, assumption_id)
            if not edge_already_existed:
                created_edge_id = edge.id

            # 2. Append evidence_id if not already present
            if evidence_id not in assumption.evidence_ids:
                assumption.evidence_ids.append(evidence_id)

            # 3. Commit REJECTED transition
            assumption.status = AssumptionStatus.REJECTED
            assumption.rejected_at_build = graph.build
            node.attrs["status"] = assumption.status.value
            node.invalidated = True
        except Exception:
            # Rollback: guarantee neither edge nor status commits alone
            assumption.status = original_status
            assumption.rejected_at_build = original_rejected_build
            if original_node_status is not None:
                node.attrs["status"] = original_node_status
            else:
                node.attrs.pop("status", None)
            try:
                node.invalidated = original_node_invalidated
            except Exception:
                pass
            assumption.evidence_ids = list(original_evidence_ids)
            if created_edge_id is not None:
                graph._discard_edge(created_edge_id)
            raise

        invalidated_edges = apply_invalidation(graph, assumption_id, radius)

        preserved = sorted(
            n.id
            for n in graph.nodes.values()
            if n.id not in radius and n.id != assumption_id and n.live
        )

        report = InvalidationReport(
            assumption_id=assumption_id,
            statement=assumption.statement,
            invalidated=radius,
            invalidated_edges=invalidated_edges,
            preserved=preserved,
        )

        if replacement:
            new = self.assume(replacement, created_by=assumption.created_by)
            new.version = assumption.version + 1
            new.supersedes = assumption_id
            # The node carries a copy of version; bumping only the record left
            # the two disagreeing, which the snapshot loader now catches.
            graph.sync_assumption(new)
            assumption.superseded_by = new.id
            graph.add_edge(new.id, EdgeType.SUPERSEDES, assumption_id)
            report.replacement_id = new.id

        self._record(
            "rejected",
            assumption_id,
            assumption.statement,
            blast_radius=report.blast_radius,
            preserved=len(preserved),
        )
        return report

    def active(self) -> List[Assumption]:
        return [
            a
            for a in self.graph.assumptions.values()
            if a.status is AssumptionStatus.ACTIVE
        ]

    def lineage(self, assumption_id: str) -> List[Assumption]:
        """Oldest first, following `supersedes` back to the original."""
        chain: List[Assumption] = []
        current: Optional[str] = assumption_id
        seen: Set[str] = set()
        while current and current not in seen:
            seen.add(current)
            assumption = self.graph.assumptions[current]
            chain.append(assumption)
            current = assumption.supersedes
        return list(reversed(chain))

    def _record(self, action: str, assumption_id: str, statement: str, **extra) -> None:
        self.history.append(
            {
                "build": self.graph.build,
                "action": action,
                "assumption_id": assumption_id,
                "statement": statement,
                **extra,
            }
        )
