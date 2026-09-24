"""Edge-level assumption binding (GRAPH.md, "What edge-level assumption
binding means"): a bound edge falls with the assumption it is bound to; its
endpoints do not, unless rule 2 reaches them some other way.
"""

import unittest

from contextmesh.assumptions import AssumptionError, AssumptionLedger
from contextmesh.decisions import DecisionLog
from contextmesh.graph import ContextGraph, SnapshotError
from contextmesh.model import EdgeType, NodeType, Provenance


def _graph() -> ContextGraph:
    graph = ContextGraph()
    graph.build = 1
    return graph


def _source(graph, label="Doc", *, id=None):
    return graph.add_node(
        NodeType.SOURCE, label, id=id, attrs={"origin": "fixture", "retrieved_at": "fixture"}
    )


def _claim(graph, label, source, *, id=None):
    return graph.add_node(
        NodeType.CLAIM, label, id=id, provenance=Provenance(source_id=source.id)
    )


def _decision(graph, label, source, *, id=None, assumptions=()):
    return DecisionLog(graph).decide(
        label, "fixture", source_id=source.id, id=id, assumptions=assumptions
    )


def _evidence(graph, label="Contradiction", *, kind="disproof", id=None):
    return graph.add_node(NodeType.EVIDENCE, label, id=id, attrs={"kind": kind})


class BindingInvalidatesEdgeNotEndpointsTest(unittest.TestCase):
    """Assumption A binds Claim --supports--> Decision.

        Assumption A
            │
            └── binds relationship:
                Claim --supports--> Decision

        Reject A
            ↓
        supports edge = invalidated
        Claim = still live
        Decision = still live
    """

    def setUp(self):
        self.graph = _graph()
        self.ledger = AssumptionLedger(self.graph)
        self.source = _source(self.graph)
        self.claim = _claim(self.graph, "Shards stay under four gigabytes", self.source)
        self.decision = _decision(self.graph, "Rebuild the index in partitions", self.source)
        self.edge = self.graph.add_edge(self.claim.id, EdgeType.SUPPORTS, self.decision.id)
        self.assumption = self.ledger.assume("Shard count grows linearly with corpus size")
        self.ledger.justifies(self.assumption.id, self.edge.id)

    def test_binding_is_recorded_on_the_edge(self):
        self.assertEqual(self.edge.assumption_id, self.assumption.id)

    def test_rejecting_the_assumption_invalidates_only_the_bound_edge(self):
        report = self.ledger.reject(
            self.assumption.id, evidence_id=_evidence(self.graph).id
        )

        self.assertTrue(self.graph.edges[self.edge.id].invalidated)
        self.assertTrue(self.graph.node(self.claim.id).live)
        self.assertTrue(self.graph.node(self.decision.id).live)

        self.assertIn(self.edge.id, report.invalidated_edges)
        self.assertNotIn(self.claim.id, report.invalidated)
        self.assertNotIn(self.decision.id, report.invalidated)


class ExplicitDependencyStillPropagatesTest(unittest.TestCase):
    """Decision --depends_on--> A.

        Decision --depends_on--> A

        Reject A
            ↓
        Decision = invalidated
    """

    def setUp(self):
        self.graph = _graph()
        self.ledger = AssumptionLedger(self.graph)
        self.source = _source(self.graph)
        self.decision = _decision(self.graph, "Rebuild the index in partitions", self.source)
        self.assumption = self.ledger.assume("Shard count grows linearly with corpus size")
        self.graph.add_edge(self.decision.id, EdgeType.DEPENDS_ON, self.assumption.id)

    def test_rejecting_the_assumption_invalidates_the_dependent_decision(self):
        report = self.ledger.reject(
            self.assumption.id, evidence_id=_evidence(self.graph).id
        )
        self.assertTrue(self.graph.node(self.decision.id).invalidated)
        self.assertIn(self.decision.id, report.invalidated)


class CombinedBindingAndDependencyTest(unittest.TestCase):
    """A binds edge X and is also the explicit dependency of node Y.

        A binds edge X
        A is also explicit dependency of node Y

        Reject A
            ↓
        edge X dies because of binding
        node Y dies because of Rule 2
        X's endpoints do not die merely because X was bound
    """

    def setUp(self):
        self.graph = _graph()
        self.ledger = AssumptionLedger(self.graph)
        self.source = _source(self.graph)

        self.claim = _claim(self.graph, "Shards stay under four gigabytes", self.source)
        self.decision_x = _decision(
            self.graph, "Rebuild the index in partitions", self.source, id="decision:x"
        )
        self.edge_x = self.graph.add_edge(self.claim.id, EdgeType.SUPPORTS, self.decision_x.id)

        self.assumption = self.ledger.assume("Shard count grows linearly with corpus size")
        self.ledger.justifies(self.assumption.id, self.edge_x.id)
        # Y's dependency is recorded by decide() itself. Adding it afterwards
        # made this graph one that failed to load (Y is an explicit-id
        # decision), and is now refused.
        self.decision_y = _decision(
            self.graph,
            "Adopt tenant-aware sizing",
            self.source,
            id="decision:y",
            assumptions=[self.assumption.id],
        )

    def test_reject_kills_the_binding_and_the_dependency_but_not_x_s_endpoints(self):
        report = self.ledger.reject(
            self.assumption.id, evidence_id=_evidence(self.graph).id
        )

        self.assertTrue(
            self.graph.edges[self.edge_x.id].invalidated, "X should fall: it is bound"
        )
        self.assertTrue(
            self.graph.node(self.decision_y.id).invalidated, "Y should fall: rule 2, depends_on"
        )
        self.assertTrue(
            self.graph.node(self.claim.id).live, "X's src must not fall merely because X was bound"
        )
        self.assertTrue(
            self.graph.node(self.decision_x.id).live,
            "X's dst must not fall merely because X was bound",
        )

        self.assertIn(self.decision_y.id, report.invalidated)
        self.assertNotIn(self.claim.id, report.invalidated)
        self.assertNotIn(self.decision_x.id, report.invalidated)


class JustifiesContractTest(unittest.TestCase):
    def setUp(self):
        self.graph = _graph()
        self.ledger = AssumptionLedger(self.graph)
        self.source = _source(self.graph)
        self.claim = _claim(self.graph, "Shards stay under four gigabytes", self.source)
        self.decision = _decision(self.graph, "Rebuild the index in partitions", self.source)
        self.edge = self.graph.add_edge(self.claim.id, EdgeType.SUPPORTS, self.decision.id)
        self.assumption = self.ledger.assume("Shard count grows linearly with corpus size")

    def test_binding_a_live_edge_to_an_active_assumption_succeeds(self):
        self.ledger.justifies(self.assumption.id, self.edge.id)
        self.assertEqual(self.edge.assumption_id, self.assumption.id)

    def test_binding_the_same_pair_twice_is_idempotent(self):
        self.ledger.justifies(self.assumption.id, self.edge.id)
        self.ledger.justifies(self.assumption.id, self.edge.id)
        self.assertEqual(self.edge.assumption_id, self.assumption.id)

    def test_rebinding_to_a_different_assumption_is_refused(self):
        self.ledger.justifies(self.assumption.id, self.edge.id)
        other = self.ledger.assume("A different assumption entirely")
        with self.assertRaises(AssumptionError):
            self.ledger.justifies(other.id, self.edge.id)
        self.assertEqual(self.edge.assumption_id, self.assumption.id)

    def test_missing_assumption_is_refused(self):
        with self.assertRaises(AssumptionError):
            self.ledger.justifies("assumption:does-not-exist", self.edge.id)
        self.assertIsNone(self.edge.assumption_id)

    def test_rejected_assumption_cannot_receive_a_new_binding(self):
        self.ledger.reject(self.assumption.id, evidence_id=_evidence(self.graph).id)

        other_claim = _claim(self.graph, "A different claim", self.source, id="claim:other")
        other_decision = _decision(
            self.graph, "A different decision", self.source, id="decision:other"
        )
        other_edge = self.graph.add_edge(
            other_claim.id, EdgeType.SUPPORTS, other_decision.id
        )
        with self.assertRaises(AssumptionError):
            self.ledger.justifies(self.assumption.id, other_edge.id)
        self.assertIsNone(other_edge.assumption_id)

    def test_missing_edge_is_refused(self):
        with self.assertRaises(AssumptionError):
            self.ledger.justifies(self.assumption.id, "edge:does-not-exist")

    def test_dead_edge_is_refused(self):
        self.graph.edges[self.edge.id].invalidated = True
        with self.assertRaises(AssumptionError):
            self.ledger.justifies(self.assumption.id, self.edge.id)
        self.assertIsNone(self.edge.assumption_id)

    def test_add_edge_does_not_accept_an_assumption_id(self):
        """ContextGraph.add_edge must not be a second, unguarded binding
        path: it takes no assumption_id at all, so a caller cannot create a
        bound edge, or attach a binding to an existing one, without going
        through justifies()'s validation.
        """
        other_claim = _claim(self.graph, "A different claim", self.source, id="claim:other")
        other_decision = _decision(
            self.graph, "A different decision", self.source, id="decision:other"
        )
        with self.assertRaises(TypeError):
            self.graph.add_edge(
                other_claim.id,
                EdgeType.SUPPORTS,
                other_decision.id,
                assumption_id=self.assumption.id,
            )

    def test_re_adding_an_existing_edge_cannot_attach_a_binding_either(self):
        """The duplicate-relationship merge path in add_edge only ever grows
        weight/evidence; passing assumption_id to re-add an already-existing
        edge (the specific bypass this review found: an id silently attached
        to an unbound duplicate) must fail exactly like a fresh add does.
        """
        with self.assertRaises(TypeError):
            self.graph.add_edge(
                self.claim.id,
                EdgeType.SUPPORTS,
                self.decision.id,  # same (src, type, dst) as self.edge -> merge path
                assumption_id=self.assumption.id,
            )
        self.assertIsNone(self.edge.assumption_id)


class SnapshotConsistencyTest(unittest.TestCase):
    def _graph_with_binding(self):
        graph = _graph()
        ledger = AssumptionLedger(graph)
        source = _source(graph)
        claim = _claim(graph, "Shards stay under four gigabytes", source)
        decision = _decision(graph, "Rebuild the index in partitions", source)
        edge = graph.add_edge(claim.id, EdgeType.SUPPORTS, decision.id)
        assumption = ledger.assume("Shard count grows linearly with corpus size")
        ledger.justifies(assumption.id, edge.id)
        return graph, ledger, edge, assumption

    def test_a_live_binding_round_trips_unchanged(self):
        graph, _ledger, edge, assumption = self._graph_with_binding()
        restored = ContextGraph.from_dict(graph.to_dict())
        self.assertEqual(restored.edges[edge.id].assumption_id, assumption.id)
        self.assertTrue(restored.edges[edge.id].live)

    def test_a_rejected_assumptions_bound_edge_cannot_restore_as_live(self):
        graph, ledger, edge, assumption = self._graph_with_binding()
        ledger.reject(assumption.id, evidence_id=_evidence(graph).id)
        self.assertTrue(graph.edges[edge.id].invalidated)  # sanity: reject already killed it

        payload = graph.to_dict()
        for row in payload["edges"]:
            if row["id"] == edge.id:
                row["invalidated"] = False  # corrupt: bound edge claims to be live
        with self.assertRaisesRegex(SnapshotError, "rejected"):
            ContextGraph.from_dict(payload)

    def test_a_superseded_assumptions_bound_edge_may_still_be_live(self):
        """Supersession is replacement, not proof of falsity -- GRAPH.md."""
        graph, ledger, edge, assumption = self._graph_with_binding()
        ledger.supersede(assumption.id, "Shard count grows with tenant skew too")
        self.assertTrue(graph.edges[edge.id].live)

        restored = ContextGraph.from_dict(graph.to_dict())
        self.assertTrue(restored.edges[edge.id].live)
        self.assertEqual(restored.edges[edge.id].assumption_id, assumption.id)

    def test_a_genuinely_rejected_and_invalidated_binding_round_trips_unchanged(self):
        """The valid counterpart to the corrupted case above: a properly
        rejected assumption whose bound edge was actually invalidated by
        reject() restores exactly as it was, binding and all -- restoration
        must not clear assumption_id just because the assumption fell.
        """
        graph, ledger, edge, assumption = self._graph_with_binding()
        ledger.reject(assumption.id, evidence_id=_evidence(graph).id)
        self.assertTrue(graph.edges[edge.id].invalidated)

        restored = ContextGraph.from_dict(graph.to_dict())
        self.assertTrue(restored.edges[edge.id].invalidated)
        self.assertEqual(restored.edges[edge.id].assumption_id, assumption.id)


if __name__ == "__main__":
    unittest.main()
