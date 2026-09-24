"""Decision identity: an immutable event, not content-addressed.

GRAPH.md rule 9 / "What a decision's identity is": two calls to `decide()`
with byte-identical title and rationale are two decisions, not one being
re-observed. An explicit `id` is an idempotency key for one call, not a way
to name a decision by its content -- same id + same payload is a no-op,
same id + different payload fails closed, and a write that fails partway
through leaves nothing behind.
"""

import unittest

from contextmesh.assumptions import AssumptionLedger
from contextmesh.decisions import DecisionLog
from contextmesh.graph import ContextGraph, SnapshotError
from contextmesh.model import (
    EdgeType,
    NodeType,
    Provenance,
    is_auto_minted_decision_id,
    slug,
)
from contextmesh.ontology import OntologyError


def _graph_with_source():
    graph = ContextGraph()
    graph.build = 1
    source = graph.add_node(
        NodeType.SOURCE,
        "Fixture source",
        attrs={"origin": "fixture", "retrieved_at": "fixture"},
    )
    return graph, source


class AutoIdentityIsAlwaysFreshTest(unittest.TestCase):
    def test_repeated_title_and_rationale_mint_two_distinct_decisions(self):
        graph, source = _graph_with_source()
        decisions = DecisionLog(graph)

        first = decisions.decide(
            "Use PostgreSQL", "It fits our access patterns.", source_id=source.id
        )
        second = decisions.decide(
            "Use PostgreSQL", "It fits our access patterns.", source_id=source.id
        )

        self.assertNotEqual(first.id, second.id)
        self.assertEqual(len(decisions.records), 2)
        self.assertEqual(graph.type_counts(live_only=False)["decision"], 2)

    def test_third_repeat_also_gets_its_own_id(self):
        graph, source = _graph_with_source()
        decisions = DecisionLog(graph)
        ids = {
            decisions.decide("Same title", "Same rationale.", source_id=source.id).id
            for _ in range(3)
        }
        self.assertEqual(len(ids), 3)


class ExplicitIdIsAnIdempotencyKeyTest(unittest.TestCase):
    def test_same_id_same_payload_is_a_true_no_op(self):
        graph, source = _graph_with_source()
        decisions = DecisionLog(graph)

        first = decisions.decide(
            "Rebuild the index",
            "Bounded memory beats build time.",
            source_id=source.id,
            id="decision:step-1",
        )
        before_edges = len(graph.edges)
        before_records = len(decisions.records)

        second = decisions.decide(
            "Rebuild the index",
            "Bounded memory beats build time.",
            source_id=source.id,
            id="decision:step-1",
        )

        self.assertEqual(first.id, second.id)
        self.assertEqual(len(graph.edges), before_edges)
        self.assertEqual(len(decisions.records), before_records)

    def test_same_id_different_payload_fails_closed(self):
        graph, source = _graph_with_source()
        decisions = DecisionLog(graph)
        decisions.decide(
            "Rebuild the index",
            "Bounded memory beats build time.",
            source_id=source.id,
            id="decision:step-1",
        )

        with self.assertRaises(OntologyError):
            decisions.decide(
                "Rebuild the index",
                "Different rationale entirely.",
                source_id=source.id,
                id="decision:step-1",
            )

    def test_rejected_retry_leaves_no_partial_state(self):
        graph, source = _graph_with_source()
        decisions = DecisionLog(graph)
        decisions.decide(
            "Rebuild the index",
            "Bounded memory beats build time.",
            source_id=source.id,
            id="decision:step-1",
        )
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)
        record_count = len(decisions.records)

        with self.assertRaises(OntologyError):
            decisions.decide(
                "Rebuild the index",
                "Different rationale entirely.",
                source_id=source.id,
                id="decision:step-1",
            )

        self.assertEqual(len(graph.nodes), node_count)
        self.assertEqual(len(graph.edges), edge_count)
        self.assertEqual(len(decisions.records), record_count)

    def test_retry_after_attempt_bump_gets_its_own_identity(self):
        """Mirrors Runner._execute_task's real usage: id includes the attempt
        number, so a genuine retry with new content is simply a new id, not
        a collision."""
        graph, source = _graph_with_source()
        decisions = DecisionLog(graph)
        v1 = decisions.decide(
            "Fetch shard manifest",
            "First attempt.",
            source_id=source.id,
            id="plan|fetch-shard-manifest|v1",
        )
        v2 = decisions.decide(
            "Fetch shard manifest",
            "Retried with backoff.",
            source_id=source.id,
            id="plan|fetch-shard-manifest|v2",
        )
        self.assertNotEqual(v1.id, v2.id)


class AtomicRollbackTest(unittest.TestCase):
    def test_bad_supported_by_claim_leaves_nothing_behind(self):
        graph, source = _graph_with_source()
        decisions = DecisionLog(graph)
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)

        with self.assertRaises(OntologyError):
            decisions.decide(
                "Rebuild the index",
                "Bounded memory beats build time.",
                source_id=source.id,
                supported_by=["claim:does-not-exist"],
            )

        self.assertEqual(len(graph.nodes), node_count)
        self.assertEqual(len(graph.edges), edge_count)
        self.assertEqual(graph.type_counts(live_only=False)["decision"], 0)

    def test_bad_produces_entity_leaves_nothing_behind(self):
        graph, source = _graph_with_source()
        decisions = DecisionLog(graph)
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)

        with self.assertRaises(OntologyError):
            decisions.decide(
                "Rebuild the index",
                "Bounded memory beats build time.",
                source_id=source.id,
                produces=["entity:does-not-exist"],
            )

        self.assertEqual(len(graph.nodes), node_count)
        self.assertEqual(len(graph.edges), edge_count)

    def test_bad_supersedes_target_leaves_nothing_behind(self):
        graph, source = _graph_with_source()
        decisions = DecisionLog(graph)
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)

        with self.assertRaises(OntologyError):
            decisions.decide(
                "Rebuild the index",
                "Bounded memory beats build time.",
                source_id=source.id,
                supersedes="decision:does-not-exist",
            )

        self.assertEqual(len(graph.nodes), node_count)
        self.assertEqual(len(graph.edges), edge_count)

    def test_bad_assumption_leaves_nothing_behind(self):
        graph, source = _graph_with_source()
        decisions = DecisionLog(graph)
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)

        with self.assertRaises(OntologyError):
            decisions.decide(
                "Rebuild the index",
                "Bounded memory beats build time.",
                source_id=source.id,
                assumptions=["assumption:does-not-exist"],
            )

        self.assertEqual(len(graph.nodes), node_count)
        self.assertEqual(len(graph.edges), edge_count)

    def test_partial_edges_from_a_multi_item_list_are_rolled_back(self):
        """First supported_by claim is real, second is not: the edges from
        the first must not survive the failure on the second."""
        graph, source = _graph_with_source()
        good_claim = graph.add_node(
            NodeType.CLAIM,
            "A real claim",
            provenance=Provenance(source_id=source.id),
        )
        graph.add_edge(good_claim.id, EdgeType.DERIVED_FROM, source.id)
        decisions = DecisionLog(graph)
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)

        with self.assertRaises(OntologyError):
            decisions.decide(
                "Rebuild the index",
                "Bounded memory beats build time.",
                source_id=source.id,
                supported_by=[good_claim.id, "claim:does-not-exist"],
            )

        self.assertEqual(len(graph.nodes), node_count)
        self.assertEqual(len(graph.edges), edge_count)
        self.assertFalse(
            [e for e in graph.edges.values() if e.type is EdgeType.SUPPORTS]
        )

    def test_supersede_wiring_is_rolled_back_together_with_the_edge(self):
        graph, source = _graph_with_source()
        decisions = DecisionLog(graph)
        old = decisions.decide(
            "Original plan", "Because reasons.", source_id=source.id
        )

        with self.assertRaises(OntologyError):
            decisions.decide(
                "Replacement plan",
                "Better reasons.",
                source_id=source.id,
                supersedes=old.id,
                produces=["entity:does-not-exist"],
            )

        self.assertNotIn("superseded_by", old.attrs)
        self.assertEqual(graph.type_counts(live_only=False)["decision"], 1)


class PreexistingEdgesSurviveRollbackTest(unittest.TestCase):
    """A rollback must only discard what this call itself created -- an
    edge or node another write already put in place is left alone, mirroring
    AssumptionLedger.reject's edge_already_existed guard."""

    def test_unrelated_assumption_survives_a_failing_decide_call(self):
        graph, source = _graph_with_source()
        ledger = AssumptionLedger(graph)
        assumption = ledger.assume("Some ground truth")
        decisions = DecisionLog(graph)

        with self.assertRaises(OntologyError):
            decisions.decide(
                "Faulty decision",
                "Rationale.",
                source_id=source.id,
                assumptions=["assumption:does-not-exist"],
            )
        self.assertEqual(graph.type_counts(live_only=False)["decision"], 0)
        self.assertIs(graph.assumptions[assumption.id], assumption)


class DurableAcrossAFreshDecisionLogTest(unittest.TestCase):
    """A DecisionLog's own `records`/in-memory bookkeeping resets to empty
    when a new instance is built over the same graph -- exactly what a
    restored Runner does when no `decisions=` is supplied. Identity has to
    be enforced from graph state, not that instance-local history, or a
    reconstruction silently loses both "always fresh" and the idempotency
    contract."""

    def test_same_title_after_new_decisionlog_still_gets_a_fresh_id(self):
        graph, source = _graph_with_source()
        first = DecisionLog(graph).decide(
            "Use PostgreSQL", "It fits our access patterns.", source_id=source.id
        )
        second = DecisionLog(graph).decide(
            "Use PostgreSQL", "It fits our access patterns.", source_id=source.id
        )
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(graph.type_counts(live_only=False)["decision"], 2)
        # And the label/rationale of the first decision was never touched.
        self.assertEqual(graph.node(first.id).attrs["rationale"],
                          "It fits our access patterns.")

    def test_explicit_id_exact_retry_after_new_decisionlog_is_a_true_no_op(self):
        graph, source = _graph_with_source()
        DecisionLog(graph).decide(
            "Rebuild the index",
            "Bounded memory beats build time.",
            source_id=source.id,
            id="decision:step-1",
        )
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)

        again = DecisionLog(graph).decide(
            "Rebuild the index",
            "Bounded memory beats build time.",
            source_id=source.id,
            id="decision:step-1",
        )

        self.assertEqual(again.id, "decision:step-1")
        self.assertEqual(len(graph.nodes), node_count)
        self.assertEqual(len(graph.edges), edge_count)

    def test_explicit_id_different_payload_after_new_decisionlog_fails_closed(self):
        graph, source = _graph_with_source()
        DecisionLog(graph).decide(
            "Rebuild the index",
            "Bounded memory beats build time.",
            source_id=source.id,
            id="decision:step-1",
        )
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)

        with self.assertRaises(OntologyError):
            DecisionLog(graph).decide(
                "Rebuild the index",
                "A different rationale entirely.",
                source_id=source.id,
                id="decision:step-1",
            )

        self.assertEqual(len(graph.nodes), node_count)
        self.assertEqual(len(graph.edges), edge_count)
        self.assertEqual(
            graph.node("decision:step-1").attrs["rationale"],
            "Bounded memory beats build time.",
        )

    def test_explicit_id_equal_to_an_auto_minted_decision_is_refused(self):
        graph, source = _graph_with_source()
        decisions = DecisionLog(graph)
        auto = decisions.decide(
            "Use PostgreSQL", "It fits our access patterns.", source_id=source.id
        )

        with self.assertRaises(OntologyError):
            decisions.decide(
                "Use PostgreSQL",
                "A completely different rationale.",
                source_id=source.id,
                id=auto.id,
            )

        self.assertEqual(
            graph.node(auto.id).attrs["rationale"], "It fits our access patterns."
        )
        self.assertEqual(graph.type_counts(live_only=False)["decision"], 1)

    def test_explicit_id_round_trips_through_a_snapshot(self):
        """The idempotency-key metadata lives in node attrs, so it has to
        survive to_dict/from_dict, not just object reconstruction."""
        graph, source = _graph_with_source()
        DecisionLog(graph).decide(
            "Rebuild the index",
            "Bounded memory beats build time.",
            source_id=source.id,
            id="decision:step-1",
        )
        restored = ContextGraph.from_dict(graph.to_dict())

        # Exact retry over the restored graph is a no-op.
        again = DecisionLog(restored).decide(
            "Rebuild the index",
            "Bounded memory beats build time.",
            source_id=source.id,
            id="decision:step-1",
        )
        self.assertEqual(again.id, "decision:step-1")
        self.assertEqual(len(restored.nodes), len(graph.nodes))

        # A mismatched retry over the restored graph still fails closed.
        with self.assertRaises(OntologyError):
            DecisionLog(restored).decide(
                "Rebuild the index",
                "Something else.",
                source_id=source.id,
                id="decision:step-1",
            )


class DecisionNodeIsNeverMergedThroughAddNodeTest(unittest.TestCase):
    """GRAPH.md rule 9: a decision has no repeat-observation sense, unlike a
    claim or entity. `add_node` must refuse a second call for an existing
    decision id even when type and label match, closing the path a direct
    (non-DecisionLog) caller could otherwise use to rewrite a decision's
    rationale in place."""

    def test_add_node_refuses_to_merge_an_existing_decision(self):
        graph, source = _graph_with_source()
        decision = DecisionLog(graph).decide(
            "Use PostgreSQL", "It fits our access patterns.", source_id=source.id
        )

        with self.assertRaises(OntologyError):
            graph.add_node(
                NodeType.DECISION,
                "Use PostgreSQL",
                id=decision.id,
                attrs={"rationale": "A silently rewritten rationale."},
            )

        self.assertEqual(
            graph.node(decision.id).attrs["rationale"],
            "It fits our access patterns.",
        )


class DecisionMintingIsRestrictedToDecideTest(unittest.TestCase):
    """GRAPH.md: only `DecisionLog.decide()` (and `from_dict`'s restoration
    path) may mint a brand-new decision node. Without this, a caller could
    create a decision directly through `add_node` with hand-picked
    `decision_identity`/`decision_payload_digest` attrs and have a later
    `decide()` call trust it as a legitimate prior event -- exactly the
    forgery the second independent review found."""

    def test_add_node_cannot_mint_a_fresh_decision_at_all(self):
        graph, source = _graph_with_source()
        with self.assertRaises(OntologyError):
            graph.add_node(
                NodeType.DECISION,
                "Use PostgreSQL",
                id="decision:forged",
                attrs={
                    "rationale": "forged",
                    "decision_identity": "explicit",
                    "decision_payload_digest": "whatever",
                },
                provenance=Provenance(source_id=source.id),
            )
        self.assertNotIn("decision:forged", graph.nodes)

    def test_a_forgery_attempt_cannot_later_be_accepted_as_a_retry(self):
        """Granting for a moment that a forged node existed under this id
        (it cannot, per the test above): the end-to-end claim the review
        raised -- that `decide()` would later accept it as a retry -- is
        closed because the id was never actually claimed. `decide()` sees
        no existing node and mints a genuine decision instead of returning
        a forged stand-in."""
        graph, source = _graph_with_source()
        with self.assertRaises(OntologyError):
            graph.add_node(
                NodeType.DECISION,
                "Use PostgreSQL",
                id="decision:forged",
                attrs={
                    "rationale": "totally different content",
                    "decision_identity": "explicit",
                    "decision_payload_digest": "forged-digest",
                },
                provenance=Provenance(source_id=source.id),
            )
        real = DecisionLog(graph).decide(
            "Use PostgreSQL",
            "It fits our access patterns.",
            source_id=source.id,
            id="decision:forged",
        )
        self.assertEqual(real.attrs["rationale"], "It fits our access patterns.")


class TamperedSnapshotIdentityMetadataFailsClosedTest(unittest.TestCase):
    """A snapshot is untrusted input: `from_dict` must not simply believe a
    decision's `decision_identity`/`decision_payload_digest` attrs, or a
    hand-edited file could make one decision masquerade as another for a
    future explicit-id retry."""

    def test_a_digest_that_disagrees_with_the_nodes_real_content_is_refused(self):
        graph, source = _graph_with_source()
        DecisionLog(graph).decide(
            "Rebuild the index",
            "Bounded memory beats build time.",
            source_id=source.id,
            id="decision:step-1",
        )
        payload = graph.to_dict()
        for row in payload["nodes"]:
            if row["id"] == "decision:step-1":
                row["attrs"]["decision_payload_digest"] = "not-the-real-digest"

        with self.assertRaisesRegex(SnapshotError, "does not match"):
            ContextGraph.from_dict(payload)

    def test_flipping_an_auto_decision_to_explicit_with_a_forged_digest_is_refused(self):
        """The other half of the auto-to-explicit takeover: even at snapshot
        load time, relabeling an auto-minted decision as an explicit one is
        only as good as its digest -- a fabricated digest that does not
        match the node's real rationale/edges is caught on load, not
        trusted until a later `decide()` call is fooled by it."""
        graph, source = _graph_with_source()
        auto = DecisionLog(graph).decide(
            "Use PostgreSQL", "It fits our access patterns.", source_id=source.id
        )
        payload = graph.to_dict()
        for row in payload["nodes"]:
            if row["id"] == auto.id:
                row["attrs"]["decision_identity"] = "explicit"
                row["attrs"]["decision_payload_digest"] = "forged-to-claim-something-else"

        with self.assertRaisesRegex(SnapshotError, "auto-minted namespace"):
            ContextGraph.from_dict(payload)

    def test_flipping_an_auto_decision_to_explicit_with_the_correct_digest_is_still_refused(self):
        """The sharper form of the takeover: an auto-minted decision's
        stored digest is already *correct* for its real content, so tamper
        that leaves the digest untouched and only flips
        `decision_identity` to `"explicit"` cannot be caught by a digest
        comparison at all -- there is nothing wrong for it to disagree with.
        What still gives it away is that the id itself was never something
        `decide()` could have handed out as an explicit id: it is exactly
        what auto-minting produces for this title."""
        graph, source = _graph_with_source()
        auto = DecisionLog(graph).decide(
            "Use PostgreSQL", "It fits our access patterns.", source_id=source.id
        )
        payload = graph.to_dict()
        for row in payload["nodes"]:
            if row["id"] == auto.id:
                self.assertEqual(row["attrs"]["decision_identity"], "auto")
                row["attrs"]["decision_identity"] = "explicit"
                # decision_payload_digest is left exactly as decide() wrote it.

        with self.assertRaisesRegex(SnapshotError, "auto-minted namespace"):
            ContextGraph.from_dict(payload)

    def test_a_source_and_cite_swap_produces_a_different_fingerprint(self):
        """source_id is a decision's provenance, not just another citation.
        Swapping which id is the primary source and which is an additional
        cite changes the real, agreed-upon payload -- the node's actual
        provenance still names the original source -- so it must not
        fingerprint identically to the unswapped call."""
        graph = ContextGraph()
        graph.build = 1
        source_a = graph.add_node(
            NodeType.SOURCE, "Source A", attrs={"origin": "fixture", "retrieved_at": "fixture"}
        )
        source_b = graph.add_node(
            NodeType.SOURCE, "Source B", attrs={"origin": "fixture", "retrieved_at": "fixture"}
        )
        DecisionLog(graph).decide(
            "Use PostgreSQL",
            "It fits our access patterns.",
            source_id=source_a.id,
            cites=[source_b.id],
            id="decision:step-1",
        )
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)

        with self.assertRaises(OntologyError):
            DecisionLog(graph).decide(
                "Use PostgreSQL",
                "It fits our access patterns.",
                source_id=source_b.id,
                cites=[source_a.id],
                id="decision:step-1",
            )

        self.assertEqual(len(graph.nodes), node_count)
        self.assertEqual(len(graph.edges), edge_count)
        self.assertEqual(graph.node("decision:step-1").provenance.source_id, source_a.id)


class ExplicitIdCannotClaimTheAutoMintedNamespaceTest(unittest.TestCase):
    """Reserving the auto-minted namespace prospectively: `decide()` must
    refuse a brand-new explicit id that collides with what auto-minting
    could have produced for the same title, or the structural check that
    protects an *existing* auto-minted decision from a mode-flip would have
    to tolerate a legitimate false positive."""

    def test_an_explicit_id_matching_the_auto_pattern_for_its_title_is_refused(self):
        graph, source = _graph_with_source()
        auto_id = DecisionLog(graph).decide(
            "Use PostgreSQL", "It fits our access patterns.", source_id=source.id
        ).id

        other_graph, other_source = _graph_with_source()
        with self.assertRaises(OntologyError):
            DecisionLog(other_graph).decide(
                "Use PostgreSQL",
                "A different rationale.",
                source_id=other_source.id,
                id=auto_id,
            )
        self.assertNotIn(auto_id, other_graph.nodes)

    def test_the_reservation_is_a_stable_prefix_not_a_bound_tied_to_graph_size(self):
        """Regression for the exact defect a bounded search had: an id that
        merely *resembles* what the old auto-mint scheme produced, but does
        not lie in the reserved `decision:auto:` namespace, is an ordinary
        explicit id -- accepting it must not depend on how large the graph
        happens to be, so it cannot later be rejected purely because the
        graph grew (by this very write, or a snapshot round-trip of it)."""
        graph, source = _graph_with_source()
        candidate = slug("Use PostgreSQL|3", "decision")
        self.assertFalse(is_auto_minted_decision_id(candidate))

        first = DecisionLog(graph).decide(
            "Use PostgreSQL",
            "It fits our access patterns.",
            source_id=source.id,
            id=candidate,
        )
        self.assertEqual(first.id, candidate)

        # The write itself grew the graph; an unstable, size-bound check
        # would now see a different bound than it did a moment ago.
        restored = ContextGraph.from_dict(graph.to_dict())
        self.assertIn(candidate, restored.nodes)

        again = DecisionLog(restored).decide(
            "Use PostgreSQL",
            "It fits our access patterns.",
            source_id=source.id,
            id=candidate,
        )
        self.assertEqual(again.id, candidate)


class DecisionToDecisionDependsOnDoesNotPolluteIdentityTest(unittest.TestCase):
    """GRAPH.md's ontology legalizes `depends_on` for both
    decision->assumption (`decide()`'s own `assumptions` parameter) and
    decision->decision (task dependencies, wired directly through
    `add_edge` after `decide()` returns -- see `Runner._execute_task`). A
    decision's fingerprint must track only the former, or wiring an
    ordinary task dependency onto a decision would make its own recorded
    identity disagree with its real content."""

    def test_a_decision_to_decision_dependency_survives_a_snapshot_round_trip(self):
        graph, source = _graph_with_source()
        decisions = DecisionLog(graph)
        upstream = decisions.decide(
            "Provision the cluster", "Needed first.", source_id=source.id
        )
        downstream = decisions.decide(
            "Load the dataset",
            "Needs the cluster.",
            source_id=source.id,
            id="decision:load-dataset",
        )
        graph.add_edge(downstream.id, EdgeType.DEPENDS_ON, upstream.id)

        restored = ContextGraph.from_dict(graph.to_dict())
        self.assertEqual(len(restored.nodes), len(graph.nodes))
        self.assertEqual(len(restored.edges), len(graph.edges))

        # An exact-content retry against the now-dependency-bearing decision
        # is still recognized as the same decision.
        again = DecisionLog(restored).decide(
            "Load the dataset",
            "Needs the cluster.",
            source_id=source.id,
            id="decision:load-dataset",
        )
        self.assertEqual(again.id, downstream.id)


class ExplicitDecisionContentCannotGrowAfterDecideTest(unittest.TestCase):
    """The fingerprint covers the edges decide() writes, and is checked
    against the node's actual edges. An edge of one of those kinds added
    afterwards used to be accepted silently -- and then the correct snapshot
    failed to load as "tampered with or corrupted", and an identical retry
    of the original decide() was refused as different content."""

    def setUp(self):
        self.graph, self.source = _graph_with_source()
        self.ledger = AssumptionLedger(self.graph)
        self.log = DecisionLog(self.graph)
        self.ground = self.ledger.assume("The shard count is stable")

    def decide(self, **extra):
        return self.log.decide(
            "Partition the rebuild",
            "Bounded memory.",
            source_id=self.source.id,
            id="decision:partition",
            **extra,
        )

    def test_a_later_dependency_on_an_assumption_is_refused_where_it_is_added(self):
        decision = self.decide()
        edges = len(self.graph.edges)
        with self.assertRaisesRegex(OntologyError, "pass it to DecisionLog.decide"):
            self.ledger.depends(decision.id, self.ground.id)
        self.assertEqual(len(self.graph.edges), edges)
        # ...so what was saved still loads, and still retries.
        restored = ContextGraph.from_dict(self.graph.to_dict())
        self.assertIs(DecisionLog(restored).decide(
            "Partition the rebuild",
            "Bounded memory.",
            source_id=self.source.id,
            id="decision:partition",
        ).id, decision.id)

    def test_the_same_dependency_passed_to_decide_is_its_own_content(self):
        decision = self.decide(assumptions=[self.ground.id])
        ContextGraph.from_dict(self.graph.to_dict())
        # Repeating an edge the decision already has is not new content.
        self.ledger.depends(decision.id, self.ground.id)
        ContextGraph.from_dict(self.graph.to_dict())

    def test_auto_minted_decisions_are_not_frozen_by_this(self):
        decision = self.log.decide("Auto", "No id.", source_id=self.source.id)
        self.ledger.depends(decision.id, self.ground.id)
        ContextGraph.from_dict(self.graph.to_dict())

    def test_a_decision_is_not_accepted_as_an_assumption(self):
        upstream = self.log.decide("Upstream", "r", source_id=self.source.id, id="decision:up")
        nodes, edges = len(self.graph.nodes), len(self.graph.edges)
        with self.assertRaisesRegex(OntologyError, "takes assumption ids"):
            self.decide(assumptions=[upstream.id])
        self.assertEqual((len(self.graph.nodes), len(self.graph.edges)), (nodes, edges))


class ProjectionRewindsExplicitDecisionDigestTest(unittest.TestCase):
    """as_of_graph keeps a decision by its own date and drops an edge whose
    other end came later. The stored digest still covered that edge, so the
    loader rejected the projection as tampered and the time-travel query
    crashed instead of answering."""

    def build(self, explicit):
        graph = ContextGraph()
        graph.build = 1
        early = graph.add_node(
            NodeType.SOURCE, "2024 review",
            attrs={"origin": "review", "retrieved_at": "2024-03-01"},
        )
        late = graph.add_node(
            NodeType.SOURCE, "2025 review",
            attrs={"origin": "review", "retrieved_at": "2025-03-01"},
        )
        claim = graph.add_node(
            NodeType.CLAIM, "later evidence", provenance=Provenance(source_id=late.id)
        )
        graph.add_edge(claim.id, EdgeType.DERIVED_FROM, late.id)
        extra = {"id": "decision:choose-x"} if explicit else {}
        decision = DecisionLog(graph).decide(
            "Choose X", "because", source_id=early.id, supported_by=[claim.id], **extra
        )
        return graph, decision, claim

    def test_the_projection_answers_and_keeps_the_decision_without_the_later_edge(self):
        from contextmesh.temporal import as_of_graph

        graph, decision, claim = self.build(explicit=True)
        projection = as_of_graph(graph, "2024-06-01")
        self.assertIn(decision.id, projection.nodes)
        self.assertNotIn(claim.id, projection.nodes)
        # A graph the ordinary loader accepts, not a special case of it.
        ContextGraph.from_dict(projection.to_dict())
        # And the live graph's own record is untouched.
        self.assertEqual(
            graph.node(decision.id).attrs["decision_payload_digest"],
            ContextGraph.from_dict(graph.to_dict())
            .node(decision.id)
            .attrs["decision_payload_digest"],
        )

    def test_a_horizon_that_keeps_every_edge_keeps_the_digest(self):
        from contextmesh.temporal import as_of_graph

        graph, decision, _ = self.build(explicit=True)
        projection = as_of_graph(graph, "2030-01-01")
        self.assertEqual(
            projection.node(decision.id).attrs["decision_payload_digest"],
            graph.node(decision.id).attrs["decision_payload_digest"],
        )


if __name__ == "__main__":
    unittest.main()
