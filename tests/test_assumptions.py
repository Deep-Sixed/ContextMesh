"""Selective invalidation: exactly the dependent work falls, and nothing else."""

import unittest

from contextmesh.assumptions import AssumptionError, AssumptionLedger
from contextmesh.decisions import DecisionLog
from contextmesh.graph import ContextGraph
from contextmesh.model import AssumptionStatus, EdgeType, NodeType, Provenance


def scenario():
    """Two independent branches, one of which stands on an assumption."""
    graph = ContextGraph()
    graph.build = 1
    ledger = AssumptionLedger(graph)
    decisions = DecisionLog(graph)

    source = graph.add_node(
        NodeType.SOURCE,
        "Capacity model",
        attrs={"origin": "fixture", "retrieved_at": "fixture"},
    )
    other_source = graph.add_node(
        NodeType.SOURCE,
        "Reranker evaluation",
        attrs={"origin": "fixture", "retrieved_at": "fixture"},
    )
    claim = graph.add_node(
        NodeType.CLAIM,
        "Shards stay under four gigabytes during rebuild",
        provenance=Provenance(source_id=source.id),
    )
    graph.add_edge(claim.id, EdgeType.DERIVED_FROM, source.id)
    unrelated_claim = graph.add_node(
        NodeType.CLAIM,
        "Reranking adds ninety milliseconds at p95",
        provenance=Provenance(source_id=other_source.id),
    )
    graph.add_edge(unrelated_claim.id, EdgeType.DERIVED_FROM, other_source.id)

    artefact = graph.add_node(
        NodeType.ENTITY,
        "Partitioned Rebuild",
        attrs={"canonical": "Partitioned Rebuild", "aliases": []},
    )
    assumption = ledger.assume("Shard count grows linearly with corpus size")

    dependent = decisions.decide(
        "Rebuild the index in partitions",
        "Bounded memory beats build time.",
        source_id=source.id,
        supported_by=[claim.id],
        assumptions=[assumption.id],
        produces=[artefact.id],
    )
    unrelated = decisions.decide(
        "Rerank only the top fifty candidates",
        "Keeps p95 inside budget.",
        source_id=other_source.id,
        supported_by=[unrelated_claim.id],
    )
    return graph, ledger, decisions, assumption, dependent, unrelated, artefact, unrelated_claim


class BlastRadiusTest(unittest.TestCase):
    def setUp(self):
        (
            self.graph,
            self.ledger,
            self.decisions,
            self.assumption,
            self.dependent,
            self.unrelated,
            self.artefact,
            self.unrelated_claim,
        ) = scenario()

    def test_radius_follows_depends_on_backwards(self):
        radius = self.ledger.blast_radius(self.assumption.id)
        self.assertIn(self.dependent.id, radius)

    def test_radius_follows_produces_forwards(self):
        radius = self.ledger.blast_radius(self.assumption.id)
        self.assertIn(self.artefact.id, radius)

    def test_radius_excludes_the_unrelated_branch(self):
        radius = self.ledger.blast_radius(self.assumption.id)
        self.assertNotIn(self.unrelated.id, radius)
        self.assertNotIn(self.unrelated_claim.id, radius)

    def test_radius_explains_itself(self):
        radius = self.ledger.blast_radius(self.assumption.id)
        chain = radius[self.artefact.id]
        self.assertIn("depends_on", " ".join(chain))
        self.assertIn("produces", " ".join(chain))


class RejectionTest(unittest.TestCase):
    def setUp(self):
        (
            self.graph,
            self.ledger,
            self.decisions,
            self.assumption,
            self.dependent,
            self.unrelated,
            self.artefact,
            self.unrelated_claim,
        ) = scenario()
        self.evidence = self.graph.add_node(
            NodeType.EVIDENCE,
            "One tenant held 31% of chunks in a single shard",
            attrs={"kind": "disproof"},
        )
        self.report = self.ledger.reject(
            self.assumption.id,
            evidence_id=self.evidence.id,
            replacement="Shards are keyed on tenant as well as corpus position",
        )

    def test_the_assumption_is_rejected_not_deleted(self):
        self.assertIs(
            self.graph.assumptions[self.assumption.id].status, AssumptionStatus.REJECTED
        )
        self.assertIn(self.assumption.id, self.graph.nodes)

    def test_dependent_work_is_invalidated(self):
        self.assertTrue(self.graph.node(self.dependent.id).invalidated)
        self.assertTrue(self.graph.node(self.artefact.id).invalidated)

    def test_unrelated_work_is_preserved(self):
        self.assertFalse(self.graph.node(self.unrelated.id).invalidated)
        self.assertFalse(self.graph.node(self.unrelated_claim.id).invalidated)
        self.assertIn(self.unrelated.id, self.report.preserved)

    def test_the_report_proves_both_halves(self):
        self.assertEqual(self.report.blast_radius, 2)
        self.assertGreater(len(self.report.preserved), self.report.blast_radius)
        self.assertIn("depends_on", self.report.why(self.dependent.id))

    def test_evidence_is_wired_as_a_contradiction(self):
        edges = self.graph.in_edges(
            self.assumption.id, (EdgeType.CONTRADICTS,), live_only=False
        )
        self.assertEqual([e.src for e in edges], [self.evidence.id])

    def test_replacement_supersedes_the_rejected_assumption(self):
        replacement = self.graph.assumptions[self.report.replacement_id]
        self.assertEqual(replacement.supersedes, self.assumption.id)
        self.assertEqual(replacement.version, self.assumption.version + 1)

    def test_lineage_reads_oldest_first(self):
        lineage = self.ledger.lineage(self.report.replacement_id)
        self.assertEqual([a.id for a in lineage], [self.assumption.id, self.report.replacement_id])


class SupersedeTest(unittest.TestCase):
    def test_superseding_keeps_the_old_assumption(self):
        graph = ContextGraph()
        ledger = AssumptionLedger(graph)
        first = ledger.assume("Latency budget is 200ms")
        second = ledger.supersede(first.id, "Latency budget is 150ms")
        self.assertIs(graph.assumptions[first.id].status, AssumptionStatus.SUPERSEDED)
        self.assertEqual(graph.assumptions[first.id].superseded_by, second.id)
        self.assertEqual(second.version, 2)
        self.assertIn(first.id, graph.nodes)
        self.assertEqual(
            [e.dst for e in graph.out_edges(second.id, (EdgeType.SUPERSEDES,))],
            [first.id],
        )

    def test_only_active_assumptions_are_listed_as_active(self):
        graph = ContextGraph()
        ledger = AssumptionLedger(graph)
        first = ledger.assume("A")
        second = ledger.supersede(first.id, "B")
        self.assertEqual([a.id for a in ledger.active()], [second.id])


class DecisionHistoryTest(unittest.TestCase):
    def test_history_is_append_only_and_ordered(self):
        graph, ledger, decisions, *_ = scenario()
        latest = decisions.records[1]
        third = decisions.decide(
            "Rebuild the index in tenant-keyed partitions",
            "Tenant skew broke the sizing rule.",
            source_id="source:capacity" if "source:capacity" in graph.nodes else
            next(n.id for n in graph.by_type(NodeType.SOURCE)),
            supersedes=latest.decision_id,
        )
        chain = decisions.history_of(third.id)
        self.assertEqual([r.decision_id for r in chain][-1], third.id)
        self.assertIn(latest.decision_id, [r.decision_id for r in chain])
        # nothing was removed
        self.assertIn(latest.decision_id, graph.nodes)

    def test_current_excludes_superseded_decisions(self):
        graph, ledger, decisions, _, dependent, unrelated, *_ = scenario()
        source = next(n.id for n in graph.by_type(NodeType.SOURCE))
        replacement = decisions.decide(
            "Rebuild in tenant-keyed partitions",
            "Tenant skew.",
            source_id=source,
            supersedes=dependent.id,
        )
        current = {r.decision_id for r in decisions.current()}
        self.assertIn(replacement.id, current)
        self.assertNotIn(dependent.id, current)


class Rule7PreconditionTest(unittest.TestCase):
    def setUp(self):
        self.graph, self.ledger, _, self.assumption, self.dependent, *_ = scenario()
        self.evidence = self.graph.add_node(
            NodeType.EVIDENCE,
            "One tenant held 31% of chunks in a single shard",
            attrs={"kind": "disproof"},
        )

    def test_reject_requires_non_empty_evidence_id(self):
        with self.assertRaises(AssumptionError):
            self.ledger.reject(self.assumption.id, evidence_id=None)
        with self.assertRaises(AssumptionError):
            self.ledger.reject(self.assumption.id, evidence_id="")
        # Status must remain ACTIVE and node must not be invalidated
        self.assertIs(self.graph.assumptions[self.assumption.id].status, AssumptionStatus.ACTIVE)
        self.assertFalse(self.graph.node(self.assumption.id).invalidated)

    def test_reject_refuses_nonexistent_evidence_id(self):
        with self.assertRaises(AssumptionError):
            self.ledger.reject(self.assumption.id, evidence_id="nonexistent-evidence-id")
        self.assertIs(self.graph.assumptions[self.assumption.id].status, AssumptionStatus.ACTIVE)
        self.assertFalse(self.graph.node(self.assumption.id).invalidated)

    def test_reject_refuses_non_evidence_node_type(self):
        claim_node = next(n for n in self.graph.nodes.values() if n.type is NodeType.CLAIM)
        with self.assertRaises(AssumptionError):
            self.ledger.reject(self.assumption.id, evidence_id=claim_node.id)
        self.assertIs(self.graph.assumptions[self.assumption.id].status, AssumptionStatus.ACTIVE)
        self.assertFalse(self.graph.node(self.assumption.id).invalidated)

    def test_reject_refuses_invalidated_evidence(self):
        self.evidence.invalidated = True
        with self.assertRaises(AssumptionError):
            self.ledger.reject(self.assumption.id, evidence_id=self.evidence.id)
        self.assertIs(self.graph.assumptions[self.assumption.id].status, AssumptionStatus.ACTIVE)
        self.assertFalse(self.graph.node(self.assumption.id).invalidated)

    def test_reject_refuses_nonexistent_assumption(self):
        with self.assertRaises(AssumptionError):
            self.ledger.reject("nonexistent-assumption-id", evidence_id=self.evidence.id)

    def test_reject_refuses_already_rejected_assumption(self):
        self.ledger.reject(self.assumption.id, evidence_id=self.evidence.id)
        second_evidence = self.graph.add_node(
            NodeType.EVIDENCE,
            "Second contradictory evidence node",
            attrs={"kind": "disproof"},
        )
        with self.assertRaises(AssumptionError):
            self.ledger.reject(self.assumption.id, evidence_id=second_evidence.id)

    def test_reject_refuses_superseded_assumption(self):
        new_assumption = self.ledger.supersede(self.assumption.id, "Latency budget is 150ms")
        with self.assertRaises(AssumptionError):
            self.ledger.reject(self.assumption.id, evidence_id=self.evidence.id)
        # New assumption remains ACTIVE
        self.assertIs(self.graph.assumptions[new_assumption.id].status, AssumptionStatus.ACTIVE)

    def test_failure_leaves_graph_and_dependent_work_unmodified(self):
        initial_edges = len(self.graph.edges)
        with self.assertRaises(AssumptionError):
            self.ledger.reject(self.assumption.id, evidence_id="bad-id")
        self.assertIs(self.graph.assumptions[self.assumption.id].status, AssumptionStatus.ACTIVE)
        self.assertFalse(self.graph.node(self.assumption.id).invalidated)
        self.assertFalse(self.graph.node(self.dependent.id).invalidated)
        self.assertEqual(len(self.graph.edges), initial_edges)


class Rule7AtomicCommitTest(unittest.TestCase):
    def setUp(self):
        self.graph, self.ledger, _, self.assumption, self.dependent, *_ = scenario()
        self.evidence = self.graph.add_node(
            NodeType.EVIDENCE,
            "One tenant held 31% of chunks in a single shard",
            attrs={"kind": "disproof"},
        )

    def test_successful_rejection_atomically_mints_contradiction_edge_and_rejects(self):
        # Pre-state: active, not invalidated, no contradiction edge
        self.assertIs(self.graph.assumptions[self.assumption.id].status, AssumptionStatus.ACTIVE)
        self.assertFalse(self.graph.node(self.assumption.id).invalidated)
        self.assertEqual(
            self.graph.in_edges(self.assumption.id, (EdgeType.CONTRADICTS,), live_only=False),
            [],
        )

        report = self.ledger.reject(self.assumption.id, evidence_id=self.evidence.id)
        self.assertIsNotNone(report)

        # Post-state: status REJECTED, node invalidated, contradiction edge established
        self.assertIs(self.graph.assumptions[self.assumption.id].status, AssumptionStatus.REJECTED)
        self.assertTrue(self.graph.node(self.assumption.id).invalidated)
        self.assertIn(self.evidence.id, self.graph.assumptions[self.assumption.id].evidence_ids)

        edges = self.graph.in_edges(self.assumption.id, (EdgeType.CONTRADICTS,), live_only=False)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].src, self.evidence.id)
        self.assertEqual(edges[0].dst, self.assumption.id)
        self.assertEqual(edges[0].type, EdgeType.CONTRADICTS)

    def test_induced_failure_during_commit_rolls_back_without_partial_state(self):
        initial_edges = len(self.graph.edges)
        key = (self.evidence.id, EdgeType.CONTRADICTS.value, self.assumption.id)

        # Induce an exception midway through the commit block (when setting node status)
        node = self.graph.node(self.assumption.id)
        class FaultyDict(dict):
            def __setitem__(self, k, v):
                if k == "status" and v == "rejected":
                    raise RuntimeError("induced commit error")
                super().__setitem__(k, v)

        node.attrs = FaultyDict(node.attrs)
        with self.assertRaises(RuntimeError):
            self.ledger.reject(self.assumption.id, evidence_id=self.evidence.id)

        # Restore plain dict for assertions
        node.attrs = dict(node.attrs)

        # Assert clean rollback: neither contradiction edge nor REJECTED status committed
        self.assertIs(self.graph.assumptions[self.assumption.id].status, AssumptionStatus.ACTIVE)
        self.assertFalse(self.graph.node(self.assumption.id).invalidated)
        self.assertNotIn(self.evidence.id, self.graph.assumptions[self.assumption.id].evidence_ids)
        self.assertNotIn(key, self.graph._edge_key)
        self.assertEqual(
            self.graph.in_edges(self.assumption.id, (EdgeType.CONTRADICTS,), live_only=False),
            [],
        )
        self.assertEqual(len(self.graph.edges), initial_edges)


class ReplacementMustBeNewGroundTest(unittest.TestCase):
    """reject(replacement=...) used to route through assume(), whose
    same-statement no-op handed back an existing record for reject() to
    rewrite in place -- or, for the rejected statement itself, raised a
    self-edge error after the rejection had already been committed."""

    def setUp(self):
        self.graph = ContextGraph()
        self.graph.build = 1
        self.ledger = AssumptionLedger(self.graph)
        self.source = self.graph.add_node(
            NodeType.SOURCE,
            "Postmortem",
            attrs={"origin": "review", "retrieved_at": "2026-07-01"},
        )

    def witness(self, text):
        return self.graph.add_node(
            NodeType.EVIDENCE,
            text,
            attrs={"kind": "postmortem"},
            provenance=Provenance(source_id=self.source.id, recorded_at_build=1),
        ).id

    def snapshot(self, assumption_id):
        record = self.graph.assumptions[assumption_id]
        node = self.graph.node(assumption_id)
        return (
            record.status,
            record.version,
            record.supersedes,
            record.superseded_by,
            node.invalidated,
            len(self.graph.edges),
        )

    def test_replacing_with_the_rejected_statement_is_refused_before_any_write(self):
        a = self.ledger.assume("A holds")
        before = self.snapshot(a.id)
        with self.assertRaisesRegex(AssumptionError, "not new ground"):
            self.ledger.reject(a.id, evidence_id=self.witness("A broke"), replacement="A holds")
        self.assertEqual(self.snapshot(a.id), before)
        self.assertIs(self.graph.assumptions[a.id].status, AssumptionStatus.ACTIVE)

    def test_replacing_with_an_existing_rejected_statement_leaves_it_untouched(self):
        a = self.ledger.assume("A holds")
        b = self.ledger.assume("B holds")
        self.ledger.reject(b.id, evidence_id=self.witness("B broke"))
        b_before, a_before = self.snapshot(b.id), self.snapshot(a.id)
        with self.assertRaisesRegex(AssumptionError, r"not new ground.*rejected"):
            self.ledger.reject(a.id, evidence_id=self.witness("A broke"), replacement="B holds")
        self.assertEqual(self.snapshot(b.id), b_before)
        self.assertEqual(self.snapshot(a.id), a_before)

    def test_a_new_replacement_still_supersedes(self):
        a = self.ledger.assume("A holds")
        report = self.ledger.reject(
            a.id, evidence_id=self.witness("A broke"), replacement="C holds"
        )
        new = self.graph.assumptions[report.replacement_id]
        self.assertEqual((new.statement, new.version, new.supersedes), ("C holds", 2, a.id))


class ReassumeDoesNotDropEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.graph = ContextGraph()
        self.ledger = AssumptionLedger(self.graph)
        source = self.graph.add_node(
            NodeType.SOURCE, "Bench", attrs={"origin": "bench", "retrieved_at": "2026-07-01"}
        )
        self.ev = self.graph.add_node(
            NodeType.EVIDENCE,
            "linear growth observed",
            attrs={"kind": "bench"},
            provenance=Provenance(source_id=source.id),
        ).id

    def test_new_evidence_on_an_existing_statement_is_refused_not_dropped(self):
        a = self.ledger.assume("X holds")
        edges = len(self.graph.edges)
        with self.assertRaisesRegex(AssumptionError, "does not add justification"):
            self.ledger.assume("X holds", evidence_ids=[self.ev])
        self.assertEqual(self.graph.assumptions[a.id].evidence_ids, [])
        self.assertEqual(len(self.graph.edges), edges)

    def test_repeating_the_same_evidence_is_still_a_no_op(self):
        a = self.ledger.assume("X holds", evidence_ids=[self.ev])
        self.assertIs(self.ledger.assume("X holds", evidence_ids=[self.ev]), a)
        self.assertIs(self.ledger.assume("X holds"), a)


if __name__ == "__main__":
    unittest.main()
