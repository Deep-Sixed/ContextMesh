"""Hybrid retrieval: BM25 candidates, graph evidence, refusal on thin evidence."""

import os
import tempfile
import unittest
from pathlib import Path

from contextmesh.hybrid import (
    REFUSED_LOW_COVERAGE,
    REFUSED_NO_MATCH,
    HybridRetriever,
)
from contextmesh.ingest import build_folder
from contextmesh.model import EdgeType, NodeType, Provenance
from contextmesh.refresh import refresh_folder

CACHE = """# Cache

The Cache Layer is keyed per tenant.
The Cache Layer holds entries for an hour.
"""
INDEX = """# Index

The Index Builder writes shards every night.
The Index Builder retries a failed shard three times.
"""


def write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    os.utime(path, (1767225600, 1767225600))


class HybridFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write(self.root, "cache.md", CACHE)
        write(self.root, "index.md", INDEX)
        self.built = build_folder(self.root)
        self.retriever = HybridRetriever(self.built.graph, self.built.resolver)

    def tearDown(self):
        self._tmp.cleanup()

    def label(self, node_id):
        return self.built.graph.node(node_id).label


class AnswerTest(HybridFixture):
    def test_it_answers_with_the_claim_and_its_evidence_path(self):
        answer = self.retriever.ask("How many times does the Index Builder retry a failed shard?")
        self.assertTrue(answer.resolved)
        self.assertEqual(
            self.label(answer.answer_id), "The Index Builder retries a failed shard three times."
        )
        graph = self.built.graph
        claim = graph.node(answer.answer_id)
        # The path ends at the source the claim was lifted from, over live edges.
        self.assertEqual(answer.path[-1], ("derived_from", claim.provenance.source_id))
        nodes = [n for _, n in answer.path]
        self.assertIn(answer.answer_id, nodes)
        for (_, a), (edge, b) in zip(answer.path, answer.path[1:]):
            if edge == "mentions":
                self.assertTrue(graph.out_edges(b, [EdgeType.MENTIONS]))
                self.assertIn(a, [e.dst for e in graph.out_edges(b, [EdgeType.MENTIONS])])
            elif edge == "derived_from":
                self.assertIn(b, [e.dst for e in graph.out_edges(a, [EdgeType.DERIVED_FROM])])
        self.assertIn("index.md:", self.retriever.render(answer))

    def test_a_question_that_names_an_entity_starts_its_path_there(self):
        # A capitalised name needs a second file before it is an entity.
        write(self.root, "ops.md", "# Ops\n\nThe Index Builder pages the on-call engineer.\n")
        built = build_folder(self.root)
        self.assertIsNotNone(built.resolver.match("Index Builder"))
        answer = HybridRetriever(built.graph, built.resolver).ask(
            "Where does the Index Builder write shards?"
        )
        first = built.graph.node(answer.path[0][1])
        self.assertIs(first.type, NodeType.ENTITY)
        self.assertEqual(first.label, "Index Builder")
        self.assertEqual(answer.path[1], ("mentions", answer.answer_id))

    def test_tokens_pay_for_every_node_returned(self):
        answer = self.retriever.ask("How long does the Cache Layer hold entries?")
        nodes = [n for _, n in answer.path]
        expected = sum(self.retriever._text_tokens(self.label(n)) for n in nodes)
        for extra in answer.answer_ids[1:]:
            claim = self.built.graph.node(extra)
            expected += self.retriever._text_tokens(claim.label)
            expected += self.retriever._text_tokens(self.label(claim.provenance.source_id))
        self.assertEqual(answer.tokens, expected)


class RefusalTest(HybridFixture):
    def test_no_shared_word_is_refused(self):
        answer = self.retriever.ask("Who won the 2018 World Cup?")
        self.assertFalse(answer.resolved)
        self.assertIn(answer.refused, (REFUSED_NO_MATCH, REFUSED_LOW_COVERAGE))

    def test_thin_evidence_is_refused_with_its_coverage(self):
        # "cache" matches, but most of the question is about things the corpus
        # has never mentioned: a path to some cache claim is not an answer.
        answer = self.retriever.ask("What refund does the cache owe enterprise customers yearly?")
        self.assertFalse(answer.resolved)
        self.assertEqual(answer.refused, REFUSED_LOW_COVERAGE)
        self.assertIn("covers", answer.detail)

    def test_the_threshold_is_configurable(self):
        lenient = HybridRetriever(self.built.graph, self.built.resolver, min_coverage=0.0)
        self.assertTrue(
            lenient.ask("What refund does the cache owe enterprise customers yearly?").resolved
        )


class RunnersUpTest(HybridFixture):
    def test_runners_up_respect_max_answers_and_margin(self):
        question = "What does the Cache Layer do?"
        one = HybridRetriever(self.built.graph, self.built.resolver, max_answers=1)
        self.assertEqual(len(one.ask(question).answer_ids), 1)
        wide = HybridRetriever(self.built.graph, self.built.resolver, max_answers=3, margin=10.0)
        ids = wide.ask(question).answer_ids
        self.assertLessEqual(len(ids), 3)
        self.assertEqual(len(ids), len(set(ids)))
        tight = HybridRetriever(self.built.graph, self.built.resolver, max_answers=3, margin=0.0)
        tight_answer = tight.ask(question)
        best = max(c.score for c in tight_answer.candidates)
        for claim_id in tight_answer.answer_ids:
            score = next(c.score for c in tight_answer.candidates if c.claim_id == claim_id)
            self.assertEqual(score, best)


class LiveEvidenceTest(HybridFixture):
    """A retired claim is never served as current evidence."""

    def test_a_rejected_claim_is_not_retrieved(self):
        question = "How many times does the Index Builder retry a failed shard?"
        graph = self.built.graph
        target = self.retriever.ask(question).answer_id
        assumption = graph.out_edges(target, [EdgeType.DEPENDS_ON])[0].dst
        evidence = graph.add_node(
            NodeType.EVIDENCE, "index.md no longer says it retries",
            attrs={"kind": "file-reread"},
            provenance=Provenance(source_id=graph.node(target).provenance.source_id,
                                  extractor="test"),
        )
        self.built.ledger.reject(assumption, evidence_id=evidence.id)
        again = self.retriever.ask(question)
        self.assertNotEqual(again.answer_id, target)
        self.assertNotIn(target, again.answer_ids)
        self.assertNotIn(target, [c.claim_id for c in again.candidates])

    def test_a_refresh_retires_old_text_and_serves_new_text(self):
        question = "How many times does the Index Builder retry a failed shard?"
        old = self.retriever.ask(question).answer_id
        write(self.root, "index.md", INDEX.replace("three times", "five times"))
        refresh_folder(self.built.graph, self.built.resolver, self.built.ledger, self.root)
        answer = self.retriever.ask(question)
        self.assertNotEqual(answer.answer_id, old)
        self.assertEqual(
            self.label(answer.answer_id), "The Index Builder retries a failed shard five times."
        )
        self.assertTrue(self.built.graph.node(answer.answer_id).live)


class ReadOnlyTest(HybridFixture):
    def test_asking_changes_nothing_and_order_does_not_matter(self):
        graph, resolver = self.built.graph, self.built.resolver
        before_graph = graph.to_json()
        before_resolver = resolver.to_json()
        questions = [
            "How long does the Cache Layer hold entries?",
            "Where does the Index Builder write shards?",
            "What refund does the cache owe enterprise customers yearly?",
        ]
        forward = [self.retriever.ask(q).to_dict() for q in questions]
        self.assertEqual(graph.to_json(), before_graph)
        self.assertEqual(resolver.to_json(), before_resolver)
        fresh = HybridRetriever(graph, resolver)
        backward = [fresh.ask(q).to_dict() for q in reversed(questions)]
        self.assertEqual(forward, list(reversed(backward)))


if __name__ == "__main__":
    unittest.main()
