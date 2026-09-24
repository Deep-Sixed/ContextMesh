"""PR #8A: observation enters as evidence, never as a client verdict."""

import hashlib
import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path

from contextmesh.decisions import DecisionLog
from contextmesh.evidence import (
    MAX_COLLECTION_LENGTH,
    MAX_EXTERNAL_ID_BYTES,
    MAX_METADATA_BYTES,
    MAX_METADATA_DEPTH,
    MAX_TEXT_BYTES,
    EvidenceConflictError,
    EvidenceIntake,
    EvidenceIntakeError,
    canonical_payload,
    evidence_id_for,
    submit_evidence,
    validate_metadata,
)
from contextmesh.graph import ContextGraph
from contextmesh.model import (
    Assumption,
    AssumptionStatus,
    EdgeType,
    NodeType,
    Provenance,
    slug,
)
from contextmesh_mcp import writes
from contextmesh_mcp.session import Checkpointer, Session, writer_lock


class EvidenceIntakeTest(unittest.TestCase):
    def setUp(self):
        self.graph = ContextGraph()
        self.source = self.graph.add_node(
            NodeType.SOURCE,
            "NVD feed",
            id="source:nvd",
            attrs={"origin": "nvd", "retrieved_at": "fixture"},
            provenance=Provenance(
                source_id="fixture",
                span=None,
                extractor="test",
                checks=[],
                recorded_at_build=0,
            ),
        )
        self.decision = DecisionLog(self.graph).decide(
            "Use Argon2id", "fixture", source_id=self.source.id, id="decision:argon"
        )

    def submit(self, **overrides):
        values = {
            "text": "CVE-2026-9999 affects argon2-cffi",
            "source_id": self.source.id,
            "external_id": "CVE-2026-9999",
            "metadata": {"package": "argon2-cffi", "severity": "critical"},
        }
        values.update(overrides)
        return submit_evidence(self.graph, **values)

    def test_valid_evidence_is_one_detached_evidence_node(self):
        before_edges = dict(self.graph.edges)
        receipt = self.submit()
        self.assertTrue(receipt.created)
        self.assertIs(receipt.node.type, NodeType.EVIDENCE)
        self.assertEqual(self.graph.edges, before_edges)
        self.assertEqual(receipt.node.provenance.source_id, self.source.id)

    def test_external_id_is_optional(self):
        receipt = self.submit(external_id=None)
        self.assertTrue(receipt.created)
        self.assertIsNone(receipt.node.attrs["evidence_intake"]["external_id"])

    def test_bad_text_is_refused_before_mutation(self):
        before = self.graph.to_dict()
        for value in ("", "   ", None, 3, True):
            with self.subTest(value=value), self.assertRaises(EvidenceIntakeError):
                self.submit(text=value)
            self.assertEqual(self.graph.to_dict(), before)

    def test_bad_source_is_refused_before_mutation(self):
        before = self.graph.to_dict()
        for value in ("", "source:missing", self.decision.id, 5):
            with self.subTest(value=value), self.assertRaises(EvidenceIntakeError):
                self.submit(source_id=value)
            self.assertEqual(self.graph.to_dict(), before)

    def test_bad_external_id_is_refused(self):
        for value in ("", "   ", 4, True, "bad\nvalue"):
            with self.subTest(value=value), self.assertRaises(EvidenceIntakeError):
                self.submit(external_id=value)

    def test_metadata_is_strict_recursively(self):
        bad = (
            "not-an-object",
            {1: "integer key"},
            {"nested": {2: "integer key"}},
            {"tuple": (1, 2)},
            {"set": {1, 2}},
            {"bytes": b"x"},
            {"nan": math.nan},
            {"inf": math.inf},
            {"callable": lambda: None},
        )
        for value in bad:
            with self.subTest(value=repr(value)), self.assertRaises(EvidenceIntakeError):
                self.submit(metadata=value)

    def test_oversized_text_is_refused_before_mutation(self):
        before = self.graph.to_dict()
        with self.assertRaisesRegex(EvidenceIntakeError, "byte limit"):
            self.submit(text="x" * (MAX_TEXT_BYTES + 1))
        self.assertEqual(self.graph.to_dict(), before)
        # The boundary itself is accepted: this is a limit, not an off-by-one trap.
        self.submit(text="x" * MAX_TEXT_BYTES, external_id="at-the-text-limit")

    def test_oversized_text_is_refused_before_it_is_stripped(self):
        """strip() on an oversized string allocates a copy of it before the
        limit can refuse it -- the reason external_id is bounded first. Text
        used to be stripped first."""
        stripped = []

        class Recording(str):
            def strip(self, *args):
                stripped.append(len(self))
                return super().strip(*args)

        with self.assertRaisesRegex(EvidenceIntakeError, "byte limit"):
            self.submit(text=Recording(" " * (MAX_TEXT_BYTES + 1)))
        self.assertEqual(stripped, [])
        # Blank text within the limit is still refused as blank.
        with self.assertRaisesRegex(EvidenceIntakeError, "non-empty"):
            self.submit(text=Recording("   "))
        self.assertEqual(stripped, [3])

    def test_oversized_metadata_is_refused_before_mutation(self):
        before = self.graph.to_dict()
        with self.assertRaisesRegex(EvidenceIntakeError, "byte limit"):
            self.submit(metadata={"blob": "x" * MAX_METADATA_BYTES})
        self.assertEqual(self.graph.to_dict(), before)

    def test_oversized_external_id_is_refused_before_mutation(self):
        """Unlike text and metadata, external_id had no size limit at all --
        a foreign system's identifier, not prose, so nothing bounded a
        caller from handing this the one unbounded field left in intake."""
        before = self.graph.to_dict()
        with self.assertRaisesRegex(EvidenceIntakeError, "byte limit"):
            self.submit(external_id="x" * (MAX_EXTERNAL_ID_BYTES + 1))
        self.assertEqual(self.graph.to_dict(), before)
        # The boundary itself is accepted: this is a limit, not an off-by-one trap.
        self.submit(text="at the external_id limit", external_id="x" * MAX_EXTERNAL_ID_BYTES)

    def test_metadata_is_refused_exactly_one_byte_over_the_serialized_limit(self):
        """Pin the boundary on the *combined* serialized size, not one field.

        ``{"pad": "<N x's>"}`` serializes (sorted keys, no spaces) to a fixed
        amount of JSON punctuation plus N bytes of payload; padding N so the
        whole thing lands on exactly MAX_METADATA_BYTES proves the ceiling is
        enforced on what validate_metadata actually measures — the canonical
        serialization — not on any single field in isolation.
        """
        overhead = len(
            json.dumps(
                {"pad": ""},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
        at_limit = "x" * (MAX_METADATA_BYTES - overhead)
        accepted = validate_metadata({"pad": at_limit})
        self.assertEqual(accepted, {"pad": at_limit})

        over_limit = at_limit + "x"
        with self.assertRaisesRegex(EvidenceIntakeError, "byte limit"):
            validate_metadata({"pad": over_limit})

    def test_a_single_oversized_string_is_rejected_without_serializing_everything(self):
        """Regression for the defect this limit was added to close.

        Measuring size by ``json.dumps()``-then-``len()`` would have to build
        the whole serialized string before rejecting it — so a metadata value
        that is one enormous string would force exactly the allocation the
        byte ceiling exists to prevent. This string is ~32x the limit: small
        enough to keep the test fast, large enough that a full-materialize
        implementation would visibly be doing multi-megabyte work to reject
        it rather than rejecting on the string's own length.
        """
        before = self.graph.to_dict()
        huge = "x" * (MAX_METADATA_BYTES * 32)
        with self.assertRaisesRegex(EvidenceIntakeError, "byte limit"):
            self.submit(metadata={"blob": huge})
        self.assertEqual(self.graph.to_dict(), before)

    def test_an_oversized_metadata_key_is_rejected_the_same_way(self):
        before = self.graph.to_dict()
        huge_key = "k" * (MAX_METADATA_BYTES + 1)
        with self.assertRaisesRegex(EvidenceIntakeError, "byte limit"):
            self.submit(metadata={huge_key: "value"})
        self.assertEqual(self.graph.to_dict(), before)

    def test_metadata_nesting_depth_is_bounded(self):
        value = "leaf"
        for _ in range(MAX_METADATA_DEPTH + 2):
            value = {"nested": value}
        with self.assertRaisesRegex(EvidenceIntakeError, "nests deeper"):
            self.submit(metadata=value)

    def test_metadata_collection_length_is_bounded(self):
        with self.assertRaisesRegex(EvidenceIntakeError, "item limit"):
            self.submit(metadata={"items": list(range(MAX_COLLECTION_LENGTH + 1))})
        with self.assertRaisesRegex(EvidenceIntakeError, "key limit"):
            self.submit(
                metadata={str(i): i for i in range(MAX_COLLECTION_LENGTH + 1)}
            )

    def test_metadata_is_detached_from_caller_memory(self):
        metadata = {"packages": ["argon2-cffi"], "nested": {"severity": "high"}}
        receipt = self.submit(metadata=metadata)
        digest = receipt.node.attrs["evidence_intake"]["payload_digest"]
        metadata["packages"][0] = "bcrypt"
        metadata["nested"]["severity"] = "low"
        self.assertEqual(
            receipt.node.attrs["evidence_intake"]["metadata"],
            {"packages": ["argon2-cffi"], "nested": {"severity": "high"}},
        )
        self.assertEqual(receipt.node.attrs["evidence_intake"]["payload_digest"], digest)

    def test_identical_replay_is_idempotent(self):
        first = self.submit()
        second = self.submit()
        self.assertFalse(second.created)
        self.assertEqual(second.evidence_id, first.evidence_id)
        self.assertEqual(len(self.graph.by_type(NodeType.EVIDENCE, live_only=False)), 1)

    def test_same_external_id_with_different_content_is_refused(self):
        self.submit()
        with self.assertRaises(EvidenceConflictError):
            self.submit(text="different report")

    def test_replay_recomputes_the_stored_payload_not_only_the_digest_field(self):
        receipt = self.submit()
        receipt.node.label = "forged label"
        with self.assertRaises(EvidenceConflictError):
            self.submit()

    def test_replay_refuses_a_corrupt_stored_digest(self):
        receipt = self.submit()
        receipt.node.attrs["evidence_intake"]["payload_digest"] = "0" * 64
        with self.assertRaises(EvidenceConflictError):
            self.submit()

    def test_deterministic_id_collision_with_a_non_evidence_node_is_refused(self):
        metadata = {"package": "argon2-cffi", "severity": "critical"}
        canonical = canonical_payload(
            text="CVE-2026-9999 affects argon2-cffi",
            source_id=self.source.id,
            external_id="CVE-2026-9999",
            metadata=metadata,
        )
        self.graph.add_node(
            NodeType.CLAIM,
            "collision",
            id=evidence_id_for(canonical),
            provenance=Provenance(source_id=self.source.id),
        )
        with self.assertRaises(EvidenceConflictError):
            self.submit()

    def test_duplicate_external_id_records_are_refused_as_corruption(self):
        first = self.submit()
        duplicate = self.graph.add_node(
            NodeType.EVIDENCE,
            "other",
            id="evidence:manual-duplicate",
            attrs={
                "kind": "observation",
                "evidence_intake": {
                    "version": 1,
                    "external_id": "CVE-2026-9999",
                    "payload_digest": first.node.attrs["evidence_intake"]["payload_digest"],
                    "metadata": {"package": "argon2-cffi", "severity": "critical"},
                },
            },
            provenance=first.node.provenance,
        )
        self.assertIsNotNone(duplicate)
        with self.assertRaises(EvidenceConflictError):
            self.submit(text="different report")

    def test_subtraction_invariant_freezes_belief_and_edges(self):
        assumption = Assumption(
            id="assumption:argon",
            statement="argon2 is safe",
            status=AssumptionStatus.ACTIVE,
        )
        self.graph.add_assumption(assumption)
        before_edges = self.graph.to_dict()["edges"]
        before_assumptions = self.graph.to_dict()["assumptions"]
        before_invalidated = {
            n.id: n.invalidated for n in self.graph.nodes.values()
        }
        before_nodes = len(self.graph.nodes)
        receipt = self.submit()
        self.assertEqual(len(self.graph.nodes), before_nodes + 1)
        self.assertIs(receipt.node.type, NodeType.EVIDENCE)
        self.assertEqual(self.graph.to_dict()["edges"], before_edges)
        self.assertEqual(self.graph.to_dict()["assumptions"], before_assumptions)
        for node_id, invalidated in before_invalidated.items():
            self.assertEqual(self.graph.node(node_id).invalidated, invalidated)
        self.assertFalse(
            any(
                e.type in (EdgeType.CONTRADICTS, EdgeType.SUPPORTS)
                for e in self.graph.edges.values()
            )
        )

    def test_intake_signature_has_no_verdict_or_edge_authority(self):
        params = set(inspect.signature(EvidenceIntake.submit).parameters)
        self.assertEqual(params, {"self", "text", "source_id", "external_id", "metadata"})
        for forbidden in (
            "edge",
            "edge_type",
            "target",
            "target_id",
            "assumption_id",
            "verdict",
            "reject",
            "invalidate",
            "status",
        ):
            self.assertNotIn(forbidden, params)

    def test_replay_and_conflict_survive_graph_round_trip(self):
        self.submit()
        restored = ContextGraph.from_dict(self.graph.to_dict())
        replay = submit_evidence(
            restored,
            text="CVE-2026-9999 affects argon2-cffi",
            source_id=self.source.id,
            external_id="CVE-2026-9999",
            metadata={"package": "argon2-cffi", "severity": "critical"},
        )
        self.assertFalse(replay.created)
        with self.assertRaises(EvidenceConflictError):
            submit_evidence(
                restored,
                text="different",
                source_id=self.source.id,
                external_id="CVE-2026-9999",
            )


class EvidenceIdentityTest(unittest.TestCase):
    """PR #12: an observation is identified by its whole payload digest.

    Intake used ``slug``, which is right for prose and wrong here. A canonical
    evidence payload opens with the envelope's constant keys, so the 40-char
    body slug derives its stem from is spent before ``text`` contributes
    anything, and every observation from one source normalises identically.
    That left the 6-hex-character sha1 tail — 24 bits — to tell observations
    apart, and the loser of a collision was refused permanently.
    """

    #: Two ordinary observations whose *legacy* ids collide. The precondition is
    #: asserted below rather than assumed, so this fixture cannot quietly stop
    #: describing a collision and let the test pass for the wrong reason.
    COLLIDING = ("observation 1340", "observation 1902")

    def setUp(self):
        self.graph = ContextGraph()
        self.source = self.graph.add_node(
            NodeType.SOURCE,
            "NVD feed",
            id="source:nvd",
            attrs={"origin": "nvd", "retrieved_at": "fixture"},
        )

    def canonical(self, text, external_id=None, metadata=None):
        return canonical_payload(
            text=text,
            source_id=self.source.id,
            external_id=external_id,
            metadata=metadata or {},
        )

    def legacy_id(self, canonical):
        """The id intake minted before PR #12."""
        return slug(canonical, "evidence")

    def plant_legacy(self, text, node_type=NodeType.EVIDENCE, intake=True):
        """A node ingested under the old scheme, as a restored graph would hold it."""
        canonical = self.canonical(text)
        attrs = {}
        if intake:
            attrs = {
                "kind": "observation",
                "evidence_intake": {
                    "version": 1,
                    "external_id": None,
                    "payload_digest": hashlib.sha256(canonical.encode()).hexdigest(),
                    "metadata": {},
                },
            }
        return self.graph.add_node(
            node_type,
            text,
            id=self.legacy_id(canonical),
            attrs=attrs,
            provenance=Provenance(source_id=self.source.id, extractor="legacy"),
        )

    def test_the_id_is_the_whole_payload_digest(self):
        receipt = submit_evidence(
            self.graph, text="a plain observation", source_id=self.source.id
        )
        canonical = self.canonical("a plain observation")
        self.assertEqual(
            receipt.evidence_id,
            "evidence:" + hashlib.sha256(canonical.encode()).hexdigest(),
        )
        self.assertEqual(
            receipt.node.attrs["evidence_intake"]["payload_digest"],
            receipt.evidence_id.split(":", 1)[1],
        )

    def test_the_old_scheme_spent_its_whole_body_on_the_envelope(self):
        left, right = (self.canonical(text) for text in self.COLLIDING)
        self.assertNotEqual(left, right)
        stem = self.legacy_id(left).split(":", 1)[1]
        self.assertTrue(stem.startswith("external-id-null-metadata-source-id-"))
        self.assertEqual(self.legacy_id(left), self.legacy_id(right))

    def test_observations_that_collided_under_the_old_scheme_both_land(self):
        first, second = (
            submit_evidence(self.graph, text=text, source_id=self.source.id)
            for text in self.COLLIDING
        )
        self.assertTrue(first.created)
        self.assertTrue(second.created)
        self.assertNotEqual(first.evidence_id, second.evidence_id)
        self.assertEqual(
            len(self.graph.by_type(NodeType.EVIDENCE, live_only=False)), 2
        )

    def test_evidence_ingested_under_the_old_scheme_is_still_deduplicated(self):
        planted = self.plant_legacy("an older observation")
        replay = submit_evidence(
            self.graph, text="an older observation", source_id=self.source.id
        )
        self.assertFalse(replay.created)
        self.assertEqual(replay.evidence_id, planted.id)
        self.assertEqual(
            len(self.graph.by_type(NodeType.EVIDENCE, live_only=False)), 1
        )

    def test_a_legacy_id_holding_other_content_no_longer_blocks(self):
        squatter = self.plant_legacy(self.COLLIDING[0])
        receipt = submit_evidence(
            self.graph, text=self.COLLIDING[1], source_id=self.source.id
        )
        self.assertTrue(receipt.created)
        self.assertNotEqual(receipt.evidence_id, squatter.id)
        self.assertEqual(receipt.node.label, self.COLLIDING[1])

    def test_a_foreign_node_on_a_legacy_id_no_longer_blocks(self):
        self.plant_legacy(self.COLLIDING[0], node_type=NodeType.CLAIM, intake=False)
        receipt = submit_evidence(
            self.graph, text=self.COLLIDING[1], source_id=self.source.id
        )
        self.assertTrue(receipt.created)


class DurableEvidenceWriteTest(unittest.TestCase):
    def _persistent(self, root: Path):
        built = Session.build(rounds=1)
        built.save(root)
        loaded = Session.load(root)
        cp = Checkpointer(loaded, "every-ask")
        source = next(n for n in loaded.graph.nodes.values() if n.type is NodeType.SOURCE)
        return loaded, cp, source.id

    def test_success_commits_and_returns_a_new_live_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, cp, source_id = self._persistent(Path(tmp))
            before = session.generation
            result = writes.mesh_submit_evidence(
                session,
                cp,
                text="new observation",
                source_id=source_id,
                external_id="OBS-1",
            )
            self.assertTrue(result.changed)
            self.assertIsNot(result.session, session)
            self.assertEqual(result.session.generation, before + 1)
            self.assertIs(cp.session, result.session)
            restored = Session.load(Path(tmp))
            self.assertIn(result.payload["evidence_id"], restored.graph.nodes)

    def test_replay_does_not_create_an_extra_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, cp, source_id = self._persistent(Path(tmp))
            first = writes.mesh_submit_evidence(
                session, cp, text="same", source_id=source_id, external_id="OBS-2"
            )
            generation = first.session.generation
            second = writes.mesh_submit_evidence(
                first.session, cp, text="same", source_id=source_id, external_id="OBS-2"
            )
            self.assertFalse(second.changed)
            self.assertFalse(second.payload["created"])
            self.assertEqual(second.session.generation, generation)

    def test_contention_leaves_the_live_session_byte_equivalent(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, cp, source_id = self._persistent(Path(tmp))
            before = session.graph.to_dict()
            with writer_lock(Path(tmp)):
                with self.assertRaises(writes.ControlledWriteError):
                    writes.mesh_submit_evidence(
                        session, cp, text="contended", source_id=source_id
                    )
            self.assertEqual(session.graph.to_dict(), before)
            self.assertEqual(cp.pending, 0)
            self.assertEqual(cp.contended, 1)

    def test_never_policy_refuses_before_mutating(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, _, source_id = self._persistent(Path(tmp))
            cp = Checkpointer(session, "never")
            before = session.graph.to_dict()
            with self.assertRaises(writes.ControlledWriteError):
                writes.mesh_submit_evidence(
                    session, cp, text="not durable", source_id=source_id
                )
            self.assertEqual(session.graph.to_dict(), before)

    def test_ephemeral_demo_refuses_before_mutating(self):
        session = Session.build(rounds=1)
        cp = Checkpointer(session, "every-ask")
        source = next(n for n in session.graph.nodes.values() if n.type is NodeType.SOURCE)
        before = session.graph.to_dict()
        with self.assertRaises(writes.ControlledWriteError):
            writes.mesh_submit_evidence(
                session, cp, text="ephemeral", source_id=source.id
            )
        self.assertEqual(session.graph.to_dict(), before)


if __name__ == "__main__":
    unittest.main()
