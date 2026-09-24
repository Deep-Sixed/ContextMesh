"""Controlled MCP evidence rechecks contain auditor callback outages."""

import tempfile
import unittest

from contextmesh.execute import Event, ExecutionError, Runner, TaskRegistry, TaskState
from contextmesh.model import AssumptionStatus
from contextmesh.recheck import EvidenceRecheckError
from contextmesh_mcp.session import Checkpointer, Session
from contextmesh_mcp.writes import mesh_recheck


class McpEvidenceRecheckAuditorFailureTest(unittest.TestCase):
    def _saved_session(self, first_audit, second_audit):
        registry = TaskRegistry()
        registry.register_worker("w", lambda ctx: {"x": 1})
        registry.register_auditor("first-audit", first_audit)
        registry.register_auditor("second-audit", second_audit)
        base = Session.build(rounds=1)
        runner = Runner("probe", graph=base.graph, registry=registry)
        runner.task(
            "first",
            worker_key="w",
            auditor_key="first-audit",
            assumes="first ground",
        )
        runner.task(
            "second",
            worker_key="w",
            auditor_key="second-audit",
            assumes="second ground",
        )
        self.assertTrue(runner.run().complete)
        base.runner = runner
        return base, registry

    def test_one_auditor_outage_is_reported_not_failed_and_second_recheck_runs(self):
        fail = {"value": False}
        calls = []

        def first(ctx):
            calls.append("first")
            if fail["value"]:
                raise RuntimeError("auditor outage")
            return ctx.ok("first holds")

        def second(ctx):
            calls.append("second")
            return ctx.ok("second holds")

        base, registry = self._saved_session(first, second)
        with tempfile.TemporaryDirectory() as tmp:
            base.save(tmp)
            live = Session.load(tmp, registry=registry)
            cp = Checkpointer(live)
            calls.clear()
            fail["value"] = True

            result = mesh_recheck(live, cp)

            self.assertTrue(result.changed)
            self.assertEqual(cp.commits, 1)
            self.assertEqual(calls, ["first", "second"])
            # The client is told, instead of seeing audited: 1 and nothing else.
            self.assertEqual(result.payload["audited"], 1)
            self.assertEqual(
                result.payload["audit_errors"],
                [{"task": "first", "error": "auditor error: RuntimeError: auditor outage"}],
            )
            runner = result.session.runner
            assert runner is not None
            self.assertIs(runner.state("first"), TaskState.DONE)
            self.assertIs(runner.state("second"), TaskState.DONE)
            assumption = runner.graph.assumptions[runner["first"].assumption_id]
            self.assertIs(assumption.status, AssumptionStatus.ACTIVE)
            errors = [
                entry for entry in runner.ledger.of("first") if entry.event is Event.AUDIT_ERROR
            ]
            self.assertIn("auditor error: RuntimeError: auditor outage", errors[-1].detail)
            restored = Session.load(tmp, registry=registry)
            assert restored.runner is not None
            self.assertIs(restored.runner.state("first"), TaskState.DONE)
            self.assertIs(restored.runner.state("second"), TaskState.DONE)

            # Recovery needs no repair: the next recheck asks again.
            fail["value"] = False
            calls.clear()
            again = mesh_recheck(restored, Checkpointer(restored))
            self.assertEqual(calls, ["first", "second"])
            self.assertEqual(again.payload["audited"], 2)
            self.assertEqual(again.payload["audit_errors"], [])

    def test_invalid_auditor_return_remains_a_contract_error_and_commits_nothing(self):
        bad = {"value": False}

        def first(ctx):
            if bad["value"]:
                return "not a verdict"
            return ctx.ok("holds")

        def second(ctx):
            return ctx.ok("holds")

        base, registry = self._saved_session(first, second)
        with tempfile.TemporaryDirectory() as tmp:
            base.save(tmp)
            live = Session.load(tmp, registry=registry)
            cp = Checkpointer(live)
            bad["value"] = True

            with self.assertRaises(ExecutionError):
                mesh_recheck(live, cp)

            self.assertEqual(cp.commits, 0)
            self.assertEqual(Session.load(tmp, registry=registry).generation, live.generation)
            assert live.runner is not None
            self.assertIs(live.runner.state("first"), TaskState.DONE)

    def test_an_evidence_contract_violation_inside_disproved_commits_nothing(self):
        """ctx.disproved(evidence_id="") breaks the evidence contract. It used
        to be caught as an auditor outage and committed as a task failure; the
        same violation detected after return was already fatal."""
        bad = {"value": False}

        def first(ctx):
            if bad["value"]:
                return ctx.disproved("gone", evidence_id="")
            return ctx.ok("holds")

        base, registry = self._saved_session(first, lambda ctx: ctx.ok("holds"))
        with tempfile.TemporaryDirectory() as tmp:
            base.save(tmp)
            live = Session.load(tmp, registry=registry)
            cp = Checkpointer(live)
            bad["value"] = True

            with self.assertRaises(EvidenceRecheckError):
                mesh_recheck(live, cp)

            self.assertEqual(cp.commits, 0)
            assert live.runner is not None
            self.assertIs(live.runner.state("first"), TaskState.DONE)
            self.assertFalse(
                [e for e in live.runner.ledger.of("first") if e.event is Event.FAILED]
            )


if __name__ == "__main__":
    unittest.main()
