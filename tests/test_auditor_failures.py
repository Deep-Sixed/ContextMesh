"""Auditor callback failures stay inside the task lifecycle."""

import unittest

from contextmesh.execute import Event, ExecutionError, Runner, TaskState
from contextmesh.model import AssumptionStatus, NodeType


class AuditorFailureTest(unittest.TestCase):
    def test_a_raising_auditor_fails_the_task_not_the_run(self):
        calls = []
        runner = Runner("auditor exception")

        def worker(ctx):
            calls.append(ctx.attempt)
            return {"value": 7}

        def broken(ctx):
            raise RuntimeError("audit service unavailable")

        runner.task("first", worker, assumes="the input is valid", audit=broken)
        runner.task(
            "second",
            lambda ctx: {"done": True},
            assumes="the sink is available",
            needs=("first",),
        )

        report = runner.run()

        self.assertIs(runner.state("first"), TaskState.FAILED)
        self.assertIn(
            "auditor error: RuntimeError: audit service unavailable",
            report.failed["first"],
        )
        self.assertEqual(runner["first"].attempt, 1)
        self.assertEqual(runner["first"].output, {"value": 7})
        self.assertIsNone(runner["first"].node_id)
        self.assertIn("second", report.blocked)
        self.assertEqual(calls, [1])
        assumption = runner.graph.assumptions[runner["first"].assumption_id]
        self.assertIs(assumption.status, AssumptionStatus.ACTIVE)
        self.assertFalse(any(node.invalidated for node in runner.graph.nodes.values()))
        self.assertFalse(
            any(node.type is NodeType.DECISION for node in runner.graph.nodes.values())
        )

    def test_repair_retries_after_the_completed_worker_attempt(self):
        calls = []
        runner = Runner("auditor retry")

        def worker(ctx):
            calls.append(ctx.attempt)
            return {"attempt": ctx.attempt}

        def broken(ctx):
            raise OSError("temporary auditor outage")

        runner.task("work", worker, assumes="ground", audit=broken)
        first = runner.run()
        self.assertIn("work", first.failed)
        self.assertEqual(runner["work"].attempt, 1)

        runner.repair("work", audit=lambda ctx: ctx.ok("auditor recovered"))
        second = runner.run()

        self.assertTrue(second.complete)
        self.assertEqual(second.executed, ["work"])
        self.assertIs(runner.state("work"), TaskState.DONE)
        self.assertEqual(runner["work"].attempt, 2)
        self.assertEqual(calls, [1, 2])
        self.assertIsNotNone(runner["work"].node_id)

    def test_recheck_contains_one_auditor_exception_and_continues(self):
        fail_now = {"value": False}
        audited = []
        runner = Runner("recheck exception")

        def first_audit(ctx):
            audited.append("first")
            if fail_now["value"]:
                raise ConnectionError("audit backend down")
            return ctx.ok("first holds")

        def second_audit(ctx):
            audited.append("second")
            return ctx.ok("second holds")

        runner.task("first", lambda ctx: {"x": 1}, assumes="a", audit=first_audit)
        runner.task("second", lambda ctx: {"x": 2}, assumes="b", audit=second_audit)
        runner.run()
        audited.clear()
        fail_now["value"] = True

        reports = runner.recheck()

        self.assertEqual(reports, [])
        self.assertEqual(audited, ["first", "second"])
        # Work that already passed its audit is not failed by an auditor
        # that cannot answer now: nothing would ever re-run it.
        self.assertIs(runner.state("first"), TaskState.DONE)
        self.assertIs(runner.state("second"), TaskState.DONE)
        assumption = runner.graph.assumptions[runner["first"].assumption_id]
        self.assertIs(assumption.status, AssumptionStatus.ACTIVE)
        errors = [
            entry
            for entry in runner.ledger.of("first")
            if entry.event is Event.AUDIT_ERROR
        ]
        self.assertIn(
            "auditor error: ConnectionError: audit backend down",
            errors[-1].detail,
        )
        self.assertFalse(
            [e for e in runner.ledger.of("first") if e.event is Event.FAILED]
        )

        # The outage clears; the next recheck simply asks again.
        fail_now["value"] = False
        audited.clear()
        self.assertEqual(runner.recheck(), [])
        self.assertEqual(audited, ["first", "second"])
        self.assertIs(runner.state("first"), TaskState.DONE)
        self.assertIs(runner.ledger.of("first")[-1].event, Event.AUDITED)

    def test_invalid_auditor_return_is_still_a_contract_error(self):
        runner = Runner("bad verdict")
        runner.task(
            "work",
            lambda ctx: {},
            assumes="ground",
            audit=lambda ctx: "not a verdict",
        )
        with self.assertRaises(ExecutionError):
            runner.run()


if __name__ == "__main__":
    unittest.main()
