"""Selective re-execution: the rerun touches the blast radius and nothing else."""

import dataclasses
import json
import unittest

from contextmesh.execute import (
    Advisories,
    AuditContext,
    Event,
    ExecutionError,
    LedgerEntry,
    RunLedger,
    Runner,
    TaskState,
    Verdict,
    demo,
)
from contextmesh.model import AssumptionStatus, EdgeType, NodeType


def counted(name, calls, **output):
    """A worker that records that it ran, so 'nothing else ran' is checkable."""

    def run(ctx):
        calls.append(name)
        return dict(output)

    return run


def plan(calls, feed):
    """Two independent branches; only one of them stands on the feed."""
    runner = Runner("test plan")
    runner.task(
        "base",
        counted("base", calls, rows=3),
        assumes="the store is reachable",
        produces=("Store",),
        audit=lambda ctx: ctx.ok("store reachable"),
    )
    runner.task(
        "hashing",
        counted("hashing", calls, package="argon2-cffi"),
        assumes="argon2-cffi has no open advisory",
        produces=("Hasher", "Argon2 Parameters"),
        audit=lambda ctx: (
            ctx.disproved(f"open advisory: {feed.advisory(ctx.output['package'])}")
            if not feed.clear(ctx.output["package"])
            else ctx.ok("no open advisory")
        ),
    )
    runner.task(
        "routes",
        counted("routes", calls, endpoints=2),
        assumes="the hasher exposes a stable interface",
        needs=("hashing",),
        produces=("Login Endpoint",),
        audit=lambda ctx: ctx.ok("two endpoints"),
    )
    runner.task(
        "unrelated",
        counted("unrelated", calls, report="ok"),
        assumes="the metrics sink accepts writes",
        needs=("base",),
        produces=("Latency Report",),
        audit=lambda ctx: ctx.ok("sink accepted the write"),
    )
    return runner


class SchedulingTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.feed = Advisories()
        self.runner = plan(self.calls, self.feed)

    def test_first_run_executes_everything_in_dependency_order(self):
        report = self.runner.run()
        self.assertEqual(sorted(report.executed), ["base", "hashing", "routes", "unrelated"])
        self.assertTrue(report.complete)
        self.assertLess(self.calls.index("hashing"), self.calls.index("routes"))
        self.assertLess(self.calls.index("base"), self.calls.index("unrelated"))

    def test_a_task_is_a_decision_node_wired_to_its_ground(self):
        self.runner.run()
        graph = self.runner.graph
        routes = self.runner["routes"]
        node = graph.node(routes.node_id)
        self.assertIs(node.type, NodeType.DECISION)
        grounds = {
            e.dst for e in graph.out_edges(node.id, [EdgeType.DEPENDS_ON])
        }
        self.assertIn(routes.assumption_id, grounds)
        self.assertIn(self.runner["hashing"].node_id, grounds)
        produced = {
            graph.node(e.dst).label
            for e in graph.out_edges(node.id, [EdgeType.PRODUCES])
        }
        self.assertEqual(produced, {"Login Endpoint"})

    def test_a_cycle_is_refused_rather_than_hung_on(self):
        runner = Runner("cyclic")
        runner.task("a", lambda ctx: {}, assumes="x", needs=("b",))
        runner.task("b", lambda ctx: {}, assumes="y", needs=("a",))
        with self.assertRaises(ExecutionError) as caught:
            runner.run()
        self.assertIn("cycle", str(caught.exception))

    def test_an_unknown_dependency_is_refused(self):
        runner = Runner("dangling")
        runner.task("a", lambda ctx: {}, assumes="x", needs=("nope",))
        with self.assertRaises(ExecutionError):
            runner.run()


class DiscoveryTest(unittest.TestCase):
    """Nothing rejects an assumption except an auditor that disproved it."""

    def setUp(self):
        self.calls = []
        self.feed = Advisories()
        self.runner = plan(self.calls, self.feed)
        self.runner.run()

    def test_a_quiet_world_invalidates_nothing(self):
        self.assertEqual(self.runner.recheck(), [])
        for task in self.runner.tasks:
            self.assertIs(task.state, TaskState.DONE)

    def test_the_auditor_finds_the_advisory_without_being_told(self):
        self.feed.publish("argon2-cffi", "CVE-2026-9999")
        reports = self.runner.recheck()
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].statement, "argon2-cffi has no open advisory")

    def test_the_disproof_is_recorded_as_evidence_against_the_assumption(self):
        self.feed.publish("argon2-cffi", "CVE-2026-9999")
        report = self.runner.recheck()[0]
        graph = self.runner.graph
        contradictions = [
            e
            for e in graph.in_edges(report.assumption_id, [EdgeType.CONTRADICTS],
                                    live_only=False)
            if graph.node(e.src).type is NodeType.EVIDENCE
        ]
        self.assertEqual(len(contradictions), 1)
        self.assertIn("CVE-2026-9999", graph.node(contradictions[0].src).label)

    def test_recheck_executes_nothing(self):
        self.calls.clear()
        self.feed.publish("argon2-cffi", "CVE-2026-9999")
        self.runner.recheck()
        self.assertEqual(self.calls, [])


class SelectiveRerunTest(unittest.TestCase):
    """The claim the package could not previously make good on."""

    def setUp(self):
        self.calls = []
        self.feed = Advisories()
        self.runner = plan(self.calls, self.feed)
        self.runner.run()
        self.before = {t.name: t.node_id for t in self.runner.tasks}
        self.feed.publish("argon2-cffi", "CVE-2026-9999")
        self.report = self.runner.recheck()[0]
        self.calls.clear()

    def test_only_the_closure_is_marked_stale(self):
        stale = {t.name for t in self.runner.tasks if t.state is TaskState.STALE}
        self.assertEqual(stale, {"hashing", "routes"})

    def test_the_rerun_runs_the_closure_and_nothing_else(self):
        self.runner.repair(
            "hashing",
            assumes="bcrypt has no open advisory",
            run=counted("hashing", self.calls, package="bcrypt"),
            produces=("Hasher", "Bcrypt Parameters"),
        )
        report = self.runner.run()
        self.assertEqual(sorted(self.calls), ["hashing", "routes"])
        self.assertEqual(sorted(report.executed), ["hashing", "routes"])
        self.assertEqual(sorted(report.cached), ["base", "unrelated"])
        self.assertTrue(report.complete)

    def test_preserved_work_keeps_its_original_result(self):
        self.runner.repair(
            "hashing",
            assumes="bcrypt has no open advisory",
            run=counted("hashing", self.calls, package="bcrypt"),
            produces=("Hasher", "Bcrypt Parameters"),
        )
        self.runner.run()
        for name in ("base", "unrelated"):
            self.assertEqual(self.runner[name].attempt, 1)
            self.assertEqual(self.runner[name].node_id, self.before[name])
        self.assertEqual(self.runner["hashing"].attempt, 2)
        self.assertNotEqual(self.runner["hashing"].node_id, self.before["hashing"])

    def test_a_rerun_supersedes_the_old_decision_rather_than_reviving_it(self):
        self.runner.repair(
            "hashing",
            assumes="bcrypt has no open advisory",
            run=counted("hashing", self.calls, package="bcrypt"),
            produces=("Hasher", "Bcrypt Parameters"),
        )
        self.runner.run()
        graph = self.runner.graph
        old = graph.node(self.before["hashing"])
        new = graph.node(self.runner["hashing"].node_id)
        self.assertTrue(old.invalidated, "the superseded decision must stay invalidated")
        self.assertTrue(new.live)
        self.assertEqual(old.attrs["superseded_by"], new.id)
        supersedes = {e.dst for e in graph.out_edges(new.id, [EdgeType.SUPERSEDES],
                                                     live_only=False)}
        self.assertIn(old.id, supersedes)

    def test_a_rebuilt_artefact_comes_back_and_an_abandoned_one_does_not(self):
        self.runner.repair(
            "hashing",
            assumes="bcrypt has no open advisory",
            run=counted("hashing", self.calls, package="bcrypt"),
            produces=("Hasher", "Bcrypt Parameters"),
        )
        self.runner.run()
        by_label = {
            n.label: n
            for n in self.runner.graph.nodes.values()
            if n.type is NodeType.ENTITY
        }
        self.assertTrue(by_label["Hasher"].live, "rebuilt, so it exists again")
        self.assertFalse(
            by_label["Argon2 Parameters"].live, "no longer produced, so still gone"
        )
        self.assertTrue(by_label["Bcrypt Parameters"].live)
        self.assertTrue(by_label["Latency Report"].live, "never fell in the first place")

    def test_rejected_ground_blocks_the_rerun_until_it_is_repaired(self):
        report = self.runner.run()
        self.assertEqual(self.calls, [], "nothing may run on rejected ground")
        self.assertIn("hashing", report.blocked)
        self.assertIn("ground rejected", report.blocked["hashing"])
        self.assertIn("routes", report.blocked)
        self.assertFalse(report.complete)

    def test_the_rejected_assumption_stays_rejected_after_repair(self):
        old_id = self.runner["hashing"].assumption_id
        new = self.runner.repair("hashing", assumes="bcrypt has no open advisory")
        graph = self.runner.graph
        self.assertIs(graph.assumptions[old_id].status, AssumptionStatus.REJECTED)
        self.assertIs(new.status, AssumptionStatus.ACTIVE)
        self.assertEqual(new.supersedes, old_id)
        lineage = [a.statement for a in self.runner.assumptions.lineage(new.id)]
        self.assertEqual(
            lineage, ["argon2-cffi has no open advisory", "bcrypt has no open advisory"]
        )


class PlainFailureTest(unittest.TestCase):
    """A wrong output is not a false assumption, and must not behave like one."""

    def setUp(self):
        self.calls = []
        self.runner = Runner("failing plan")
        self.runner.task(
            "first",
            counted("first", self.calls, value=0),
            assumes="the input file is well formed",
            produces=("Parsed Input",),
            audit=lambda ctx: ctx.fail("value came back as zero"),
        )
        self.runner.task(
            "second",
            counted("second", self.calls),
            assumes="the sink accepts writes",
            needs=("first",),
            audit=lambda ctx: ctx.ok("fine"),
        )

    def test_the_task_fails_and_its_ground_survives(self):
        report = self.runner.run()
        self.assertIs(self.runner.state("first"), TaskState.FAILED)
        self.assertIn("first", report.failed)
        assumption = self.runner.graph.assumptions[self.runner["first"].assumption_id]
        self.assertIs(assumption.status, AssumptionStatus.ACTIVE)
        self.assertEqual(
            [n for n in self.runner.graph.nodes.values() if n.invalidated], []
        )

    def test_the_dependant_is_blocked_not_run(self):
        report = self.runner.run()
        self.assertNotIn("second", self.calls)
        self.assertIn("second", report.blocked)
        self.assertIn("waiting on first", report.blocked["second"])

    def test_a_worker_that_raises_fails_the_task_rather_than_the_run(self):
        runner = Runner("throwing plan")

        def boom(ctx):
            raise RuntimeError("connection refused")

        runner.task("boom", boom, assumes="the service is up")
        report = runner.run()
        self.assertIs(runner.state("boom"), TaskState.FAILED)
        self.assertIn("connection refused", report.failed["boom"])

    def test_a_worker_that_returns_a_non_mapping_fails_the_task_rather_than_the_run(self):
        for bad in ("not a dict", ["a", "b"], 3):
            with self.subTest(output=bad):
                runner = Runner("bad output plan")
                runner.task("bad", lambda ctx, bad=bad: bad, assumes="the service is up")
                runner.task(
                    "fine", counted("fine", self.calls, ok=True), assumes="the sink is up"
                )
                report = runner.run()
                self.assertIs(runner.state("bad"), TaskState.FAILED)
                self.assertIn("not a mapping", report.failed["bad"])
                self.assertEqual(runner["bad"].attempt, 0, "no attempt was completed")
                self.assertIs(runner.state("fine"), TaskState.DONE)

    def test_a_worker_that_returns_none_still_counts_as_empty_output(self):
        runner = Runner("none plan")
        runner.task("quiet", lambda ctx: None, assumes="the service is up")
        runner.run()
        self.assertIs(runner.state("quiet"), TaskState.DONE)
        self.assertEqual(runner.output("quiet"), {})


class RepairGuardTest(unittest.TestCase):
    """A repair that cannot leave the task runnable must change nothing."""

    def setUp(self):
        self.calls = []
        self.feed = Advisories()
        self.runner = plan(self.calls, self.feed)
        self.runner.run()
        self.feed.publish("argon2-cffi", "CVE-2026-9999")
        self.runner.recheck()

    def snapshot(self):
        tasks = [
            (t.name, t.state, t.assumption_id, t.assumes, t.run, t.audit, t.produces)
            for t in self.runner.tasks
        ]
        return (
            tasks,
            json.dumps(self.runner.to_dict(), sort_keys=True, default=str),
            self.runner.ledger.head,
            json.dumps(self.runner.graph.to_dict(), sort_keys=True),
        )

    def test_regrounding_on_the_rejected_statement_is_refused(self):
        before = self.snapshot()
        with self.assertRaises(ExecutionError) as caught:
            self.runner.repair("hashing", assumes="argon2-cffi has no open advisory")
        self.assertIn("rejected", str(caught.exception))
        self.assertEqual(self.snapshot(), before)

    def test_a_repair_without_new_ground_on_rejected_ground_is_refused(self):
        before = self.snapshot()
        with self.assertRaises(ExecutionError):
            self.runner.repair("hashing", produces=("Hasher",))
        self.assertEqual(self.snapshot(), before)

    def test_repairing_back_to_an_earlier_rejected_ground_writes_no_lineage_cycle(self):
        """argon2 → bcrypt, bcrypt falls, back to argon2: refused, history intact."""
        bcrypt = self.runner.repair(
            "hashing",
            assumes="bcrypt has no open advisory",
            run=counted("hashing", self.calls, package="bcrypt"),
        )
        argon2_id = bcrypt.supersedes
        self.runner.run()
        self.feed.publish("bcrypt", "CVE-2026-0001")
        self.runner.recheck()
        self.assertIs(bcrypt.status, AssumptionStatus.REJECTED)
        with self.assertRaises(ExecutionError):
            self.runner.repair("hashing", assumes="argon2-cffi has no open advisory")
        argon2 = self.runner.graph.assumptions[argon2_id]
        self.assertIsNone(argon2.supersedes)
        self.assertEqual(argon2.version, 1)
        self.assertEqual(
            [a.statement for a in self.runner.assumptions.lineage(bcrypt.id)],
            ["argon2-cffi has no open advisory", "bcrypt has no open advisory"],
        )

    def test_regrounding_on_another_tasks_active_ground_leaves_its_lineage_alone(self):
        shared = self.runner.graph.assumptions[self.runner["base"].assumption_id]
        new = self.runner.repair("hashing", assumes=shared.statement)
        self.assertEqual(new.id, shared.id)
        self.assertIsNone(shared.supersedes, "an existing ground was relabelled a replacement")
        self.assertEqual(shared.version, 1)
        self.assertEqual(self.runner["hashing"].assumption_id, shared.id)

    def test_a_task_with_no_assumption_is_refused_before_anything_changes(self):
        task = self.runner["unrelated"]
        task.assumption_id = None
        original_run = task.run
        entries = len(self.runner.ledger)
        with self.assertRaises(ExecutionError):
            self.runner.repair("unrelated", run=counted("x", self.calls))
        self.assertIs(task.run, original_run)
        self.assertIs(task.state, TaskState.DONE)
        self.assertEqual(len(self.runner.ledger), entries)


class VerdictTest(unittest.TestCase):
    def context(self):
        runner = Runner("verdicts")
        task = runner.task("t", lambda ctx: {}, assumes="something")
        return AuditContext(
            task=task,
            output={},
            assumption=runner.graph.assumptions[task.assumption_id],
            graph=runner.graph,
        )

    def test_a_bare_false_is_a_failure_not_a_disproof(self):
        runner = Runner("bools")
        runner.task("t", lambda ctx: {"x": 1}, assumes="something", audit=lambda ctx: False)
        runner.run()
        self.assertIs(runner.state("t"), TaskState.FAILED)
        assumption = runner.graph.assumptions[runner["t"].assumption_id]
        self.assertIs(assumption.status, AssumptionStatus.ACTIVE)

    def test_a_bare_true_passes(self):
        runner = Runner("bools")
        runner.task("t", lambda ctx: {"x": 1}, assumes="something", audit=lambda ctx: True)
        runner.run()
        self.assertIs(runner.state("t"), TaskState.DONE)

    def test_anything_else_is_an_error(self):
        runner = Runner("bad auditor")
        runner.task("t", lambda ctx: {}, assumes="something", audit=lambda ctx: "sure")
        with self.assertRaises(ExecutionError):
            runner.run()

    def test_the_helpers_say_which_kind_of_failure_it_is(self):
        ctx = self.context()
        self.assertEqual(ctx.ok("fine"), Verdict(True, "fine"))
        self.assertFalse(ctx.fail("wrong").disproves)
        self.assertTrue(ctx.disproved("false ground").disproves)

    def test_an_unaudited_task_is_reported_rather_than_failed(self):
        runner = Runner("unaudited")
        runner.task("t", lambda ctx: {}, assumes="something")
        runner.run()
        self.assertIs(runner.state("t"), TaskState.DONE)
        self.assertEqual(runner.unaudited, ("t",))


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.runner = demo().runner

    def test_the_chain_verifies(self):
        self.assertTrue(self.runner.ledger.verify())

    def test_a_rewritten_entry_breaks_the_chain(self):
        ledger = self.runner.ledger
        entries = ledger._entries
        entries[1] = dataclasses.replace(entries[1], detail="never happened")
        self.assertFalse(ledger.verify())

    def test_a_dropped_entry_breaks_the_chain(self):
        ledger = self.runner.ledger
        del ledger._entries[2]
        self.assertFalse(ledger.verify())

    def test_the_public_view_cannot_be_mutated(self):
        entries = self.runner.ledger.entries
        self.assertIsInstance(entries, tuple)
        self.assertEqual(len(entries), len(self.runner.ledger))

    def test_the_invalidation_is_in_the_record_with_its_reason(self):
        details = [
            e.detail
            for e in self.runner.ledger.entries
            if e.event.value == "disproved"
        ]
        self.assertEqual(len(details), 1)
        self.assertIn("CVE-2026-9999", details[0])

    def test_the_rerun_is_in_the_record_as_a_second_attempt(self):
        executed = [e for e in self.runner.ledger.entries if e.event.value == "executed"]
        hashing = [e for e in executed if e.task == "hashing"]
        self.assertEqual(len(hashing), 2)
        self.assertIn("supersedes", hashing[1].detail)

    def test_cached_work_is_recorded_as_cached(self):
        cached = {e.task for e in self.runner.ledger.entries if e.event.value == "cached"}
        self.assertEqual(cached, {"schema", "tokens"})


class DemoTest(unittest.TestCase):
    def setUp(self):
        self.run = demo()

    def test_the_blast_radius_is_three_decisions_and_their_artefacts(self):
        graph = self.run.runner.graph
        report = self.run.invalidations[0]
        kinds = {}
        for node_id in report.invalidated:
            kinds.setdefault(graph.node(node_id).type.value, []).append(node_id)
        self.assertEqual(len(kinds["decision"]), 3)
        self.assertEqual(len(kinds["entity"]), 5)

    def test_the_schema_and_token_work_is_preserved(self):
        report = self.run.invalidations[0]
        runner = self.run.runner
        self.assertIn(runner["schema"].node_id, report.preserved)
        self.assertIn(runner["tokens"].node_id, report.preserved)

    def test_everything_ends_standing(self):
        for task in self.run.runner.tasks:
            self.assertIs(task.state, TaskState.DONE)
        self.assertTrue(self.run.runner.rounds[-1].complete)

    def test_the_graph_stays_typed(self):
        self.assertEqual(self.run.runner.graph.untyped_edges, 0)

    def test_the_run_is_reproducible(self):
        a, b = demo(), demo()
        self.assertEqual(a.runner.ledger.head, b.runner.ledger.head)
        self.assertEqual(a.runner.graph.to_dict(), b.runner.graph.to_dict())
        self.assertEqual(a.runner.to_dict(), b.runner.to_dict())


class LedgerDigestTest(unittest.TestCase):
    """The digest is taken over canonical JSON, not joined fields."""

    def entries(self, *specs):
        ledger = RunLedger()
        for task, detail in specs:
            ledger.record(1, Event.EXECUTED, task, detail)
        return ledger

    def test_a_separator_inside_a_field_cannot_forge_a_digest(self):
        # Joined on "|", both of these render as ...|a|b|c|... — the exact
        # collision canonical JSON exists to rule out.
        first = self.entries(("a", "b|c"))
        second = self.entries(("a|b", "c"))
        self.assertNotEqual(first.head, second.head)
        self.assertTrue(first.verify())
        self.assertTrue(second.verify())

    def test_identical_content_gives_an_identical_head(self):
        self.assertEqual(
            self.entries(("a", "one"), ("b", "two")).head,
            self.entries(("a", "one"), ("b", "two")).head,
        )

    def test_the_digest_is_a_full_sha256(self):
        ledger = self.entries(("a", "one"))
        self.assertEqual(len(ledger.head), 64)
        self.assertEqual(ledger.short_head, ledger.head[:12])
        self.assertEqual(ledger.entries[0].short_digest, ledger.entries[0].digest[:12])

    def test_the_first_entry_chains_off_genesis(self):
        ledger = self.entries(("a", "one"))
        self.assertEqual(
            ledger.entries[0].digest, ledger.entries[0].compute_digest(RunLedger.GENESIS)
        )

    def test_reordering_two_entries_breaks_the_chain(self):
        ledger = self.entries(("a", "one"), ("b", "two"))
        ledger._entries[0], ledger._entries[1] = ledger._entries[1], ledger._entries[0]
        self.assertFalse(ledger.verify())

    def test_a_rewritten_receipt_breaks_the_chain(self):
        ledger = RunLedger()
        ledger.record(1, Event.DISPROVED, "t", "why", data={"invalidated": {"a": ["x"]}})
        self.assertTrue(ledger.verify())
        ledger._entries[0].data["invalidated"]["a"] = ["something else"]
        self.assertFalse(ledger.verify())

    def test_unicode_survives_the_round_trip(self):
        ledger = self.entries(("t", "argon2 → bcrypt · CVE-2026-9999"))
        self.assertTrue(ledger.verify())


class LedgerDataGuardTest(unittest.TestCase):
    """Payload values without one canonical JSON form are refused, not coerced."""

    def record(self, data):
        RunLedger().record(1, Event.EXECUTED, "t", "d", data=data)

    def test_a_set_is_refused(self):
        with self.assertRaises(ExecutionError) as caught:
            self.record({"nodes": {"a", "b"}})
        self.assertIn("not JSON-deterministic", str(caught.exception))

    def test_bytes_are_refused(self):
        with self.assertRaises(ExecutionError):
            self.record({"blob": b"\x00"})

    def test_a_nan_is_refused(self):
        with self.assertRaises(ExecutionError):
            self.record({"score": float("nan")})

    def test_an_infinity_is_refused(self):
        with self.assertRaises(ExecutionError):
            self.record({"score": float("inf")})

    def test_a_non_string_key_is_refused(self):
        with self.assertRaises(ExecutionError):
            self.record({"m": {1: "a"}})

    def test_an_arbitrary_object_is_refused(self):
        with self.assertRaises(ExecutionError):
            self.record({"task": object()})

    def test_nesting_is_walked_not_just_the_top_level(self):
        with self.assertRaises(ExecutionError) as caught:
            self.record({"outer": [{"inner": {"deep"}}]})
        self.assertIn("outer[0].inner", str(caught.exception))

    def test_a_tuple_is_stored_as_the_list_it_reads_back_as(self):
        ledger = RunLedger()
        entry = ledger.record(1, Event.EXECUTED, "t", "d", data={"xs": ("a", "b")})
        self.assertEqual(entry.data["xs"], ["a", "b"])
        self.assertTrue(ledger.verify())

    def test_the_ordinary_payload_types_pass(self):
        ledger = RunLedger()
        ledger.record(
            1, Event.EXECUTED, "t", "d",
            data={"s": "x", "i": 1, "f": 1.5, "b": True, "n": None, "l": [1], "d": {"k": "v"}},
        )
        self.assertTrue(ledger.verify())


class ReceiptTest(unittest.TestCase):
    """One entry carries the whole blast radius, so the ledger can be read alone."""

    def setUp(self):
        self.run = demo()
        self.ledger = self.run.runner.ledger
        self.receipts = self.ledger.receipts()

    def test_one_receipt_per_disproof(self):
        self.assertEqual(len(self.receipts), 1)
        self.assertEqual(self.ledger.count(Event.DISPROVED), 1)

    def test_it_names_the_ground_the_evidence_and_the_auditor(self):
        receipt = self.receipts[0]
        report = self.run.invalidations[0]
        self.assertEqual(receipt["assumption_id"], report.assumption_id)
        self.assertEqual(receipt["assumption"], "argon2-cffi has no open advisory")
        self.assertEqual(receipt["disproved_by"], "hashing")
        self.assertIn("CVE-2026-9999", receipt["reason"])
        self.assertIn("disproof", self.run.runner.graph.node(receipt["evidence_id"]).label)

    def test_it_carries_the_closure_with_a_reason_chain_for_each_node(self):
        receipt = self.receipts[0]
        report = self.run.invalidations[0]
        self.assertEqual(set(receipt["invalidated"]), set(report.invalidated))
        for node_id, chain in receipt["invalidated"].items():
            self.assertEqual(chain, report.invalidated[node_id])
            self.assertTrue(chain, f"{node_id} has an empty chain")

    def test_it_carries_the_preserved_set_as_well(self):
        receipt = self.receipts[0]
        self.assertEqual(receipt["preserved"], self.run.invalidations[0].preserved)
        self.assertNotIn(
            self.run.runner["schema"].node_id, receipt["invalidated"]
        )
        self.assertIn(self.run.runner["schema"].node_id, receipt["preserved"])

    def test_the_per_node_events_are_kept_alongside_it(self):
        # The receipt is the atomic event; these are each node's own history.
        # Both are wanted, so the duplication is deliberate.
        invalidated = [e for e in self.ledger.entries if e.event is Event.INVALIDATED]
        self.assertEqual([e.task for e in invalidated], ["hashing", "routes", "rate_limit"])
        for entry in invalidated:
            self.assertTrue(entry.detail)

    def test_the_receipt_is_covered_by_the_chain(self):
        self.assertTrue(self.ledger.verify())
        entry = next(e for e in self.ledger.entries if e.event is Event.DISPROVED)
        self.assertEqual(entry.digest, entry.compute_digest(
            self.ledger.entries[entry.seq - 2].digest
        ))


class ReplayTest(unittest.TestCase):
    """Answer the four questions from serialised ledger entries and nothing else."""

    def setUp(self):
        run = demo()
        self.expected = run.invalidations[0]
        self.tasks = run.runner
        # Round-trip through JSON: whatever survives this is all a reader gets.
        self.rows = json.loads(json.dumps(run.runner.ledger.to_dict()))

    def replay(self):
        for row in self.rows:
            if row["event"] == "disproved" and row["data"]:
                return row["data"]
        raise AssertionError("no receipt in the ledger")

    def test_what_assumption_failed(self):
        self.assertEqual(self.replay()["assumption_id"], self.expected.assumption_id)

    def test_what_evidence_disproved_it(self):
        self.assertTrue(self.replay()["evidence_id"])
        self.assertIn("CVE-2026-9999", self.replay()["reason"])

    def test_what_exactly_fell_and_why(self):
        invalidated = self.replay()["invalidated"]
        self.assertEqual(set(invalidated), set(self.expected.invalidated))
        for node_id, chain in invalidated.items():
            self.assertEqual(" → ".join(chain), self.expected.why(node_id))

    def test_what_explicitly_survived(self):
        self.assertEqual(self.replay()["preserved"], self.expected.preserved)

    def test_the_chain_verifies_off_the_serialised_rows(self):
        previous = RunLedger.GENESIS
        for row in self.rows:
            entry = LedgerEntry(
                seq=row["seq"],
                round=row["round"],
                event=Event(row["event"]),
                task=row["task"],
                detail=row["detail"],
                node_id=row["node_id"],
                assumption_id=row["assumption_id"],
                data=row["data"],
            )
            self.assertEqual(entry.compute_digest(previous), row["digest"])
            previous = row["digest"]


if __name__ == "__main__":
    unittest.main()
