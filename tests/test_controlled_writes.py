"""PR #8B-8D controlled-write integration over the native durable session."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from contextmesh.execute import Runner, TaskRegistry, TaskState
from contextmesh.model import AssumptionStatus, EdgeType, NodeType
from contextmesh_mcp.session import Checkpointer, Session, writer_lock
from contextmesh_mcp.writes import (
    ControlledWriteError,
    mesh_recheck,
    mesh_repair,
    mesh_resume,
    mesh_submit_evidence,
)


def deployment() -> TaskRegistry:
    registry = TaskRegistry()
    registry.register_worker("auth.schema.v1", lambda ctx: {"schema": "v1"})
    registry.register_worker("auth.hash.argon2.v1", lambda ctx: {"impl": "argon2"})
    registry.register_worker("auth.hash.bcrypt.v1", lambda ctx: {"impl": "bcrypt"})
    registry.register_worker(
        "auth.routes.v1",
        lambda ctx: {"hasher": ctx.inputs["hashing"]["impl"]},
    )
    registry.register_worker("auth.tokens.v1", lambda ctx: {"algorithm": "EdDSA"})

    def audit_hash(ctx):
        # The advisory contradicts Argon2 ground, not every future hasher this
        # task may be repaired onto. This condition is load-bearing: without it
        # a successful bcrypt rerun would be rejected by evidence about argon2.
        if ctx.output.get("impl") != "argon2":
            return ctx.ok(
                "the active implementation is not affected by the Argon2 advisory"
            )
        for node in ctx.graph.nodes.values():
            if node.type is not NodeType.EVIDENCE:
                continue
            record = node.attrs.get("evidence_intake", {})
            metadata = record.get("metadata", {})
            if (
                metadata.get("package") == "argon2"
                and metadata.get("severity") == "critical"
            ):
                return ctx.disproved(
                    "CVE-2026-9999 affects the active Argon2 implementation",
                    evidence_id=node.id,
                )
        return ctx.ok("no critical advisory applies")

    registry.register_auditor("auth.hash.audit.v1", audit_hash)
    return registry


def saved_auth_session(root: Path) -> Session:
    base = Session.build(rounds=2)
    base.graph.add_node(
        NodeType.SOURCE,
        "Vendor advisory feed",
        id="source:vendor-advisory",
        attrs={"origin": "vendor", "retrieved_at": "fixture"},
    )
    runner = Runner("auth", graph=base.graph, registry=deployment())
    runner.task(
        "schema",
        worker_key="auth.schema.v1",
        assumes="the schema store is available",
        produces=("Auth Schema",),
    )
    runner.task(
        "hashing",
        worker_key="auth.hash.argon2.v1",
        auditor_key="auth.hash.audit.v1",
        assumes="Argon2 has no open advisory",
        needs=("schema",),
        produces=("Password Hasher",),
    )
    runner.task(
        "routes",
        worker_key="auth.routes.v1",
        assumes="the hasher interface is stable",
        needs=("hashing",),
        produces=("Auth Routes",),
    )
    runner.task(
        "tokens",
        worker_key="auth.tokens.v1",
        assumes="the signing key is provisioned",
        needs=("schema",),
        produces=("Tokens",),
    )
    first = runner.run()
    assert first.complete
    base.runner = runner
    base.save(root)
    return Session.load(root, registry=deployment())


class ControlledWriteIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "session"
        self.session = saved_auth_session(self.root)
        self.checkpointer = Checkpointer(self.session)

    def tearDown(self):
        self.temp.cleanup()

    def submit_cve(self):
        result = mesh_submit_evidence(
            self.session,
            self.checkpointer,
            text="CVE-2026-9999 affects Argon2",
            source_id="source:vendor-advisory",
            external_id="CVE-2026-9999",
            metadata={"package": "argon2", "severity": "critical"},
        )
        self.session = result.session
        return result

    def test_recheck_is_a_registered_auditor_decision_not_a_client_verdict(self):
        evidence = self.submit_cve().payload["evidence_id"]
        result = mesh_recheck(self.session, self.checkpointer)
        self.session = result.session

        runner = self.session.runner
        assert runner is not None
        hashing = runner["hashing"]
        routes = runner["routes"]
        schema = runner["schema"]
        tokens = runner["tokens"]
        assumption = runner.graph.assumptions[hashing.assumption_id]

        self.assertIs(assumption.status, AssumptionStatus.REJECTED)
        self.assertIs(hashing.state, TaskState.STALE)
        self.assertIs(routes.state, TaskState.STALE)
        self.assertIs(schema.state, TaskState.DONE)
        self.assertIs(tokens.state, TaskState.DONE)
        self.assertEqual(result.payload["audited"], 1)
        self.assertEqual(len(result.payload["invalidations"]), 1)

        contradictions = [
            edge
            for edge in runner.graph.edges.values()
            if edge.type is EdgeType.CONTRADICTS
            and edge.src == evidence
            and edge.dst == assumption.id
        ]
        self.assertEqual(
            len(contradictions),
            1,
            "8B must materialize exactly the evidence→assumption contradiction it audited",
        )
        self.assertEqual(
            sum(
                1
                for node in runner.graph.nodes.values()
                if node.type is NodeType.EVIDENCE and node.id == evidence
            ),
            1,
        )

    def test_recheck_checkpoint_contention_leaves_the_live_session_unchanged(self):
        self.submit_cve()
        runner = self.session.runner
        assert runner is not None
        hashing = runner["hashing"]
        assumption = runner.graph.assumptions[hashing.assumption_id]
        generation = self.session.generation
        head = runner.ledger.head
        contradictions_before = {
            edge.id
            for edge in runner.graph.edges.values()
            if edge.type is EdgeType.CONTRADICTS
        }

        assert self.session.path is not None
        with writer_lock(self.session.path):
            with self.assertRaises(ControlledWriteError):
                mesh_recheck(self.session, self.checkpointer)

        self.assertEqual(self.session.generation, generation)
        self.assertEqual(runner.ledger.head, head)
        self.assertIs(hashing.state, TaskState.DONE)
        self.assertIs(assumption.status, AssumptionStatus.ACTIVE)
        self.assertEqual(
            {
                edge.id
                for edge in runner.graph.edges.values()
                if edge.type is EdgeType.CONTRADICTS
            },
            contradictions_before,
        )

    def test_unknown_repair_key_fails_before_the_committed_session_changes(self):
        self.submit_cve()
        rechecked = mesh_recheck(self.session, self.checkpointer)
        self.session = rechecked.session
        runner = self.session.runner
        assert runner is not None
        generation = self.session.generation
        binding = dict(runner["hashing"].binding())
        head = runner.ledger.head

        with self.assertRaisesRegex(Exception, "not registered"):
            mesh_repair(
                self.session,
                self.checkpointer,
                task="hashing",
                worker_key="auth.hash.unknown.v9",
                assumes="replacement ground",
            )

        self.assertEqual(self.session.generation, generation)
        self.assertEqual(runner.ledger.head, head)
        self.assertEqual(runner["hashing"].binding(), binding)
        self.assertIs(runner["hashing"].state, TaskState.STALE)

    def test_repair_onto_the_rejected_ground_commits_nothing(self):
        """A client cannot re-ground stale work on the statement that just fell."""
        self.submit_cve()
        self.session = mesh_recheck(self.session, self.checkpointer).session
        runner = self.session.runner
        assert runner is not None
        generation = self.session.generation
        head = runner.ledger.head
        rejected = runner["hashing"].assumption_id

        with self.assertRaisesRegex(Exception, "rejected"):
            mesh_repair(
                self.session,
                self.checkpointer,
                task="hashing",
                worker_key="auth.hash.bcrypt.v1",
                assumes="Argon2 has no open advisory",
            )

        self.assertEqual(self.session.generation, generation)
        self.assertEqual(Session.load(self.root, registry=deployment()).generation, generation)
        self.assertEqual(runner.ledger.head, head)
        self.assertEqual(runner["hashing"].assumption_id, rejected)
        self.assertIsNone(self.session.graph.assumptions[rejected].supersedes)

    def test_repair_and_resume_rerun_only_the_stale_closure(self):
        self.submit_cve()
        self.session = mesh_recheck(self.session, self.checkpointer).session

        repaired = mesh_repair(
            self.session,
            self.checkpointer,
            task="hashing",
            worker_key="auth.hash.bcrypt.v1",
            assumes="bcrypt has no open advisory",
            produces=["Password Hasher"],
        )
        self.session = repaired.session
        runner = self.session.runner
        assert runner is not None
        self.assertEqual(runner["hashing"].worker_key, "auth.hash.bcrypt.v1")
        self.assertIs(runner["hashing"].state, TaskState.STALE)

        resumed = mesh_resume(self.session, self.checkpointer)
        self.session = resumed.session
        final = self.session.runner
        assert final is not None

        self.assertEqual(final["hashing"].output, {"impl": "bcrypt"})
        self.assertEqual(final["routes"].output, {"hasher": "bcrypt"})
        self.assertEqual(final["hashing"].attempt, 2)
        self.assertEqual(final["routes"].attempt, 2)
        self.assertEqual(final["schema"].attempt, 1)
        self.assertEqual(final["tokens"].attempt, 1)
        self.assertEqual(set(resumed.payload["executed"]), {"hashing", "routes"})
        self.assertEqual(set(resumed.payload["cached"]), {"schema", "tokens"})
        self.assertTrue(final.ledger.verify())

    def test_the_complete_flow_survives_fresh_session_objects(self):
        self.submit_cve()

        fresh_b = Session.load(self.root, registry=deployment())
        b_generation = fresh_b.generation
        cp_b = Checkpointer(fresh_b)
        after_recheck = mesh_recheck(fresh_b, cp_b).session
        self.assertEqual(after_recheck.generation, b_generation + 1)

        fresh_c = Session.load(self.root, registry=deployment())
        cp_c = Checkpointer(fresh_c)
        repaired = mesh_repair(
            fresh_c,
            cp_c,
            task="hashing",
            worker_key="auth.hash.bcrypt.v1",
            assumes="bcrypt has no open advisory",
            produces=["Password Hasher"],
        ).session
        assert repaired.runner is not None
        self.assertEqual(repaired.runner["hashing"].worker_key, "auth.hash.bcrypt.v1")
        self.assertEqual(repaired.runner["hashing"].attempt, 1)

        fresh_d = Session.load(self.root, registry=deployment())
        cp_d = Checkpointer(fresh_d)
        final_write = mesh_resume(fresh_d, cp_d)

        verified = Session.load(self.root, registry=deployment())
        runner = verified.runner
        assert runner is not None
        self.assertEqual(runner["hashing"].worker_key, "auth.hash.bcrypt.v1")
        self.assertEqual(runner["hashing"].output, {"impl": "bcrypt"})
        self.assertEqual(runner["routes"].output, {"hasher": "bcrypt"})
        self.assertEqual(
            {task.name: task.attempt for task in runner.tasks},
            {"schema": 1, "hashing": 2, "routes": 2, "tokens": 1},
        )
        self.assertTrue(runner.ledger.verify())
        self.assertEqual(runner.ledger.head, final_write.payload["ledger_head"])

    def test_8d_crosses_four_real_python_processes(self):
        scripts = [
            """
import json, runpy, sys
from contextmesh_mcp.session import Checkpointer, Session
from contextmesh_mcp.writes import mesh_submit_evidence
D = runpy.run_path('tests/test_controlled_writes.py', run_name='pr8defs')
s = Session.load(sys.argv[1], registry=D['deployment']())
cp = Checkpointer(s)
r = mesh_submit_evidence(
    s, cp,
    text='CVE-2026-9999 affects Argon2',
    source_id='source:vendor-advisory',
    external_id='CVE-2026-9999',
    metadata={'package': 'argon2', 'severity': 'critical'},
)
print(json.dumps({'generation': r.session.generation, 'evidence': r.payload['evidence_id']}))
""",
            """
import json, runpy, sys
from contextmesh_mcp.session import Checkpointer, Session
from contextmesh_mcp.writes import mesh_recheck, mesh_repair
D = runpy.run_path('tests/test_controlled_writes.py', run_name='pr8defs')
s = Session.load(sys.argv[1], registry=D['deployment']())
cp = Checkpointer(s)
s = mesh_recheck(s, cp).session
s = mesh_repair(
    s, cp,
    task='hashing',
    worker_key='auth.hash.bcrypt.v1',
    assumes='bcrypt has no open advisory',
    produces=['Password Hasher'],
).session
r = s.runner
assert r is not None
print(json.dumps({
    'generation': s.generation,
    'worker_key': r['hashing'].worker_key,
    'attempt': r['hashing'].attempt,
    'states': {t.name: t.state.value for t in r.tasks},
}))
""",
            """
import json, runpy, sys
from contextmesh_mcp.session import Checkpointer, Session
from contextmesh_mcp.writes import mesh_resume
D = runpy.run_path('tests/test_controlled_writes.py', run_name='pr8defs')
s = Session.load(sys.argv[1], registry=D['deployment']())
cp = Checkpointer(s)
r = mesh_resume(s, cp)
print(json.dumps({
    'generation': r.session.generation,
    'executed': r.payload['executed'],
    'cached': r.payload['cached'],
}))
""",
            """
import json, runpy, sys
from contextmesh_mcp.session import Session
D = runpy.run_path('tests/test_controlled_writes.py', run_name='pr8defs')
s = Session.load(sys.argv[1], registry=D['deployment']())
r = s.runner
assert r is not None
assert r.ledger.verify()
print(json.dumps({
    'generation': s.generation,
    'head': r.ledger.head,
    'worker_key': r['hashing'].worker_key,
    'hashing_output': r['hashing'].output,
    'routes_output': r['routes'].output,
    'attempts': {t.name: t.attempt for t in r.tasks},
}))
""",
        ]

        outputs = []
        for index, script in enumerate(scripts, start=1):
            proc = subprocess.run(
                [sys.executable, "-c", script, str(self.root)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"process {index} failed:\nstdout={proc.stdout}\nstderr={proc.stderr}",
            )
            outputs.append(json.loads(proc.stdout))

        self.assertEqual(outputs[1]["worker_key"], "auth.hash.bcrypt.v1")
        self.assertEqual(outputs[1]["attempt"], 1, "bcrypt must not run before restart")
        self.assertEqual(set(outputs[2]["executed"]), {"hashing", "routes"})
        self.assertEqual(set(outputs[2]["cached"]), {"schema", "tokens"})
        self.assertEqual(outputs[3]["worker_key"], "auth.hash.bcrypt.v1")
        self.assertEqual(outputs[3]["hashing_output"], {"impl": "bcrypt"})
        self.assertEqual(outputs[3]["routes_output"], {"hasher": "bcrypt"})
        self.assertEqual(
            outputs[3]["attempts"],
            {"schema": 1, "hashing": 2, "routes": 2, "tokens": 1},
        )
        self.assertEqual(
            [row["generation"] for row in outputs],
            sorted(row["generation"] for row in outputs),
        )


if __name__ == "__main__":
    unittest.main()
