"""Controlled MCP writes over a durable Context Mesh session.

The read tool layer deliberately knows nothing about persistence. Structural
writes need a stronger contract than ``mesh_ask`` telemetry: a successful reply
means the mutation is in the committed session generation. To make failure
atomic, writes are applied to a lossless in-memory clone first; only the clone is
made live after its manifest has committed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from contextmesh.assumptions import AssumptionLedger
from contextmesh.evidence import EvidenceIntakeError, submit_evidence
from contextmesh.execute import Event, RunLedger, Runner, TaskState
from contextmesh.graph import ContextGraph
from contextmesh.recheck import recheck as evidence_recheck
from contextmesh.resolve import Resolver

from .session import (
    Checkpointer,
    Session,
    SessionError,
    SessionLockedError,
    WalkerConfig,
)


class ControlledWriteError(EvidenceIntakeError):
    """A requested MCP mutation could not be committed durably."""


@dataclass(frozen=True)
class WriteResult:
    payload: Dict[str, Any]
    session: Session
    changed: bool


def _clone_session(source: Session) -> Session:
    """Losslessly clone the durable state without sharing mutable graph objects."""
    graph = ContextGraph.from_dict(source.graph.to_dict())
    resolver = Resolver.from_dict(source.resolver.to_dict())
    walker = WalkerConfig.of(source.walker).walker(graph, resolver)

    assumption_ledger = AssumptionLedger(graph)
    assumption_ledger.history = [dict(row) for row in source.ledger.history]

    runner: Optional[Runner] = None
    if source.runner is not None:
        registry = source.runner.registry
        if registry is None:
            raise ControlledWriteError(
                "the session has execution state but no TaskRegistry; controlled writes refuse it"
            )
        run_ledger = RunLedger.from_snapshot(
            source.runner.ledger.snapshot(),
            expect_head=source.runner.ledger.head,
        )
        runner = Runner.from_snapshot(
            source.runner.snapshot(),
            graph=graph,
            registry=registry,
            ledger=run_ledger,
        )
        # ``rounds`` is intentionally not durable, but a write inside one live
        # process should not erase the reports it already had in memory.
        runner.rounds = list(source.runner.rounds)
        runner.assumptions.history = [
            dict(row) for row in source.runner.assumptions.history
        ]

    return Session(
        graph=graph,
        resolver=resolver,
        walker=walker,
        ledger=assumption_ledger,
        rounds=source.rounds,
        source=source.source,
        path=source.path,
        _checkpoint_path=source._checkpoint_path,
        _checkpoint_identity=source._checkpoint_identity,
        generation=source.generation,
        runner=runner,
    )


def _require_durable(
    session: Session, checkpointer: Optional[Checkpointer]
) -> Checkpointer:
    if checkpointer is None:
        raise ControlledWriteError(
            "controlled writes require an active session checkpointer"
        )
    if checkpointer.session is not session:
        raise ControlledWriteError(
            "the checkpointer belongs to a different session; "
            "refusing to commit the wrong graph"
        )
    if session.path is None:
        raise ControlledWriteError(
            "controlled writes require a saved session directory; "
            "the demo graph is ephemeral"
        )
    if checkpointer.policy == "never":
        raise ControlledWriteError(
            "checkpoint policy 'never' cannot acknowledge a durable structural write"
        )
    return checkpointer


def _runner(session: Session) -> Runner:
    runner = session.runner
    if runner is None:
        raise ControlledWriteError(
            "this session has no execution plan; recheck, repair and resume require one"
        )
    if runner.registry is None:
        raise ControlledWriteError(
            "this execution plan has no TaskRegistry; controlled execution writes refuse it"
        )
    return runner


def _durable_digest(session: Session, generation: int) -> str:
    """Fingerprint exactly the durable state a generation is meant to publish.

    Generation numbers establish ordering, not authorship. Recovery after an
    exception therefore needs a second fact: the live generation must contain
    the state this transaction staged. The digest covers the manifest and every
    companion payload without adding a new field to the on-disk session schema.
    """
    manifest = json.dumps(
        session.manifest(generation),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    parts = [
        ("manifest", manifest),
        ("graph", session.graph.to_json()),
        ("resolver", session.resolver.to_json()),
    ]
    if session.runner is not None:
        parts.extend(
            [
                ("execution", session.runner.to_json()),
                ("ledger", session.runner.ledger.to_json()),
            ]
        )

    digest = hashlib.sha256()
    for name, payload in parts:
        encoded = payload.encode("utf-8")
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _live_commit_is_staged(
    staged: Session, expected_generation: int
) -> bool:
    """Prove that the live commit is this transaction before adopting it."""
    assert staged.path is not None
    registry = staged.runner.registry if staged.runner is not None else None
    try:
        target = staged._checkpoint_target()
        committed = Session.load(target, registry=registry)
        if (
            staged._checkpoint_identity is not None
            and committed._checkpoint_identity != staged._checkpoint_identity
        ):
            return False
        if committed.generation != expected_generation:
            return False
        return _durable_digest(committed, expected_generation) == _durable_digest(
            staged, expected_generation
        )
    except Exception:
        # Recovery is the fail-closed path. If the live directory cannot be
        # proved equivalent, the original checkpoint exception remains fatal.
        return False


def commit_mutation(
    session: Session,
    checkpointer: Optional[Checkpointer],
    mutate: Callable[[Session], Tuple[Dict[str, Any], bool]],
) -> WriteResult:
    """Run ``mutate`` on a clone and publish it only after the manifest commits."""
    cp = _require_durable(session, checkpointer)
    staged = _clone_session(session)
    payload, changed = mutate(staged)
    if not changed:
        return WriteResult(payload=payload, session=session, changed=False)

    before_generation = session.generation
    expected_generation = before_generation + 1
    try:
        staged.checkpoint()
    except SessionLockedError as exc:
        cp.contended += 1
        raise ControlledWriteError(str(exc)) from None
    except SessionError as exc:
        # SessionError includes compare-and-swap refusal. It is known to happen
        # before this writer replaces the manifest, so the live session remains
        # authoritative and needs no rollback at all.
        raise ControlledWriteError(str(exc)) from None
    except Exception as exc:
        # ``session.json`` is the commit point, but generation arithmetic alone
        # cannot tell which writer published it. Another writer can advance the
        # directory by exactly one while this checkpoint is failing. Recover
        # only when the actual committed state fingerprints as the staged state.
        if not _live_commit_is_staged(staged, expected_generation):
            raise ControlledWriteError(f"session checkpoint failed: {exc}") from exc
        staged.generation = expected_generation

    cp.session = staged
    cp.pending = 0
    cp.commits += 1
    return WriteResult(payload=payload, session=staged, changed=True)


def _receipt_payload(receipt: Any) -> Dict[str, Any]:
    node = receipt.node
    return {
        "evidence_id": receipt.evidence_id,
        "created": receipt.created,
        "node": {
            "id": node.id,
            "type": node.type.value,
            "label": node.label,
            "attrs": node.attrs,
            "provenance": node.provenance.to_dict() if node.provenance else None,
        },
    }


def mesh_submit_evidence(
    session: Session,
    checkpointer: Optional[Checkpointer],
    *,
    text: str,
    source_id: str,
    external_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> WriteResult:
    """Store one observation. It creates no edge and expresses no verdict."""

    def mutate(staged: Session) -> Tuple[Dict[str, Any], bool]:
        receipt = submit_evidence(
            staged.graph,
            text=text,
            source_id=source_id,
            external_id=external_id,
            metadata=metadata,
        )
        return _receipt_payload(receipt), receipt.created

    return commit_mutation(session, checkpointer, mutate)


def mesh_recheck(
    session: Session,
    checkpointer: Optional[Checkpointer],
) -> WriteResult:
    """Ask registered auditors to interpret standing work; the client supplies no verdict."""

    def mutate(staged: Session) -> Tuple[Dict[str, Any], bool]:
        runner = _runner(staged)
        before = len(runner.ledger.entries)
        reports = evidence_recheck(runner, require_evidence=True)
        new_entries = runner.ledger.entries[before:]
        audited = sum(1 for entry in new_entries if entry.event is Event.AUDITED)
        audit_errors = [
            {"task": entry.task, "error": entry.detail}
            for entry in new_entries
            if entry.event is Event.AUDIT_ERROR
        ]
        return (
            {
                "round": runner.round,
                "audited": audited,
                "audit_errors": audit_errors,
                "invalidations": [report.to_dict() for report in reports],
                "ledger_head": runner.ledger.head,
            },
            True,
        )

    return commit_mutation(session, checkpointer, mutate)


def _non_empty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControlledWriteError(f"{name} must be a non-empty string")
    return value


def mesh_repair(
    session: Session,
    checkpointer: Optional[Checkpointer],
    *,
    task: str,
    worker_key: str,
    assumes: str,
    auditor_key: Optional[str] = None,
    produces: Optional[List[str]] = None,
    rationale: Optional[str] = None,
) -> WriteResult:
    """Re-ground stale work using only deployment-owned TaskRegistry keys."""
    task = _non_empty(task, "task")
    worker_key = _non_empty(worker_key, "worker_key")
    assumes = _non_empty(assumes, "assumes")
    if auditor_key is not None:
        auditor_key = _non_empty(auditor_key, "auditor_key")
    if rationale is not None and not isinstance(rationale, str):
        raise ControlledWriteError("rationale must be a string or null")
    if produces is not None:
        if not isinstance(produces, list) or not all(
            isinstance(item, str) and item for item in produces
        ):
            raise ControlledWriteError("produces must be an array of non-empty strings")

    def mutate(staged: Session) -> Tuple[Dict[str, Any], bool]:
        runner = _runner(staged)
        try:
            target = runner[task]
        except KeyError:
            raise ControlledWriteError(f"task {task!r} is not in this execution plan") from None
        if target.state not in (TaskState.STALE, TaskState.FAILED):
            raise ControlledWriteError(
                f"task {task!r} is {target.state.value}, not stale or failed; "
                "controlled repair does not replace healthy cached work"
            )
        new = runner.repair(
            task,
            assumes=assumes,
            worker_key=worker_key,
            auditor_key=auditor_key,
            produces=produces,
            rationale=rationale,
        )
        rebound = runner[task]
        return (
            {
                "task": task,
                "state": rebound.state.value,
                "worker_key": rebound.worker_key,
                "auditor_key": rebound.auditor_key,
                "assumption_id": new.id,
                "assumption": new.statement,
                "ledger_head": runner.ledger.head,
            },
            True,
        )

    return commit_mutation(session, checkpointer, mutate)


def mesh_resume(
    session: Session,
    checkpointer: Optional[Checkpointer],
) -> WriteResult:
    """Run only pending/stale work; DONE tasks stay cached under Runner.run()."""

    def mutate(staged: Session) -> Tuple[Dict[str, Any], bool]:
        runner = _runner(staged)
        unsettled = [
            task.name
            for task in runner.tasks
            if task.state in (TaskState.PENDING, TaskState.STALE)
        ]
        if not unsettled:
            return (
                {
                    "round": runner.round,
                    "executed": [],
                    "cached": [task.name for task in runner.tasks if task.state is TaskState.DONE],
                    "changed": False,
                    "ledger_head": runner.ledger.head,
                },
                False,
            )
        report = runner.run()
        payload = report.to_dict()
        payload["ledger_head"] = runner.ledger.head
        payload["changed"] = True
        return payload, True

    return commit_mutation(session, checkpointer, mutate)


WRITE_TOOLS: Dict[str, Dict[str, Any]] = {
    "mesh_submit_evidence": {
        "fn": mesh_submit_evidence,
        "description": (
            "Submit a raw observation as an EVIDENCE node. The client supplies "
            "no edge, assumption, verdict, rejection or invalidation target; "
            "successful new evidence is committed before this call returns."
        ),
    },
    "mesh_recheck": {
        "fn": mesh_recheck,
        "description": (
            "Ask the execution plan's registered auditors to recheck standing work. "
            "The client supplies no audit verdict or assumption target; a disproof "
            "must identify pre-ingested evidence and is committed before return."
        ),
    },
    "mesh_repair": {
        "fn": mesh_repair,
        "description": (
            "Repair stale or failed work by selecting worker/auditor keys already "
            "registered in this deployment. No callable, module path or import "
            "string crosses MCP."
        ),
    },
    "mesh_resume": {
        "fn": mesh_resume,
        "description": (
            "Resume pending or stale execution through the native scheduler. "
            "Previously DONE work remains cached and is not re-executed."
        ),
    },
}


def names() -> list[str]:
    return sorted(WRITE_TOOLS)


__all__ = [
    "ControlledWriteError",
    "WRITE_TOOLS",
    "WriteResult",
    "commit_mutation",
    "mesh_recheck",
    "mesh_repair",
    "mesh_resume",
    "mesh_submit_evidence",
    "names",
]
