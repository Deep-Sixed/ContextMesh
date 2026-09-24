"""Evidence-bound rechecks for controlled MCP writes.

PR #8A lets an untrusted client park an observation in the graph without
supplying any semantic edge or verdict.  PR #8B is the other half of that
boundary: a deployment-owned auditor may inspect those observations and, if it
finds that its task's ground no longer holds, identify the *existing* EVIDENCE
node that supports the disproof.

This module deliberately does not replace :class:`contextmesh.execute.Runner`.
It drives the Runner's registered auditors and native AssumptionLedger, and it
keeps the old in-process behaviour available: callers that do not require an
external evidence id may still use ``Runner.recheck()`` unchanged.  The MCP
write path calls :func:`recheck` with ``require_evidence=True`` so a remote
recheck can never manufacture its own observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from .assumptions import InvalidationReport
from .execute import (
    AuditContext,
    Event,
    ExecutionError,
    Runner,
    Task,
    TaskState,
    Verdict,
)
from .model import Assumption, AssumptionStatus, Node, NodeType


class EvidenceRecheckError(ExecutionError):
    """A registered auditor's evidence-bound finding cannot be applied safely."""


@dataclass(frozen=True)
class EvidenceVerdict(Verdict):
    """A normal audit verdict plus the observation that supports a disproof."""

    evidence_id: Optional[str] = None


@dataclass(frozen=True)
class EvidenceAuditContext(AuditContext):
    """The native audit context with an optional pre-ingested evidence binding."""

    def ok(self, reason: str) -> EvidenceVerdict:
        return EvidenceVerdict(True, reason)

    def fail(self, reason: str) -> EvidenceVerdict:
        return EvidenceVerdict(False, reason)

    def disproved(
        self, reason: str, *, evidence_id: Optional[str] = None
    ) -> EvidenceVerdict:
        if evidence_id is not None:
            if not isinstance(evidence_id, str) or not evidence_id:
                raise EvidenceRecheckError(
                    "evidence_id must be a non-empty string when a disproof binds evidence"
                )
        return EvidenceVerdict(False, reason, disproves=True, evidence_id=evidence_id)


def _coerce(result: Any) -> EvidenceVerdict:
    if isinstance(result, EvidenceVerdict):
        return result
    if isinstance(result, Verdict):
        return EvidenceVerdict(result.ok, result.reason, result.disproves)
    if isinstance(result, bool):
        return EvidenceVerdict(
            result,
            "audit passed" if result else "audit failed",
        )
    raise EvidenceRecheckError(
        f"an auditor must return a Verdict or a bool, got {type(result).__name__}"
    )


def _evidence(runner: Runner, evidence_id: str) -> Node:
    node = runner.graph.nodes.get(evidence_id)
    if node is None:
        raise EvidenceRecheckError(
            f"auditor supplied evidence_id {evidence_id!r}, but it is not in this graph"
        )
    if node.type is not NodeType.EVIDENCE:
        raise EvidenceRecheckError(
            f"auditor supplied evidence_id {evidence_id!r}, but it is a "
            f"{node.type.value}, not evidence"
        )
    if node.invalidated:
        raise EvidenceRecheckError(
            f"auditor supplied evidence_id {evidence_id!r}, but that evidence is invalidated"
        )
    return node


def _disprove_with_evidence(
    runner: Runner,
    task: Task,
    assumption: Assumption,
    verdict: EvidenceVerdict,
) -> InvalidationReport:
    """Reject one ground using the exact evidence node the auditor identified."""
    assert verdict.evidence_id is not None
    evidence = _evidence(runner, verdict.evidence_id)
    report = runner.assumptions.reject(assumption.id, evidence_id=evidence.id)

    runner.ledger.record(
        runner.round,
        Event.DISPROVED,
        task.name,
        verdict.reason,
        node_id=evidence.id,
        assumption_id=assumption.id,
        data={
            "assumption_id": assumption.id,
            "assumption": assumption.statement,
            "evidence_id": evidence.id,
            "reason": verdict.reason,
            "disproved_by": task.name,
            "invalidated": {
                node_id: list(chain) for node_id, chain in report.invalidated.items()
            },
            "preserved": list(report.preserved),
        },
    )

    for other in runner.tasks:
        grounded_on_it = other.assumption_id == assumption.id
        in_radius = bool(other.node_id) and other.node_id in report.invalidated
        if not grounded_on_it and not in_radius:
            continue
        if other.state is TaskState.DONE or other is task:
            other.state = TaskState.STALE
            runner.ledger.record(
                runner.round,
                Event.INVALIDATED,
                other.name,
                (
                    report.why(other.node_id)
                    if in_radius
                    else f"stood on: {assumption.statement}"
                ),
                node_id=other.node_id,
                assumption_id=assumption.id,
            )
    return report


def recheck(
    runner: Runner, *, require_evidence: bool = False
) -> List[InvalidationReport]:
    """Re-audit standing work without executing workers.

    Auditors are the same callables bound through the Runner's TaskRegistry.
    They receive a graph-reading context and may return ordinary ``Verdict`` or
    ``bool`` values exactly as before.  An evidence-aware auditor can instead
    call ``ctx.disproved(reason, evidence_id=...)``.

    With ``require_evidence=True`` (the MCP contract), a disproving verdict that
    does not bind a pre-existing EVIDENCE node is refused.  The caller cannot
    provide the verdict or the evidence id; only the registered auditor can.
    """
    runner.round += 1
    runner.graph.build += 1
    reports: List[InvalidationReport] = []

    for task in runner.tasks:
        if task.state is not TaskState.DONE or task.audit is None:
            continue
        assumption = runner.graph.assumptions.get(task.assumption_id or "")
        if assumption is None or assumption.status is not AssumptionStatus.ACTIVE:
            continue

        try:
            result = task.audit(
                EvidenceAuditContext(
                    task=task,
                    output=task.output,
                    assumption=assumption,
                    graph=runner.graph,
                )
            )
        except EvidenceRecheckError:
            # Raised by ctx.disproved() itself: the auditor broke the evidence
            # contract. That is the same violation as a disproof returned
            # without evidence, which is fatal below, so it is fatal here too
            # and the staged MCP transaction is discarded rather than
            # committing it as an ordinary auditor outage.
            raise
        except Exception as exc:
            # Match Runner.recheck(): an unavailable auditor is never a
            # disproof, must not prevent independent rechecks, and does not
            # fail work that already passed its audit. Record it; the task
            # stays DONE and the next recheck asks again.
            runner.ledger.record(
                runner.round,
                Event.AUDIT_ERROR,
                task.name,
                f"auditor error: {type(exc).__name__}: {exc}",
                node_id=task.node_id,
                assumption_id=assumption.id,
            )
            continue
        verdict = _coerce(result)

        # Resolve the evidence before writing the audit receipt or touching the
        # assumption.  MCP applies this whole function to a staged Session, so
        # any exception also discards the round/build changes with the clone.
        if verdict.disproves:
            if verdict.evidence_id is None:
                if require_evidence:
                    raise EvidenceRecheckError(
                        f"auditor for {task.name!r} disproved its ground without "
                        "identifying pre-ingested evidence"
                    )
            else:
                _evidence(runner, verdict.evidence_id)

        runner.ledger.record(
            runner.round,
            Event.AUDITED,
            task.name,
            f"{'holds' if verdict.ok else 'fails'}: {verdict.reason}",
            node_id=task.node_id,
            assumption_id=assumption.id,
        )
        if verdict.ok:
            continue
        if verdict.disproves:
            if verdict.evidence_id is None:
                # Backward-compatible in-process behaviour. Controlled MCP
                # rechecks use require_evidence=True and can never reach this.
                reports.append(runner._disprove(task, assumption, verdict.reason))
            else:
                reports.append(
                    _disprove_with_evidence(runner, task, assumption, verdict)
                )
        else:
            task.state = TaskState.FAILED
            runner.ledger.record(
                runner.round,
                Event.FAILED,
                task.name,
                verdict.reason,
                node_id=task.node_id,
            )
    return reports


__all__ = [
    "EvidenceAuditContext",
    "EvidenceRecheckError",
    "EvidenceVerdict",
    "recheck",
]
