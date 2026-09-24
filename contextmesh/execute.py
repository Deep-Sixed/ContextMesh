"""Selective re-execution: what has to *run again* when the ground moves.

``assumptions.reject`` answers what falls. Until this module, nothing answered
what has to be recomputed, so rejecting an assumption left a graph full of nodes
flagged ``invalidated`` and no work scheduled to replace them — the expensive
half of the idea (redo the blast radius, keep everything else cached) was
described but never performed.

A :class:`Runner` binds units of work to the graph's own vocabulary, so
scheduling reads off the ontology instead of a second copy of it kept here:

===============  =========================================
``task``         a ``decision`` node
``task.assumes`` ``decision -[depends_on]-> assumption``
``task.needs``   ``decision -[depends_on]-> decision``
``task.produces````decision -[produces]-> entity``
an audit         ``decision -[justified_by]-> evidence``
a disproof       ``evidence -[contradicts]-> assumption``
===============  =========================================

Two rules keep re-execution compatible with an append-only history:

1. **A decision is never revived; it is superseded.** Re-running a step appends
   a new ``decision`` node that ``supersedes`` the invalidated one, so the old
   reasoning stays walkable (GRAPH.md rule 3).
2. **An artefact is revived, because it was rebuilt.** An ``entity`` fell only
   because the decision that produced it fell. When that work is redone the
   thing exists again — under the same id, since it is the same real thing. An
   artefact the rerun *stops* producing simply stays invalidated.

And one rule keeps invalidation honest: an assumption is rejected here only
because an auditor disproved it. :meth:`Runner.recheck` re-runs every standing
auditor against current facts without re-executing anything, and a disproof
there is what computes a blast radius and schedules the reruns. No caller
reaches in and flips a node.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from .assumptions import AssumptionLedger, InvalidationReport
from .decisions import DecisionLog
from .graph import ContextGraph
from .model import Assumption, AssumptionStatus, EdgeType, NodeType, slug
from .ontology import OntologyError


class ExecutionError(Exception):
    """Raised when a plan cannot be scheduled, or an auditor misbehaves."""


class _AuditorRaised(ExecutionError):
    """Internal wrapper for an exception raised by an auditor callback."""


class LedgerIntegrityError(ExecutionError):
    """Raised when a serialised ledger is not the history it claims to be.

    Separate from :class:`ExecutionCheckpointError`, which is about *identity*
    — a key this deployment cannot bind. This one is about *history*: the file
    parses, but its entries, its ordering or its digests do not add up, or it
    is a perfectly self-consistent chain that is simply not the one committed.
    """


class ExecutionSnapshotError(ExecutionError):
    """Raised when a serialised execution state is not one this build can restore.

    Distinct from :class:`ExecutionCheckpointError`, which is about a *key* this
    deployment cannot bind, and from :class:`LedgerIntegrityError`, which is
    about a *history* that does not add up. This one is about the plan: a field
    of the wrong shape, a task named twice, or a reference into a graph that
    does not hold what the snapshot says it holds.
    """


class ExecutionCheckpointError(ExecutionError):
    """Raised when durable execution identity is missing, wrong, or ambiguous.

    Separate from ``ExecutionError`` so a caller can tell "this plan will not
    schedule" from "this plan will not survive a restart", but a subclass of it
    so existing ``except ExecutionError`` handlers keep working.
    """


class TaskState(str, Enum):
    """What the runner believes about a task.

    ``BLOCKED`` is deliberately absent: being blocked is a fact about the
    current round, not about the task, so it is recomputed every run rather
    than stored and left to go stale.
    """

    PENDING = "pending"
    DONE = "done"
    STALE = "stale"
    FAILED = "failed"


class Event(str, Enum):
    EXECUTED = "executed"
    AUDITED = "audited"
    DISPROVED = "disproved"
    INVALIDATED = "invalidated"
    REPAIRED = "repaired"
    CACHED = "cached"
    BLOCKED = "blocked"
    FAILED = "failed"
    #: A recheck auditor raised instead of returning a verdict. Recorded
    #: without changing the task: the work was verified when it ran, and an
    #: auditor that could not answer now proves nothing either way.
    AUDIT_ERROR = "audit_error"


# ── verdicts ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Verdict:
    """An auditor's finding.

    ``disproves`` is the whole distinction. "The output is wrong" and "the
    ground this stood on is false" are different failures with different
    blast radii, and an auditor that cannot say which one it found forces the
    engine to guess. A plain failure fails one task; a disproof rejects an
    assumption and takes down everything that stood on it.
    """

    ok: bool
    reason: str
    disproves: bool = False


def _coerce(result: Any) -> Verdict:
    if isinstance(result, Verdict):
        return result
    if isinstance(result, bool):
        # A bare ``False`` is read as "the work is wrong", never as a disproof.
        # Tearing down a blast radius is not something to infer from a boolean.
        return Verdict(result, "audit passed" if result else "audit failed")
    raise ExecutionError(
        f"an auditor must return a Verdict or a bool, got {type(result).__name__}"
    )


@dataclass(frozen=True)
class RunContext:
    """What a worker is handed: its ground, and its dependencies' outputs."""

    task: "Task"
    attempt: int
    graph: ContextGraph
    assumption: Assumption
    inputs: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class AuditContext:
    """What an auditor is handed: the output, and the assumption it must hold under."""

    task: "Task"
    output: Dict[str, Any]
    assumption: Assumption
    graph: ContextGraph

    def ok(self, reason: str) -> Verdict:
        return Verdict(True, reason)

    def fail(self, reason: str) -> Verdict:
        """The work is wrong. Fails this task; invalidates nothing."""
        return Verdict(False, reason)

    def disproved(self, reason: str) -> Verdict:
        """The ground is false. Rejects the assumption and invalidates its radius."""
        return Verdict(False, reason, disproves=True)


Worker = Callable[[RunContext], Dict[str, Any]]
Auditor = Callable[[AuditContext], Any]


def _valid_key(key: Any, kind: str) -> str:
    """A registry key has to survive a round trip through a file.

    Not a naming scheme — dots, versions and namespaces are conventions the
    caller picks. Only the properties a *durable identifier* needs: it is text,
    it is not empty, and it is not carrying whitespace or control characters
    that a log line, a diff or a JSON reader would quietly change.
    """
    if not isinstance(key, str):
        raise ExecutionCheckpointError(
            f"{kind} key must be a string, got {type(key).__name__}"
        )
    if not key:
        raise ExecutionCheckpointError(f"{kind} key must not be empty")
    if key != key.strip():
        raise ExecutionCheckpointError(
            f"{kind} key {key!r} has leading or trailing whitespace; a durable "
            "identifier has to be exactly what it looks like"
        )
    if any(ch.isspace() for ch in key):
        # Tabs, newlines and the exotic spaces are already non-printable, so in
        # practice this catches the interior ASCII space — the one whitespace
        # character that survives a JSON round trip looking innocent and then
        # splits a log column, a shell word or a CSV cell in half. Tightening a
        # key format once checkpoints exist on disk is the expensive direction,
        # so it is refused now rather than regretted later.
        raise ExecutionCheckpointError(
            f"{kind} key {key!r} contains whitespace; a durable identifier has "
            "to survive a log line, a shell word and a diff unsplit"
        )
    if not key.isprintable():
        raise ExecutionCheckpointError(
            f"{kind} key {key!r} contains a control character or newline"
        )
    return key


class TaskRegistry:
    """The one place a durable key becomes executable code.

    A checkpoint cannot hold a Python callable, and it must not hold anything
    that could *become* one — no module path to import, no qualified name to
    look up, no pickle. What it holds is a key, and a key means something only
    because a running process was configured to say so. That configuration is
    this object: deployment state, never file state.

    So the restore path is a lookup and nothing else::

        checkpoint ──"auth.hash.bcrypt.v1"──► registry ──► callable

    If this deployment never registered that key, the restore fails and says
    which key is missing. It does not fall back to a similar one, guess from a
    module path, or leave a half-bound runner behind.

    Workers and auditors are kept in separate namespaces on purpose. They have
    different signatures and different authority — an auditor may disprove an
    assumption, which is the one thing a worker must never do — so a key that
    fits one is not silently accepted for the other.
    """

    def __init__(self) -> None:
        self._workers: Dict[str, Worker] = {}
        self._auditors: Dict[str, Auditor] = {}

    # ── registration ─────────────────────────────────────────────────────
    def _register(
        self,
        table: Dict[str, Any],
        other: Dict[str, Any],
        kind: str,
        key: str,
        fn: Any,
    ) -> None:
        _valid_key(key, kind)
        if not callable(fn):
            raise ExecutionCheckpointError(
                f"{kind} {key!r} must be callable, got {type(fn).__name__}"
            )
        if key in table:
            # Refused even when ``fn is table[key]``. A key is an identity, and
            # re-registering one is either a copy-paste or a genuine swap; both
            # want to be seen. Silently replacing a bcrypt worker with an argon2
            # one because they share a key is the failure this whole layer is
            # built to prevent, and it should not be reachable at startup either.
            raise ExecutionCheckpointError(
                f"{kind} key {key!r} is already registered; a durable key names "
                "one implementation for the life of the process"
            )
        if key in other:
            raise ExecutionCheckpointError(
                f"{key!r} is already registered as "
                f"{'an auditor' if kind == 'worker' else 'a worker'}; workers and "
                "auditors do not share a namespace"
            )
        table[key] = fn

    def register_worker(self, key: str, fn: Worker) -> None:
        self._register(self._workers, self._auditors, "worker", key, fn)

    def register_auditor(self, key: str, fn: Auditor) -> None:
        self._register(self._auditors, self._workers, "auditor", key, fn)

    # ── resolution ───────────────────────────────────────────────────────
    def _resolve(
        self,
        table: Dict[str, Any],
        other: Dict[str, Any],
        kind: str,
        key: str,
    ) -> Any:
        _valid_key(key, kind)
        if key in table:
            return table[key]
        if key in other:
            # The most confusing failure to debug, so it is named rather than
            # reported as a plain absence.
            raise ExecutionCheckpointError(
                f"{key!r} is registered as "
                f"{'an auditor' if kind == 'worker' else 'a worker'}, not as "
                f"{'a worker' if kind == 'worker' else 'an auditor'}"
            )
        raise ExecutionCheckpointError(
            f"{kind} {key!r} is not registered in this process; a checkpoint "
            "that names it cannot be restored here"
        )

    def worker(self, key: str) -> Worker:
        return self._resolve(self._workers, self._auditors, "worker", key)

    def auditor(self, key: str) -> Auditor:
        return self._resolve(self._auditors, self._workers, "auditor", key)

    def has_worker(self, key: str) -> bool:
        return isinstance(key, str) and key in self._workers

    def has_auditor(self, key: str) -> bool:
        return isinstance(key, str) and key in self._auditors

    # ── reporting ────────────────────────────────────────────────────────
    def describe(self) -> Dict[str, List[str]]:
        """What this deployment can bind, without exposing what it binds to.

        Keys only — never the callables, their module paths or their
        qualified names, since any of those would be a route back to importing
        code the checkpoint named. Sorted so two deployments can be diffed:
        "this one cannot resume that checkpoint because it lacks
        ``auth.hash.bcrypt.v1``" is the question this exists to answer.
        """
        return {
            "workers": sorted(self._workers),
            "auditors": sorted(self._auditors),
        }

    def __repr__(self) -> str:
        return (
            f"<TaskRegistry workers={len(self._workers)} "
            f"auditors={len(self._auditors)}>"
        )


@dataclass
class Task:
    """A unit of work and the assumption it is only valid under."""

    name: str
    title: str
    run: Worker
    assumes: str
    rationale: str = ""
    needs: Tuple[str, ...] = ()
    produces: Tuple[str, ...] = ()
    audit: Optional[Auditor] = None

    # ── durable identity ─────────────────────────────────────────────────
    #: What this task's code is *called*, as opposed to what it currently *is*.
    #: ``run`` is a callable and dies with the process; ``worker_key`` is a
    #: string a later process can look up in its own ``TaskRegistry``. A task
    #: may have either — a plain callable runs perfectly well in memory and
    #: simply cannot be checkpointed — but never both, because two sources of
    #: truth for "which code is this" can disagree, and the checkpoint would
    #: record the wrong one.
    worker_key: Optional[str] = None
    auditor_key: Optional[str] = None

    #: The Runner whose registry defines what this task's keys mean, if any.
    #: A task declared through ``Runner.task`` is governed by that Runner, and
    #: rebinding it against some other registry is refused — otherwise one task
    #: could run code from a table the Runner never adopted while its binding
    #: still named a key the Runner resolves differently. Never serialised: a
    #: checkpoint carries keys, not the tables that explain them.
    governed_by: Optional["Runner"] = field(default=None, repr=False, compare=False)

    # ── runner-owned state ───────────────────────────────────────────────
    state: TaskState = TaskState.PENDING
    attempt: int = 0
    node_id: Optional[str] = None
    assumption_id: Optional[str] = None
    output: Dict[str, Any] = field(default_factory=dict)
    artefacts: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "state": self.state.value,
            "attempt": self.attempt,
            "assumes": self.assumes,
            "needs": list(self.needs),
            "produces": list(self.produces),
            "audited": self.audit is not None,
            "node_id": self.node_id,
            "assumption_id": self.assumption_id,
            "artefacts": list(self.artefacts),
            # Strings, never the callables. This is the only place execution
            # identity crosses into data, and what crosses is a name.
            "worker_key": self.worker_key,
            "auditor_key": self.auditor_key,
        }

    # ── durable identity ─────────────────────────────────────────────────
    @property
    def checkpointable(self) -> bool:
        """Whether a later process could rebuild this task's code from a name."""
        return self.unbindable_reason() is None

    def unbindable_reason(self) -> Optional[str]:
        """Why this task could not be restored, or ``None`` if it could.

        A reason rather than a bare ``False``, because "this plan cannot be
        checkpointed" is useless without "and here is the task that stopped it".
        """
        if self.worker_key is None:
            return (
                f"task {self.name!r} runs a plain callable with no worker_key, so "
                "no later process could find its code again"
            )
        if self.audit is not None and self.auditor_key is None:
            return (
                f"task {self.name!r} has an auditor with no auditor_key; the "
                "audit would be silently dropped on restore"
            )
        return None

    def require_checkpointable(self) -> None:
        reason = self.unbindable_reason()
        if reason is not None:
            raise ExecutionCheckpointError(reason)

    def binding(self) -> Dict[str, Optional[str]]:
        """The durable half of this task: two names, no code."""
        return {"worker_key": self.worker_key, "auditor_key": self.auditor_key}

    def snapshot(self) -> Dict[str, Any]:
        """Everything a later process needs to be *this* task again.

        Not :meth:`to_dict`, which is the report view: that one carries the
        derived ``audited`` flag and omits ``rationale`` and ``output``, because
        it answers "what happened" rather than "what state is this in".
        """
        return {
            "name": self.name,
            "title": self.title,
            "rationale": self.rationale,
            "state": self.state.value,
            "attempt": self.attempt,
            "assumes": self.assumes,
            "needs": list(self.needs),
            "produces": list(self.produces),
            "worker_key": self.worker_key,
            "auditor_key": self.auditor_key,
            "node_id": self.node_id,
            "assumption_id": self.assumption_id,
            "output": self._storable_output(),
            "artefacts": list(self.artefacts),
        }

    def _storable_output(self) -> Dict[str, Any]:
        """A worker may return anything; what may be *stored* is narrower.

        Refused at the write rather than the read, because here the run that
        produced it is still in front of you. A set has no order and a NaN is
        not JSON, so either would come back rendered differently than it went in.
        """
        try:
            return _canonical(dict(self.output), path="output")
        except ExecutionError as exc:
            raise ExecutionSnapshotError(
                f"task {self.name!r} cannot be checkpointed: {exc}"
            ) from None

    def rebind(self, registry: "TaskRegistry") -> "Task":
        """Reconnect this task's callables from its keys, or refuse.

        The restore side of the boundary. It resolves; it does not discover —
        there is no import, no module path, no fallback to a similar key. A
        task whose worker key this deployment never registered does not come
        back half-bound and ready to run the wrong thing.

        A task that belongs to a Runner rebinds through that Runner. Doing it
        here with some other registry would put the Runner's own invariant back
        in play — one key meaning two implementations — one task at a time.
        """
        if self.governed_by is not None and registry is not self.governed_by.registry:
            raise ExecutionCheckpointError(
                f"task {self.name!r} belongs to a Runner and its keys mean what "
                "that Runner's registry says; rebind the Runner instead of "
                "rebinding one task against a different table"
            )
        self.require_checkpointable()
        assert self.worker_key is not None  # require_checkpointable proved it
        run = registry.worker(self.worker_key)
        audit = registry.auditor(self.auditor_key) if self.auditor_key else None
        # Resolved both before mutating either: a task that fails on its
        # auditor key must not be left holding a new worker.
        self.run = run
        self.audit = audit
        return self


# ── the ledger ───────────────────────────────────────────────────────────
#: The durable ledger format. Versioned separately from the graph snapshot and
#: the session manifest, because these three change for different reasons and a
#: single version number would force a rewrite of all of them to move one.
LEDGER_SCHEMA = "contextmesh.runledger"
LEDGER_VERSION = 1

#: A SHA-256 digest as it is written down: 64 lowercase hex characters. Matched
#: exactly rather than parsed loosely, so an uppercase or truncated digest is a
#: refusal at the door instead of a mismatch three checks later.
_DIGEST = re.compile(r"\A[0-9a-f]{64}\Z")

#: The fields a serialised entry has — all of them, and only these. An unknown
#: key is refused rather than dropped: the digest is taken over a fixed set of
#: fields, so anything extra in the file is content the chain does not cover,
#: and a later reader that started honouring it would be trusting unsigned data.
_ENTRY_KEYS = frozenset(
    {
        "seq",
        "round",
        "event",
        "task",
        "detail",
        "node_id",
        "assumption_id",
        "data",
        "digest",
    }
)


#: The fields the v1 container has — all of them, and only these. Exact for the
#: same reason an entry record is exact: a file carrying `approved_by` or an
#: `external_anchor` is a file whose meaning a v1 reader would silently discard,
#: and a later version that gives those names semantics would then disagree with
#: every v1 reader about what the same bytes said.
_LEDGER_KEYS = frozenset({"schema", "version", "head", "entries"})


class _DuplicateKey(ValueError):
    """Internal. Raised inside the JSON hook, re-raised as the format's own error.

    The hook is shared by both durable formats, so it cannot name either one —
    ``_parse_durable_json`` translates it into whichever error the caller's
    format uses.
    """


def _strict_object(pairs: Any) -> Dict[str, Any]:
    """json object hook. Two keys with one name have no agreed meaning.

    Python keeps the last, other parsers keep the first, and some refuse the
    document outright. That is tolerable in a config file and not here: these
    files exist to be read back by an implementation that is not this one, and
    for the ledger a digest only means something if every reader agrees which
    JSON value it was taken over. Applied through ``object_pairs_hook`` so it
    reaches the container, each record, and every nested payload alike.
    """
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(
                f"duplicate JSON key {key!r}; two values under one name mean "
                "different things to different parsers, so readers would "
                "disagree about what this file says"
            )
        result[key] = value
    return result


def _parse_durable_json(text: str, error: type) -> Any:
    """Read a durable file, with syntax failures inside the format's own error.

    A caller catching ``LedgerIntegrityError`` or ``ExecutionSnapshotError``
    should not also have to catch ``json.JSONDecodeError`` to cover "the file is
    truncated" — that is the same answer as any other unreadable file, arriving
    under a different name because of how it happens to be parsed.
    """
    try:
        return json.loads(
            text,
            parse_constant=_no_ledger_constants,
            object_pairs_hook=_strict_object,
        )
    except (_DuplicateKey, _NonJsonConstant) as exc:
        # Checked before JSONDecodeError: all of these are ValueErrors.
        raise error(str(exc)) from None
    except json.JSONDecodeError as exc:
        raise error(f"file is not valid JSON: {exc}") from None


class _NonJsonConstant(ValueError):
    """Internal, for the same reason as :class:`_DuplicateKey`."""


def _no_ledger_constants(value: str) -> float:
    """json.load hook. ``NaN``/``Infinity`` are not JSON and do not round-trip."""
    raise _NonJsonConstant(
        f"contains the non-JSON constant {value!r}; a value other parsers "
        "refuse is not a value anyone else can read back"
    )


#: Values a ledger entry's structured payload may contain. Anything else has no
#: single canonical JSON form — a set has no order, bytes have no encoding, NaN
#: is not JSON at all — so a digest taken over it would depend on how Python
#: happened to render it that run. Rejecting them at the door keeps the chain
#: meaningful rather than merely computed.
_JSON_SCALARS = (str, bool, int, float, type(None))


def _canonical(value: Any, *, path: str = "data") -> Any:
    """Return `value` in its canonical JSON form, or refuse it."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ExecutionError(f"{path}: {value!r} has no JSON form")
        return value
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise ExecutionError(f"{path}: keys must be strings, got {key!r}")
        return {k: _canonical(v, path=f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        # A tuple round-trips out of JSON as a list, so store it as one now
        # rather than letting the type change between write and read.
        return [_canonical(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
    raise ExecutionError(
        f"{path}: {type(value).__name__} is not JSON-deterministic; "
        f"ledger data accepts {', '.join(t.__name__ for t in _JSON_SCALARS)}, "
        f"list, tuple and dict with string keys"
    )


@dataclass(frozen=True)
class LedgerEntry:
    seq: int
    round: int
    event: Event
    task: str
    detail: str
    node_id: Optional[str] = None
    assumption_id: Optional[str] = None
    #: Structured payload. On a DISPROVED entry this is the whole blast-radius
    #: receipt, so the event can be read off the ledger without the graph.
    data: Dict[str, Any] = field(default_factory=dict)
    digest: str = ""

    def payload(self, previous: str) -> bytes:
        """The exact bytes the digest is taken over.

        Canonical JSON rather than delimiter-joined fields: `detail` is free
        text, and any separator that can appear inside a field makes two
        different entries capable of producing one payload. Sorted keys and no
        whitespace make the encoding a function of the values alone.
        """
        return json.dumps(
            {
                "seq": self.seq,
                "round": self.round,
                "event": self.event.value,
                "task": self.task,
                "detail": self.detail,
                "node_id": self.node_id,
                "assumption_id": self.assumption_id,
                "data": self.data,
                "previous": previous,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def compute_digest(self, previous: str) -> str:
        return hashlib.sha256(self.payload(previous)).hexdigest()

    @property
    def short_digest(self) -> str:
        """For display only. Comparisons use the full digest."""
        return self.digest[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "round": self.round,
            "event": self.event.value,
            "task": self.task,
            "detail": self.detail,
            "node_id": self.node_id,
            "assumption_id": self.assumption_id,
            "data": dict(self.data),
            "digest": self.digest,
        }


class RunLedger:
    """Append-only, and checkably so.

    Every entry's digest is SHA-256 over the canonical JSON of its own fields
    *and* the digest before it, so a rewritten, reordered or dropped entry
    breaks the chain and :meth:`verify` says so. "Append-only" as a convention
    is just a promise; as a hash chain it is a test. Note what that buys and
    what it does not: tampering is *detectable*, not prevented.

    There are no wall-clock timestamps in it. The package guarantees builds that
    are byte-identical across runs and hash seeds, and a clock would be the one
    field in the record that could not be reproduced. For the same reason
    :func:`_canonical` refuses payload values without a single JSON form.
    """

    GENESIS = "0" * 64

    def __init__(self) -> None:
        self._entries: List[LedgerEntry] = []

    @property
    def entries(self) -> Tuple[LedgerEntry, ...]:
        """A copy. The list itself is private so nothing can rewrite history."""
        return tuple(self._entries)

    def record(
        self,
        round: int,
        event: Event,
        task: str,
        detail: str,
        *,
        node_id: Optional[str] = None,
        assumption_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> LedgerEntry:
        previous = self._entries[-1].digest if self._entries else self.GENESIS
        entry = LedgerEntry(
            seq=len(self._entries) + 1,
            round=round,
            event=event,
            task=task,
            detail=detail,
            node_id=node_id,
            assumption_id=assumption_id,
            data=_canonical(dict(data or {})),
        )
        entry = LedgerEntry(
            **{**entry.__dict__, "digest": entry.compute_digest(previous)}
        )
        self._entries.append(entry)
        return entry

    def verify(self) -> bool:
        previous = self.GENESIS
        for index, entry in enumerate(self._entries, start=1):
            if entry.seq != index:
                return False
            if entry.compute_digest(previous) != entry.digest:
                return False
            previous = entry.digest
        return True

    @property
    def head(self) -> str:
        return self._entries[-1].digest if self._entries else self.GENESIS

    @property
    def short_head(self) -> str:
        return self.head[:12]

    def of(self, task: str) -> Tuple[LedgerEntry, ...]:
        return tuple(e for e in self._entries if e.task == task)

    def count(self, event: Event) -> int:
        return sum(1 for e in self._entries if e.event is event)

    def receipts(self) -> Tuple[Dict[str, Any], ...]:
        """Every blast radius that happened, read off the ledger alone.

        Each receipt answers the whole chain — which assumption failed, what
        evidence disproved it, exactly what fell and why, and what was left
        standing — without needing the graph the run was performed against.
        """
        return tuple(
            dict(e.data) for e in self._entries if e.event is Event.DISPROVED and e.data
        )

    def to_dict(self) -> List[Dict[str, Any]]:
        """The entries as rows, for a report or a dashboard.

        Not the durable format — see :meth:`snapshot`, which adds the schema,
        the version and the head that make a file checkable on the way back in.
        """
        return [e.to_dict() for e in self._entries]

    # ── the durable format ───────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        """The ledger as a versioned, self-describing container.

        The head is written down even though it is the last entry's digest and
        could be read off the array. Storing it makes the file state what it
        claims to be, so a truncated array is a contradiction the loader can
        name rather than a shorter history it would happily accept.
        """
        return {
            "schema": LEDGER_SCHEMA,
            "version": LEDGER_VERSION,
            "head": self.head,
            "entries": [e.to_dict() for e in self._entries],
        }

    def to_json(self) -> str:
        """The snapshot as text, in the exact form :meth:`save_json` writes."""
        return (
            json.dumps(
                self.snapshot(),
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                # NaN and Infinity are Python's, not JSON's. A ledger that saves
                # is one another tool can read and re-check.
                allow_nan=False,
            )
            + "\n"
        )

    def save_json(self, path: Any) -> Any:
        from pathlib import Path

        target = Path(path)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, path: Any, *, expect_head: Optional[str] = None) -> "RunLedger":
        from pathlib import Path

        text = Path(path).read_text(encoding="utf-8")
        data = _parse_durable_json(text, LedgerIntegrityError)
        return cls.from_snapshot(data, expect_head=expect_head)

    @classmethod
    def from_snapshot(
        cls, data: Any, *, expect_head: Optional[str] = None
    ) -> "RunLedger":
        """Rebuild a ledger from a snapshot, or refuse it.

        Every entry is taken **exactly as written** and then checked. It is not
        replayed through :meth:`record`, and that is the whole point: replaying
        would recompute each digest from whatever the file happened to say, so
        an edited entry would come back with a freshly consistent digest and a
        chain that verifies. Tampering would be laundered by the loader that was
        supposed to catch it. Here the stored digest is evidence, not a field to
        be regenerated, and the load fails if it does not hold up.

        What this proves, and what it does not::

            edit an entry     → its digest no longer recomputes      REFUSED
            delete an entry   → the next one's previous is wrong     REFUSED
            reorder entries   → seq and previous both disagree       REFUSED
            rebuild the lot   → internally perfect                   ACCEPTED

        That last row is not a hole; it is what a hash chain is. Anyone who can
        rewrite every entry can recompute every digest, and no amount of
        self-checking distinguishes that chain from the original. What does
        distinguish them is a head you trusted *before* the file could be
        rewritten. Pass it as ``expect_head`` and the forgery is refused,
        because the honest history and its digest already exist, so producing a
        *different* history ending in that same digest is a SHA-256 second
        preimage. Omit it and you get tamper *evidence* — modification, deletion
        and reordering — but not continuity.
        """
        if not isinstance(data, dict):
            raise LedgerIntegrityError(
                f"ledger snapshot must be an object, got {type(data).__name__}"
            )
        missing = sorted(_LEDGER_KEYS - set(data))
        if missing:
            raise LedgerIntegrityError(
                "ledger snapshot is missing "
                + ", ".join(repr(key) for key in missing)
            )
        unknown = sorted(set(data) - _LEDGER_KEYS)
        if unknown:
            raise LedgerIntegrityError(
                "ledger snapshot carries "
                + ", ".join(repr(key) for key in unknown)
                + ", which v1 does not define. A future version may give those "
                "names meaning, and a v1 reader dropping them would disagree "
                "with it about what the same file said"
            )
        if data["schema"] != LEDGER_SCHEMA:
            raise LedgerIntegrityError(
                f"not a {LEDGER_SCHEMA} snapshot: schema is {data['schema']!r}"
            )
        version = data["version"]
        # ``True == 1`` in Python, so a bool would sail through the equality
        # check below and load as version 1.
        if isinstance(version, bool) or not isinstance(version, int):
            raise LedgerIntegrityError(
                f"ledger version must be an integer, got {version!r}"
            )
        if version != LEDGER_VERSION:
            raise LedgerIntegrityError(
                f"ledger version {version!r} cannot be read by this build, "
                f"which writes and reads version {LEDGER_VERSION}"
            )
        declared = data["head"]
        if not isinstance(declared, str) or not _DIGEST.match(declared):
            raise LedgerIntegrityError(
                f"ledger head must be 64 lowercase hex characters, got {declared!r}"
            )
        rows = data["entries"]
        if not isinstance(rows, list):
            raise LedgerIntegrityError(
                f"ledger entries must be an array, got {type(rows).__name__}"
            )

        ledger = cls()
        previous = cls.GENESIS
        for index, row in enumerate(rows, start=1):
            entry = _entry_from_row(row, index)
            if entry.compute_digest(previous) != entry.digest:
                raise LedgerIntegrityError(
                    f"ledger entry {index}: digest does not match its contents. "
                    "The entry was edited after it was written, or the entry "
                    "before it was changed, removed or moved"
                )
            ledger._entries.append(entry)
            previous = entry.digest

        if previous != declared:
            raise LedgerIntegrityError(
                f"ledger head says {declared} but the entries end at {previous}; "
                "the file disagrees with itself about how much history it holds"
            )
        if expect_head is not None and ledger.head != expect_head:
            raise LedgerIntegrityError(
                f"ledger does not continue the history it was checked against: "
                f"expected head {expect_head}, restored {ledger.head}. The chain "
                "may verify perfectly and still be a different chain"
            )
        return ledger

    def __len__(self) -> int:
        return len(self._entries)


def _entry_from_row(row: Any, index: int) -> LedgerEntry:
    """One serialised entry, checked field by field before it becomes an object.

    Every check here is about a value the digest is taken over. A row that got
    this far and then fails its digest is tampering; a row that fails *here* is
    something that could never have been written by this build at all, and
    saying which field is wrong beats reporting a hash mismatch and leaving the
    reader to find it.
    """
    where = f"ledger entry {index}"
    if not isinstance(row, dict):
        raise LedgerIntegrityError(f"{where} must be an object, got {type(row).__name__}")
    missing = sorted(_ENTRY_KEYS - set(row))
    if missing:
        raise LedgerIntegrityError(f"{where} is missing {', '.join(repr(k) for k in missing)}")
    unknown = sorted(set(row) - _ENTRY_KEYS)
    if unknown:
        raise LedgerIntegrityError(
            f"{where} carries {', '.join(repr(k) for k in unknown)}, which the "
            "digest does not cover; a durable entry holds only signed fields"
        )

    seq = row["seq"]
    if isinstance(seq, bool) or not isinstance(seq, int):
        raise LedgerIntegrityError(f"{where}: seq must be an integer, got {seq!r}")
    if seq != index:
        raise LedgerIntegrityError(
            f"{where}: seq is {seq}, but this is entry {index}. A ledger is "
            "numbered from 1 with no gaps, so a jump is a removal or a reorder"
        )

    round_ = row["round"]
    if isinstance(round_, bool) or not isinstance(round_, int) or round_ < 0:
        raise LedgerIntegrityError(
            f"{where}: round must be a non-negative integer, got {round_!r}"
        )

    event = row["event"]
    if not isinstance(event, str):
        raise LedgerIntegrityError(f"{where}: event must be a string, got {event!r}")
    try:
        parsed = Event(event)
    except ValueError:
        known = ", ".join(sorted(e.value for e in Event))
        raise LedgerIntegrityError(
            f"{where}: {event!r} is not an event this build knows; it reads {known}"
        ) from None

    for name in ("task", "detail"):
        if not isinstance(row[name], str):
            raise LedgerIntegrityError(
                f"{where}: {name} must be a string, got {row[name]!r}"
            )
    for name in ("node_id", "assumption_id"):
        if row[name] is not None and not isinstance(row[name], str):
            raise LedgerIntegrityError(
                f"{where}: {name} must be a string or null, got {row[name]!r}"
            )

    data = row["data"]
    if not isinstance(data, dict):
        raise LedgerIntegrityError(
            f"{where}: data must be an object, got {type(data).__name__}"
        )
    try:
        # Not cosmetic. ``from_snapshot`` also accepts a dict that never went
        # through JSON, and a payload holding a set or a NaN would produce a
        # digest that depends on how this interpreter rendered it.
        canonical = _canonical(data)
    except ExecutionError as exc:
        raise LedgerIntegrityError(f"{where}: {exc}") from None

    digest = row["digest"]
    if not isinstance(digest, str) or not _DIGEST.match(digest):
        raise LedgerIntegrityError(
            f"{where}: digest must be 64 lowercase hex characters, got {digest!r}"
        )

    return LedgerEntry(
        seq=seq,
        round=round_,
        event=parsed,
        task=row["task"],
        detail=row["detail"],
        node_id=row["node_id"],
        assumption_id=row["assumption_id"],
        data=canonical,
        # Kept exactly as the file wrote it. Recomputing here instead would make
        # every entry consistent by construction and the chain unfalsifiable.
        digest=digest,
    )


def _task_from_row(row: Any, index: int) -> Task:
    """One serialised task, checked field by field before it becomes an object.

    No callables are set here. A task arrives from a file with two names and
    leaves this function still holding only names; binding them is a separate
    step, so a plan that fails to bind never half-exists.
    """
    where = f"execution task {index}"
    if not isinstance(row, dict):
        raise ExecutionSnapshotError(
            f"{where} must be an object, got {type(row).__name__}"
        )
    missing = sorted(_TASK_KEYS - set(row))
    if missing:
        raise ExecutionSnapshotError(
            f"{where} is missing " + ", ".join(repr(key) for key in missing)
        )
    unknown = sorted(set(row) - _TASK_KEYS)
    if unknown:
        raise ExecutionSnapshotError(
            f"{where} carries "
            + ", ".join(repr(key) for key in unknown)
            + ", which v1 does not define"
        )

    for field_name in ("name", "title", "rationale", "assumes"):
        if not isinstance(row[field_name], str):
            raise ExecutionSnapshotError(
                f"{where}: {field_name} must be a string, got {row[field_name]!r}"
            )
    if not row["name"]:
        raise ExecutionSnapshotError(f"{where}: name must not be empty")
    where = f"execution task {row['name']!r}"

    state = row["state"]
    if not isinstance(state, str):
        raise ExecutionSnapshotError(f"{where}: state must be a string, got {state!r}")
    try:
        parsed_state = TaskState(state)
    except ValueError:
        known = ", ".join(sorted(s.value for s in TaskState))
        raise ExecutionSnapshotError(
            f"{where}: {state!r} is not a state this build knows; it reads {known}. "
            "Blocked is deliberately absent — it is a fact about a round, not a task"
        ) from None

    attempt = row["attempt"]
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ExecutionSnapshotError(
            f"{where}: attempt must be a non-negative integer, got {attempt!r}"
        )

    for field_name in ("needs", "produces", "artefacts"):
        value = row[field_name]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ExecutionSnapshotError(
                f"{where}: {field_name} must be an array of strings, got {value!r}"
            )

    worker_key = row["worker_key"]
    if worker_key is None:
        raise ExecutionSnapshotError(
            f"{where}: has no worker_key. A plain callable runs perfectly well in "
            "memory and simply cannot be checkpointed, so it cannot be restored"
        )
    _valid_key(worker_key, "worker")
    auditor_key = row["auditor_key"]
    if auditor_key is not None:
        _valid_key(auditor_key, "auditor")

    for field_name in ("node_id", "assumption_id"):
        value = row[field_name]
        if value is not None and (not isinstance(value, str) or not value):
            raise ExecutionSnapshotError(
                f"{where}: {field_name} must be a non-empty string or null, "
                f"got {value!r}"
            )
    if row["assumption_id"] is None:
        # ``Runner.task`` binds one through ``_assume`` before the task joins the
        # plan, so an unbound task is not a state this engine can produce. A
        # restored one would be permanently blocked on "no assumption bound".
        raise ExecutionSnapshotError(
            f"{where}: has no assumption_id. Every task in a plan is bound to "
            "its ground when it is declared, so an unbound one could only ever "
            "block"
        )

    output = row["output"]
    if not isinstance(output, dict):
        raise ExecutionSnapshotError(
            f"{where}: output must be an object, got {type(output).__name__}"
        )
    try:
        # A worker may return anything; what may be *stored* is narrower. A set
        # has no order and a NaN is not JSON, so either would restore into
        # something a later reader renders differently than this one wrote it.
        canonical = _canonical(output, path="output")
    except ExecutionError as exc:
        raise ExecutionSnapshotError(f"{where}: {exc}") from None

    return Task(
        name=row["name"],
        title=row["title"],
        run=_unbound,
        assumes=row["assumes"],
        rationale=row["rationale"],
        needs=tuple(row["needs"]),
        produces=tuple(row["produces"]),
        audit=None,
        worker_key=worker_key,
        auditor_key=auditor_key,
        state=parsed_state,
        attempt=attempt,
        node_id=row["node_id"],
        assumption_id=row["assumption_id"],
        output=canonical,
        artefacts=tuple(row["artefacts"]),
    )


def _unbound(ctx: "RunContext") -> Dict[str, Any]:
    """Placeholder between shape-checking a task and binding its worker.

    Never reachable through a restored Runner: ``from_snapshot`` binds every
    task before it hands one back, and refuses as a whole if any key is missing.
    It exists so a half-checked task is never a callable that would silently run.
    """
    raise ExecutionCheckpointError(
        "this task was restored but never bound to a worker; "
        "Runner.from_snapshot binds every task or refuses the whole plan"
    )


def _close_dependencies(tasks: List[Task]) -> None:
    """Resolve every ``needs`` against this same file, before anything else.

    Rows only — no graph. Ordered ahead of the cycle check on purpose: a task
    naming a dependency that is not in the snapshot would otherwise never become
    ready in the topological walk and be reported as a cycle, which is true but
    not the reason.
    """
    known = {task.name for task in tasks}
    for task in tasks:
        where = f"execution task {task.name!r}"
        for need in task.needs:
            if need not in known:
                raise ExecutionSnapshotError(
                    f"{where}: needs {need!r}, which is not a task in this snapshot"
                )
        if task.name in task.needs:
            raise ExecutionSnapshotError(f"{where}: needs itself")


def _refuse_dependency_cycles(tasks: List[Task]) -> None:
    """A plan that cannot be scheduled is not a plan this loader hands back.

    ``Runner._validate`` finds cycles too, but not until ``run()`` — so without
    this a restore succeeds, the Runner looks healthy, and the contradiction
    surfaces later in code that has no idea a file was involved. Checked on the
    parsed rows, before the Runner exists, so a refused snapshot does not even
    dedupe a source node into the graph on its way out.
    """
    remaining = {task.name: set(task.needs) for task in tasks}
    settled: Set[str] = set()
    while True:
        ready = [n for n in remaining if n not in settled and not remaining[n] - settled]
        if not ready:
            break
        settled.update(ready)
    if len(settled) != len(remaining):
        stuck = sorted(set(remaining) - settled)
        raise ExecutionSnapshotError(
            "execution snapshot has a dependency cycle among "
            + ", ".join(repr(name) for name in stuck)
            + "; no order runs these, so the plan could never be scheduled"
        )


def _close_plan_namespace(plan: str, tasks: List[Task], graph: ContextGraph) -> None:
    """The plan names a provenance namespace, so the graph has to agree it owns it.

    ``plan`` is persisted because ids are derived from it — the source node and,
    through ``_commit``, every decision::

        slug(f"{plan}|{task.name}|v{attempt}", "decision")

    So changing nothing but this one string restores the old decisions under a
    newly created source, and the next rerun writes into a different namespace
    while superseding the previous one. One task's lineage would then cross two
    plans, each of which looks internally consistent.

    Both halves are native invariants, not new ones: ``Runner.__init__`` derives
    the source from ``plan``, and ``DecisionLog.decide`` always links its
    decision to that source with CITES.
    """
    source_id = slug(f"execution plan: {plan}", "source")
    source = graph.nodes.get(source_id)
    if source is None:
        raise ExecutionSnapshotError(
            f"execution snapshot names plan {plan!r}, whose source {source_id!r} "
            "is not in this graph. Ids are derived from the plan, so restoring "
            "under a name the graph never recorded would start a second "
            "provenance namespace beside the one this plan already has"
        )
    if source.type is not NodeType.SOURCE:
        raise ExecutionSnapshotError(
            f"execution snapshot names plan {plan!r}, but {source_id!r} is a "
            f"{source.type.value}, not a source"
        )
    for task in tasks:
        if task.node_id is None:
            continue
        # live_only=False on purpose: a stale task's decision is invalidated,
        # and its provenance link is still the fact being checked.
        cited = [
            edge.dst
            for edge in graph.out_edges(
                task.node_id, [EdgeType.CITES], live_only=False
            )
        ]
        if source_id not in cited:
            raise ExecutionSnapshotError(
                f"execution task {task.name!r}: decision {task.node_id!r} does "
                f"not cite plan {plan!r}. It was produced under a different "
                "plan, so this snapshot would graft one lineage onto another"
            )


def _close_task_references(task: Task, *, graph: ContextGraph) -> None:
    """Every id a restored task names has to point at something, now.

    A dangling reference that loads quietly becomes a crash three rounds later,
    in code that has no idea a file was involved.
    """
    where = f"execution task {task.name!r}"
    assert task.assumption_id is not None  # _task_from_row proved it
    assumption = graph.assumptions.get(task.assumption_id)
    if assumption is None:
        raise ExecutionSnapshotError(
            f"{where}: assumption {task.assumption_id!r} is not in this graph"
        )
    if assumption.statement != task.assumes:
        # Two records of the same fact, and a file can hold them disagreeing.
        # The reporting side reads ``assumes`` while the scheduler and the
        # auditor read the bound assumption, so a plan restored like this shows
        # one ground and runs on another. Ids are content-slugged, so a genuine
        # binding cannot drift into this by accident.
        raise ExecutionSnapshotError(
            f"{where}: says its ground is {task.assumes!r}, but assumption "
            f"{task.assumption_id!r} states {assumption.statement!r}. A task "
            "cannot report one ground and be scheduled on another"
        )
    if task.node_id is not None:
        node = graph.nodes.get(task.node_id)
        if node is None:
            raise ExecutionSnapshotError(
                f"{where}: decision {task.node_id!r} is not in this graph"
            )
        if node.type is not NodeType.DECISION:
            raise ExecutionSnapshotError(
                f"{where}: {task.node_id!r} is a {node.type.value}, not a decision"
            )
        if task.state is TaskState.DONE and node.invalidated:  # noqa: SIM102
            # Measured against the live lifecycle rather than assumed: a STALE
            # task legitimately points at an invalidated decision — that is what
            # selective invalidation leaves behind. A DONE one does not, and a
            # plan restored in that state would schedule as settled while the
            # graph says its ground is gone.
            raise ExecutionSnapshotError(
                f"{where}: says done, but decision {task.node_id!r} is invalidated. "
                "A task whose ground fell is stale, not done"
            )
    if task.state is TaskState.DONE:
        # Measured against the lifecycle rather than assumed. `_commit` creates
        # the decision and only then marks the task done, so done-without-one is
        # a state this engine never reaches — and a dangerous one to accept,
        # because `_ready` skips DONE tasks. A restored plan would cache work as
        # complete that the graph has no record of ever happening.
        if task.node_id is None:
            raise ExecutionSnapshotError(
                f"{where}: says done, but names no decision. Done is a claim "
                "about provenance, and the graph has none for this task"
            )
        if task.attempt < 1:
            raise ExecutionSnapshotError(
                f"{where}: says done after {task.attempt} attempts. Work that "
                "never ran cannot be complete"
            )
        if assumption.status is AssumptionStatus.REJECTED:
            raise ExecutionSnapshotError(
                f"{where}: says done, but its ground {assumption.statement!r} "
                "is rejected. A task whose ground fell is stale, not done"
            )

    for artefact_id in task.artefacts:
        node = graph.nodes.get(artefact_id)
        if node is None:
            raise ExecutionSnapshotError(
                f"{where}: artefact {artefact_id!r} is not in this graph"
            )
        if node.type is not NodeType.ENTITY:
            raise ExecutionSnapshotError(
                f"{where}: artefact {artefact_id!r} is a {node.type.value}, "
                "not an entity"
            )


#: The durable execution format. Versioned apart from the graph snapshot and the
#: run ledger, because plan state, provenance and history change for different
#: reasons and one shared number would force all three to move together.
EXECUTION_SCHEMA = "contextmesh.execution"
EXECUTION_VERSION = 1

_EXECUTION_KEYS = frozenset({"schema", "version", "plan", "round", "tasks"})

#: What a serialised task holds — all of it, and only this. Callables are absent
#: by construction: ``run`` and ``audit`` are rebuilt from their keys through the
#: Runner's registry. So are the derived facts — ready-ness and blocked-ness are
#: recomputed every round from state, ground and dependencies, and storing them
#: would let a file assert a schedule its own contents contradict.
_TASK_KEYS = frozenset(
    {
        "name",
        "title",
        "rationale",
        "state",
        "attempt",
        "assumes",
        "needs",
        "produces",
        "worker_key",
        "auditor_key",
        "node_id",
        "assumption_id",
        "output",
        "artefacts",
    }
)


@dataclass
class RunReport:
    round: int
    layers: List[List[str]] = field(default_factory=list)
    executed: List[str] = field(default_factory=list)
    cached: List[str] = field(default_factory=list)
    blocked: Dict[str, str] = field(default_factory=dict)
    failed: Dict[str, str] = field(default_factory=dict)
    invalidations: List[InvalidationReport] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.blocked and not self.failed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round,
            "layers": [list(layer) for layer in self.layers],
            "executed": list(self.executed),
            "cached": list(self.cached),
            "blocked": dict(self.blocked),
            "failed": dict(self.failed),
            "invalidations": [r.to_dict() for r in self.invalidations],
            "complete": self.complete,
        }


class Runner:
    """Schedules work onto a ContextGraph, and reschedules exactly what fell.

    The scheduling rule is the same one the graph already uses for
    invalidation, read the other way round: a task runs when everything it
    ``depends_on`` is standing, and it reruns when something it depends on
    stopped standing.
    """

    #: A guard against a plan that keeps re-invalidating itself. Reruns are
    #: bounded by the graph, not by the runner, so hitting this means the plan
    #: has a live cycle the topological check could not see statically.
    MAX_PASSES = 6

    def __init__(
        self,
        plan: str,
        *,
        graph: Optional[ContextGraph] = None,
        assumptions: Optional[AssumptionLedger] = None,
        decisions: Optional[DecisionLog] = None,
        registry: Optional[TaskRegistry] = None,
    ) -> None:
        # ``is None``, not ``or``: an empty graph is falsy-looking but real, and
        # ``graph or ContextGraph()`` would silently substitute a fresh one.
        self.plan = plan
        self.graph = graph if graph is not None else ContextGraph()
        self.assumptions = (
            assumptions if assumptions is not None else AssumptionLedger(self.graph)
        )
        self.decisions = decisions if decisions is not None else DecisionLog(self.graph)
        # Deployment configuration, not plan state, and exactly one per Runner.
        # A durable key has to mean one implementation across this whole plan:
        # if two registries were in play, the same string could name argon2 in
        # the checkpoint and bcrypt at runtime. Read-only on purpose — the only
        # ways in are here and ``rebind()``, which re-resolves every task before
        # it adopts one. Never serialised — see ``TaskRegistry``.
        self._registry = registry
        self.ledger = RunLedger()
        self.rounds: List[RunReport] = []
        self.round = 0
        self._tasks: Dict[str, Task] = {}
        self._order: List[str] = []
        self._blocked: Dict[str, str] = {}
        self.source = self.graph.add_node(
            NodeType.SOURCE,
            f"execution plan: {plan}",
            attrs={"origin": "runner", "retrieved_at": "at plan time"},
        )

    @property
    def registry(self) -> Optional[TaskRegistry]:
        """The one registry that says what this plan's durable keys mean.

        Set at construction or swapped wholesale by :meth:`rebind`. Not
        assignable, because a bare assignment would leave the tasks bound to
        the previous registry's callables while every later ``repair()``
        resolved against the new one — a Runner running one implementation and
        checkpointing the name of another.
        """
        return self._registry

    # ── declaring work ───────────────────────────────────────────────────
    def task(
        self,
        name: str,
        run: Optional[Worker] = None,
        *,
        assumes: str,
        title: Optional[str] = None,
        rationale: str = "",
        needs: Iterable[str] = (),
        produces: Iterable[str] = (),
        audit: Optional[Auditor] = None,
        worker_key: Optional[str] = None,
        auditor_key: Optional[str] = None,
    ) -> Task:
        """Declare a task, either as a plain callable or as a durable key.

        Two ways in, and they do not mix:

            task("temp", run=fn, assumes=...)                 in-memory only
            task("hash", worker_key="auth.hash.argon2.v1",    checkpointable
                 assumes=...)

        A plain callable is still perfectly good work — it runs, it audits, it
        invalidates. It simply cannot be checkpointed, because nothing in a file
        could name it again. Passing both ``run`` and ``worker_key`` is refused
        rather than reconciled: the two can disagree, and a checkpoint would
        then record a name for code it is not actually running.

        The key is always resolved through the Runner's own registry. There is
        deliberately no per-task ``registry=``: a checkpoint records the key and
        not the table it came from, so two tables would make the same string
        mean two different implementations and a restore could pick the wrong
        one. To run a task against different code, give it a different key.
        """
        if name in self._tasks:
            raise ExecutionError(f"duplicate task {name!r}")
        table = self._registry

        if run is not None and worker_key is not None:
            raise ExecutionCheckpointError(
                f"task {name!r}: give run= or worker_key=, not both — they are "
                "two answers to 'which code is this' and can disagree"
            )
        if audit is not None and auditor_key is not None:
            raise ExecutionCheckpointError(
                f"task {name!r}: give audit= or auditor_key=, not both"
            )
        if run is None and worker_key is None:
            raise ExecutionError(f"task {name!r} needs run= or worker_key=")
        if (worker_key is not None or auditor_key is not None) and table is None:
            raise ExecutionCheckpointError(
                f"task {name!r} names a durable key but this Runner has no "
                "TaskRegistry; pass one to Runner(registry=...)"
            )
        if worker_key is not None:
            assert table is not None  # the check above proved it
            run = table.worker(worker_key)
        if auditor_key is not None:
            assert table is not None
            audit = table.auditor(auditor_key)

        assert run is not None  # one of the two branches above supplied it
        task = Task(
            name=name,
            title=title or name.replace("_", " ").strip().capitalize(),
            run=run,
            assumes=assumes,
            rationale=rationale,
            needs=tuple(needs),
            produces=tuple(produces),
            audit=audit,
            worker_key=worker_key,
            auditor_key=auditor_key,
            governed_by=self,
        )
        task.assumption_id = self._assume(assumes).id
        self._tasks[name] = task
        self._order.append(name)
        return task

    def __getitem__(self, name: str) -> Task:
        return self._tasks[name]

    @property
    def tasks(self) -> Tuple[Task, ...]:
        return tuple(self._tasks[n] for n in self._order)

    @property
    def unaudited(self) -> Tuple[str, ...]:
        """Tasks running on nothing but hope. Worth surfacing, not worth failing on."""
        return tuple(n for n in self._order if self._tasks[n].audit is None)

    def state(self, name: str) -> TaskState:
        return self._tasks[name].state

    def blocked_reason(self, name: str) -> Optional[str]:
        return self._blocked.get(name)

    def output(self, name: str) -> Dict[str, Any]:
        return dict(self._tasks[name].output)

    # ── running ──────────────────────────────────────────────────────────
    def run(self) -> RunReport:
        """Execute every task that is pending or stale and whose ground stands.

        Anything already ``DONE`` is reported as cached and not touched. That is
        the payoff for selective invalidation: the second run is small.
        """
        self._validate()
        self.round += 1
        self.graph.build += 1
        report = RunReport(round=self.round)
        self._blocked = {}
        done_at_start = {n for n in self._order if self._tasks[n].state is TaskState.DONE}

        passes = 0
        while True:
            ready = self._ready()
            if not ready:
                break
            passes += 1
            if passes > self.MAX_PASSES * max(1, len(self._order)):
                raise ExecutionError(
                    "scheduler made no progress towards a stable graph; "
                    "a task is invalidating something it depends on"
                )
            report.layers.append(list(ready))
            for name in ready:
                task = self._tasks[name]
                if task.state not in (TaskState.PENDING, TaskState.STALE):
                    # An earlier task in this same layer disproved its ground.
                    continue
                invalidation = self._execute(task)
                if invalidation is not None:
                    report.invalidations.append(invalidation)
                if task.state is TaskState.DONE:
                    report.executed.append(name)
                elif task.state is TaskState.FAILED:
                    report.failed[name] = self._last_detail(name)

        for name in self._order:
            task = self._tasks[name]
            already = name in done_at_start and name not in report.executed
            if task.state is TaskState.DONE and already:
                report.cached.append(name)
                self.ledger.record(
                    self.round, Event.CACHED, name,
                    "unaffected by this round; previous result stands",
                    node_id=task.node_id,
                )
            elif task.state in (TaskState.PENDING, TaskState.STALE):
                reason = self._blocked.get(name) or "no reason recorded"
                report.blocked[name] = reason
                self.ledger.record(
                    self.round, Event.BLOCKED, name, reason,
                    assumption_id=task.assumption_id,
                )

        self.rounds.append(report)
        return report

    def recheck(self) -> List[InvalidationReport]:
        """Re-audit standing work against current facts. Executes nothing.

        This is where the outside world gets to change its mind. Each auditor is
        handed the output its task already produced and asked whether it still
        holds; a disproof rejects the assumption through the ledger, which is
        what produces the blast radius and marks the reruns.
        """
        self.round += 1
        self.graph.build += 1
        reports: List[InvalidationReport] = []
        for name in self._order:
            task = self._tasks[name]
            if task.state is not TaskState.DONE or task.audit is None:
                continue
            assumption = self.graph.assumptions.get(task.assumption_id or "")
            if assumption is None or assumption.status is not AssumptionStatus.ACTIVE:
                continue
            try:
                verdict = self._audit(task, task.output, assumption)
            except _AuditorRaised as exc:
                # One broken auditor must not abort independent rechecks.
                # An exception proves nothing -- not that the ground is
                # false, and not that the work is: it already passed its
                # audit when it ran. So the task stays DONE and the error
                # is recorded; the next recheck asks again. Failing it here
                # would be permanent, since nothing re-runs a FAILED task.
                self.ledger.record(
                    self.round,
                    Event.AUDIT_ERROR,
                    name,
                    f"auditor error: {exc}",
                    node_id=task.node_id,
                    assumption_id=assumption.id,
                )
                continue
            self.ledger.record(
                self.round,
                Event.AUDITED,
                name,
                f"{'holds' if verdict.ok else 'fails'}: {verdict.reason}",
                node_id=task.node_id,
                assumption_id=assumption.id,
            )
            if verdict.ok:
                continue
            if verdict.disproves:
                reports.append(self._disprove(task, assumption, verdict.reason))
            else:
                task.state = TaskState.FAILED
                self.ledger.record(
                    self.round, Event.FAILED, name, verdict.reason, node_id=task.node_id
                )
        return reports

    def repair(
        self,
        name: str,
        *,
        assumes: Optional[str] = None,
        run: Optional[Worker] = None,
        audit: Optional[Auditor] = None,
        produces: Optional[Iterable[str]] = None,
        rationale: Optional[str] = None,
        worker_key: Optional[str] = None,
        auditor_key: Optional[str] = None,
    ) -> Assumption:
        """Put new ground under a task so it can run again.

        A rejected assumption is never edited back to life — it stays rejected,
        and the replacement records that it supersedes it. That is what keeps
        "why did this change" answerable by walking the assumption's lineage.

        A repair that swaps the code has to swap the *name* of the code too.
        Repairing a durable task with a bare ``run=`` is refused, because the
        result would be a task running bcrypt while its checkpoint still said
        argon2 — and the next restore would faithfully bring back the worker
        the CVE was about. That is the exact failure this layer exists to stop,
        so it is refused at the repair rather than caught at the restore.
        """
        task = self._tasks[name]
        table = self._registry

        if run is not None and worker_key is not None:
            raise ExecutionCheckpointError(
                f"repair of {name!r}: give run= or worker_key=, not both"
            )
        if audit is not None and auditor_key is not None:
            raise ExecutionCheckpointError(
                f"repair of {name!r}: give audit= or auditor_key=, not both"
            )
        if run is not None and task.worker_key is not None:
            raise ExecutionCheckpointError(
                f"task {name!r} is bound to worker {task.worker_key!r}; repairing "
                "it with a bare run= would leave the checkpoint naming the old "
                "worker while the new one runs. Pass worker_key= instead"
            )
        if audit is not None and task.auditor_key is not None:
            raise ExecutionCheckpointError(
                f"task {name!r} is bound to auditor {task.auditor_key!r}; repairing "
                "it with a bare audit= would leave the checkpoint naming the old "
                "auditor. Pass auditor_key= instead"
            )
        if (worker_key is not None or auditor_key is not None) and table is None:
            raise ExecutionCheckpointError(
                f"repair of {name!r} names a durable key but this Runner has no "
                "TaskRegistry; pass one to Runner(registry=...)"
            )
        if worker_key is not None:
            assert table is not None
            # Resolve before mutating: a key this deployment cannot bind must
            # leave the task exactly as it was.
            run = table.worker(worker_key)
        if auditor_key is not None:
            assert table is not None
            audit = table.auditor(auditor_key)
        old = self.graph.assumptions.get(task.assumption_id or "")
        new = old
        if assumes:
            new = self._assume(assumes)
            if old is not None and new.id != old.id and new.supersedes is None:
                new.version = old.version + 1
                new.supersedes = old.id
                old.superseded_by = new.id
                # Same mirror as above: the record is not the only copy.
                self.graph.sync_assumption(new)
                self.graph.add_edge(new.id, EdgeType.SUPERSEDES, old.id)
            task.assumes = assumes
            task.assumption_id = new.id
        if run is not None:
            task.run = run
        if worker_key is not None:
            task.worker_key = worker_key
        if audit is not None:
            task.audit = audit
        if auditor_key is not None:
            task.auditor_key = auditor_key
        if produces is not None:
            task.produces = tuple(produces)
        if rationale is not None:
            task.rationale = rationale
        task.state = TaskState.STALE
        self.ledger.record(
            self.round,
            Event.REPAIRED,
            name,
            f"reground on: {new.statement}" if new is not None else "repaired",
            node_id=task.node_id,
            assumption_id=new.id if new is not None else None,
        )
        if new is None:
            raise ExecutionError(f"task {name!r} has no assumption to repair")
        return new

    # ── durable identity ─────────────────────────────────────────────────
    @property
    def checkpointable(self) -> bool:
        """Whether every task in this plan could be rebuilt in a later process."""
        return not self.unbindable()

    def unbindable(self) -> Tuple[str, ...]:
        """The reasons this plan could not be checkpointed, one per task.

        All of them, not the first. A plan with six plain callables should say
        so once rather than over six edit-and-retry cycles.
        """
        reasons = []
        for name in self._order:
            reason = self._tasks[name].unbindable_reason()
            if reason is not None:
                reasons.append(reason)
        return tuple(reasons)

    def require_checkpointable(self) -> None:
        """The gate a checkpoint writer calls before it writes anything."""
        reasons = self.unbindable()
        if reasons:
            raise ExecutionCheckpointError(
                "this plan cannot be checkpointed:\n  " + "\n  ".join(reasons)
            )

    def bindings(self) -> Dict[str, Dict[str, Optional[str]]]:
        """Every task's durable identity, by name. Names only, never callables."""
        return {name: self._tasks[name].binding() for name in self._order}

    # ── the durable format ───────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        """This plan's semantic state, as a versioned container.

        ``tasks`` is in declaration order and is **not** sorted. Order is part
        of the scheduler's semantics, not presentation: :meth:`_ready` walks
        ``_order``, so for two tasks that become ready in the same round it
        decides which runs first. Sorting the array would reorder a restored
        plan's execution without changing a single field.

        ``plan`` is here because it is load-bearing rather than a label — the
        source node's id and every decision id are derived from it, so a restore
        under a different plan string would append a parallel provenance trail
        beside the one it claims to continue.

        The run ledger is deliberately not inside this: it has its own versioned
        format and its own head, and nesting it would give one history two
        containers that could disagree.
        """
        self.require_checkpointable()
        return {
            "schema": EXECUTION_SCHEMA,
            "version": EXECUTION_VERSION,
            "plan": self.plan,
            "round": self.round,
            "tasks": [self._tasks[name].snapshot() for name in self._order],
        }

    def to_json(self) -> str:
        """The snapshot as text, in the exact form :meth:`save_json` writes."""
        return (
            json.dumps(
                self.snapshot(),
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )

    def save_json(self, path: Any) -> Any:
        from pathlib import Path

        target = Path(path)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def load_json(
        cls,
        path: Any,
        *,
        graph: ContextGraph,
        registry: TaskRegistry,
        ledger: Optional[RunLedger] = None,
        assumptions: Optional[AssumptionLedger] = None,
        decisions: Optional[DecisionLog] = None,
    ) -> "Runner":
        from pathlib import Path

        text = Path(path).read_text(encoding="utf-8")
        data = _parse_durable_json(text, ExecutionSnapshotError)
        return cls.from_snapshot(
            data,
            graph=graph,
            registry=registry,
            ledger=ledger,
            assumptions=assumptions,
            decisions=decisions,
        )

    @classmethod
    def from_snapshot(
        cls,
        data: Any,
        *,
        graph: ContextGraph,
        registry: TaskRegistry,
        ledger: Optional[RunLedger] = None,
        assumptions: Optional[AssumptionLedger] = None,
        decisions: Optional[DecisionLog] = None,
    ) -> "Runner":
        """Rebuild a plan from a snapshot against a live graph, or refuse it.

        Three things are reconstructed rather than read, and each for a reason::

            run / audit    from worker_key / auditor_key, through `registry`
            ready/blocked  recomputed, because they are facts about this round
            source node    re-derived from `plan`, and dedupes onto the old one

        A snapshot is untrusted input, so its references are closed here rather
        than left to fail somewhere downstream: the assumption must exist, the
        decision must be a decision, the artefacts must be entities, and every
        ``needs`` must name a task in this same file. A task that says it is
        DONE while its decision has been invalidated is refused outright — that
        is a plan which would restore as runnable and then contradict the graph
        it was restored against.
        """
        if not isinstance(data, dict):
            raise ExecutionSnapshotError(
                f"execution snapshot must be an object, got {type(data).__name__}"
            )
        missing = sorted(_EXECUTION_KEYS - set(data))
        if missing:
            raise ExecutionSnapshotError(
                "execution snapshot is missing "
                + ", ".join(repr(key) for key in missing)
            )
        unknown = sorted(set(data) - _EXECUTION_KEYS)
        if unknown:
            raise ExecutionSnapshotError(
                "execution snapshot carries "
                + ", ".join(repr(key) for key in unknown)
                + ", which v1 does not define"
            )
        if data["schema"] != EXECUTION_SCHEMA:
            raise ExecutionSnapshotError(
                f"not a {EXECUTION_SCHEMA} snapshot: schema is {data['schema']!r}"
            )
        version = data["version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise ExecutionSnapshotError(
                f"execution version must be an integer, got {version!r}"
            )
        if version != EXECUTION_VERSION:
            raise ExecutionSnapshotError(
                f"execution version {version!r} cannot be read by this build, "
                f"which writes and reads version {EXECUTION_VERSION}"
            )
        plan = data["plan"]
        if not isinstance(plan, str) or not plan:
            raise ExecutionSnapshotError(
                f"execution plan must be a non-empty string, got {plan!r}"
            )
        round_ = data["round"]
        if isinstance(round_, bool) or not isinstance(round_, int) or round_ < 0:
            raise ExecutionSnapshotError(
                f"execution round must be a non-negative integer, got {round_!r}"
            )
        rows = data["tasks"]
        if not isinstance(rows, list):
            raise ExecutionSnapshotError(
                f"execution tasks must be an array, got {type(rows).__name__}"
            )

        # Shape first, in declaration order, before anything is looked up.
        tasks: List[Task] = []
        seen: Set[str] = set()
        for index, row in enumerate(rows, start=1):
            task = _task_from_row(row, index)
            if task.name in seen:
                raise ExecutionSnapshotError(
                    f"execution task {task.name!r} appears twice; a task name is "
                    "how dependencies and the ledger refer to it, so two of them "
                    "have no single meaning"
                )
            seen.add(task.name)
            tasks.append(task)

        # Rows-only checks first, each ordered so a failure reports its own
        # cause: a dependency on nothing is not a cycle, and a cycle is not a
        # dangling reference.
        _close_dependencies(tasks)
        _refuse_dependency_cycles(tasks)

        # Then the graph: each task's own ids, then the plan that has to own
        # the provenance all of them sit in.
        for task in tasks:
            _close_task_references(task, graph=graph)
        _close_plan_namespace(plan, tasks, graph)

        # Then code, which is the one thing the file never carried.
        for task in tasks:
            assert task.worker_key is not None  # every row was checked above
            task.run = registry.worker(task.worker_key)
            if task.auditor_key is not None:
                task.audit = registry.auditor(task.auditor_key)

        runner = cls(
            plan,
            graph=graph,
            assumptions=assumptions,
            decisions=decisions,
            registry=registry,
        )
        if ledger is not None:
            runner.ledger = ledger
        runner.round = round_
        for task in tasks:
            task.governed_by = runner
            runner._tasks[task.name] = task
            runner._order.append(task.name)
        return runner

    def rebind(self, registry: Optional[TaskRegistry] = None) -> "Runner":
        """Reconnect every task's callables from its keys, or refuse.

        All-or-nothing: each task is resolved against the registry before any
        of them is mutated, so a plan that names one key this deployment lacks
        does not come back with five tasks bound and one dangling.

        The registry is adopted in that same step. A rebind that swapped the
        callables but left the old registry in place would give the Runner two
        answers to "what does this key mean" — the tasks running the new code
        while the next ``repair()`` resolved against the old table — so the
        adoption and the callables land together or neither does.
        """
        table = registry if registry is not None else self._registry
        if table is None:
            raise ExecutionCheckpointError(
                "rebind needs a TaskRegistry, given here or to the Runner"
            )
        self.require_checkpointable()
        resolved = {}
        for name in self._order:
            task = self._tasks[name]
            assert task.worker_key is not None  # require_checkpointable proved it
            resolved[name] = (
                table.worker(task.worker_key),
                table.auditor(task.auditor_key) if task.auditor_key else None,
            )
        # Past here nothing can raise, so the swap is a single visible step.
        for name, (run, audit) in resolved.items():
            self._tasks[name].run = run
            self._tasks[name].audit = audit
        self._registry = table
        return self

    # ── internals ────────────────────────────────────────────────────────
    def _assume(self, statement: str) -> Assumption:
        """Reuse the assumption this statement already names, or mint one.

        ``slug`` truncates its digest (GRAPH.md rule 8), so the id this
        statement derives to is not proof the statement is the one already
        recorded under it. Reusing an id whose statement disagrees would
        silently bind the caller to different ground than the one they
        asked for -- exactly the write ``add_node``'s collision guard
        exists to refuse, so this checks the same thing before this
        shortcut can bypass it.
        """
        existing = self.graph.assumptions.get(slug(statement, "assumption"))
        if existing is not None:
            if existing.statement != statement:
                raise OntologyError(
                    f"assumption id {existing.id!r} is already "
                    f"{existing.statement!r}; refusing to treat {statement!r} "
                    "as the same assumption"
                )
            return existing
        return self.assumptions.assume(statement, created_by="runner")

    def _validate(self) -> None:
        for name in self._order:
            for need in self._tasks[name].needs:
                if need not in self._tasks:
                    raise ExecutionError(f"task {name!r} needs unknown task {need!r}")
        remaining = {n: set(self._tasks[n].needs) for n in self._order}
        settled: Set[str] = set()
        while True:
            ready = [n for n in self._order if n not in settled and not remaining[n] - settled]
            if not ready:
                break
            settled.update(ready)
        if len(settled) != len(self._order):
            stuck = sorted(set(self._order) - settled)
            raise ExecutionError(f"dependency cycle among tasks: {', '.join(stuck)}")

    def _ready(self) -> List[str]:
        """Tasks that can run right now, in declaration order.

        Also the place blocked-ness is decided, because "why is this not
        running" and "what runs next" are the same question asked twice.
        """
        ready: List[str] = []
        for name in self._order:
            task = self._tasks[name]
            if task.state not in (TaskState.PENDING, TaskState.STALE):
                continue
            assumption = self.graph.assumptions.get(task.assumption_id or "")
            if assumption is None:
                self._blocked[name] = "no assumption bound"
                continue
            if assumption.status is AssumptionStatus.REJECTED:
                self._blocked[name] = (
                    f"ground rejected: {assumption.statement} — repair before rerunning"
                )
                continue
            if assumption.status is AssumptionStatus.SUPERSEDED:
                self._blocked[name] = (
                    f"ground superseded by {assumption.superseded_by}; rebind the task"
                )
                continue
            unmet = [
                need
                for need in task.needs
                if self._tasks[need].state is not TaskState.DONE
            ]
            if unmet:
                self._blocked[name] = "waiting on " + ", ".join(sorted(unmet))
                continue
            self._blocked.pop(name, None)
            ready.append(name)
        return ready

    def _audit(self, task: Task, output: Dict[str, Any], assumption: Assumption) -> Verdict:
        if task.audit is None:
            return Verdict(True, "no auditor declared")
        context = AuditContext(
            task=task, output=output, assumption=assumption, graph=self.graph
        )
        try:
            result = task.audit(context)
        except Exception as exc:
            # Callback failures are operational task failures. Keep
            # _coerce outside this boundary so a bad return type is
            # still an auditor contract error, not a retryable outage.
            raise _AuditorRaised(f"{type(exc).__name__}: {exc}") from exc
        return _coerce(result)

    def _execute(self, task: Task) -> Optional[InvalidationReport]:
        assumption = self.graph.assumptions[task.assumption_id]
        inputs = {need: dict(self._tasks[need].output) for need in task.needs}
        context = RunContext(
            task=task,
            attempt=task.attempt + 1,
            graph=self.graph,
            assumption=assumption,
            inputs=inputs,
        )
        try:
            output = task.run(context)
        except Exception as exc:  # a worker that raises is a failed task, not a crash
            task.state = TaskState.FAILED
            self.ledger.record(
                self.round,
                Event.FAILED,
                task.name,
                f"{type(exc).__name__}: {exc}",
                assumption_id=assumption.id,
            )
            return None

        task.attempt += 1
        task.output = dict(output or {})

        try:
            verdict = self._audit(task, task.output, assumption)
        except _AuditorRaised as exc:
            # The worker completed, so its attempt/output remain. The
            # task cannot become DONE without an audit verdict, but the
            # callback exception must not tear down the scheduler.
            task.state = TaskState.FAILED
            self.ledger.record(
                self.round,
                Event.FAILED,
                task.name,
                f"auditor error: {exc}",
                assumption_id=assumption.id,
            )
            return None
        self.ledger.record(
            self.round,
            Event.AUDITED,
            task.name,
            f"{'holds' if verdict.ok else 'fails'}: {verdict.reason}",
            assumption_id=assumption.id,
        )
        if not verdict.ok:
            if verdict.disproves:
                return self._disprove(task, assumption, verdict.reason)
            task.state = TaskState.FAILED
            self.ledger.record(
                self.round, Event.FAILED, task.name, verdict.reason, assumption_id=assumption.id
            )
            return None

        self._commit(task, assumption, verdict)
        return None

    def _commit(self, task: Task, assumption: Assumption, verdict: Verdict) -> None:
        """Append the decision, its artefacts, and the evidence that cleared it."""
        labels = list(task.produces)
        artefacts = [
            self.graph.add_node(
                NodeType.ENTITY, label, attrs={"canonical": label, "aliases": []}
            ).id
            for label in labels
        ]

        previous = task.node_id
        rationale = task.rationale or f"Executed under: {assumption.statement}"
        if task.attempt > 1:
            rationale = (
                f"{rationale} (rerun {task.attempt}: the previous ground did not survive audit)"
            )
        node = self.decisions.decide(
            task.title,
            rationale,
            id=slug(f"{self.plan}|{task.name}|v{task.attempt}", "decision"),
            source_id=self.source.id,
            assumptions=[assumption.id],
            produces=artefacts,
            supersedes=previous,
        )
        for need in task.needs:
            dependency = self._tasks[need]
            if dependency.node_id:
                self.graph.add_edge(node.id, EdgeType.DEPENDS_ON, dependency.node_id)

        # Rule 2: the artefact exists again because it was rebuilt. An artefact
        # this rerun no longer produces is simply not in ``artefacts`` and stays
        # invalidated, which is the correct answer for a thing that is now gone.
        rebuilt = []
        for artefact_id in artefacts:
            artefact = self.graph.node(artefact_id)
            if artefact.invalidated:
                artefact.invalidated = False
                rebuilt.append(artefact.label)

        evidence = self.graph.add_node(
            NodeType.EVIDENCE,
            f"audit of {task.name}: {verdict.reason}",
            attrs={"kind": "audit"},
        )
        self.graph.add_edge(node.id, EdgeType.JUSTIFIED_BY, evidence.id)

        task.node_id = node.id
        task.artefacts = tuple(artefacts)
        task.state = TaskState.DONE
        detail = f"attempt {task.attempt}"
        if previous:
            detail += f", supersedes {previous}"
        if rebuilt:
            detail += f", rebuilt {', '.join(rebuilt)}"
        self.ledger.record(
            self.round,
            Event.EXECUTED,
            task.name,
            detail,
            node_id=node.id,
            assumption_id=assumption.id,
        )

    def _disprove(
        self, task: Task, assumption: Assumption, reason: str
    ) -> InvalidationReport:
        """An auditor found the ground false. Reject it and mark what fell."""
        evidence = self.graph.add_node(
            NodeType.EVIDENCE,
            f"disproof from {task.name}: {reason}",
            attrs={"kind": "disproof"},
        )
        report = self.assumptions.reject(assumption.id, evidence_id=evidence.id)

        # The whole event as one entry: which ground failed, what disproved it,
        # exactly what fell and by which chain, and what was deliberately left
        # standing. Reconstructing that by aggregating the per-node entries
        # below would work, but it would mean the ledger can only answer
        # "what fell" to a reader who already knows how to reassemble it.
        # Saying three nodes fell proves half of selective invalidation; the
        # preserved set is the other half, so it goes in the same receipt.
        self.ledger.record(
            self.round,
            Event.DISPROVED,
            task.name,
            reason,
            node_id=evidence.id,
            assumption_id=assumption.id,
            data={
                "assumption_id": assumption.id,
                "assumption": assumption.statement,
                "evidence_id": evidence.id,
                "reason": reason,
                "disproved_by": task.name,
                "invalidated": {
                    node_id: list(chain) for node_id, chain in report.invalidated.items()
                },
                "preserved": list(report.preserved),
            },
        )

        for name in self._order:
            other = self._tasks[name]
            grounded_on_it = other.assumption_id == assumption.id
            in_radius = bool(other.node_id) and other.node_id in report.invalidated
            if not grounded_on_it and not in_radius:
                continue
            if other.state is TaskState.DONE or other is task:
                other.state = TaskState.STALE
                self.ledger.record(
                    self.round,
                    Event.INVALIDATED,
                    name,
                    report.why(other.node_id) if in_radius else f"stood on: {assumption.statement}",
                    node_id=other.node_id,
                    assumption_id=assumption.id,
                )
        return report

    def _last_detail(self, name: str) -> str:
        entries = self.ledger.of(name)
        return entries[-1].detail if entries else ""

    # ── reporting ────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan,
            "round": self.round,
            "tasks": [self._tasks[n].to_dict() for n in self._order],
            "unaudited": list(self.unaudited),
            "rounds": [r.to_dict() for r in self.rounds],
            "ledger": self.ledger.to_dict(),
            "ledger_head": self.ledger.head,
            "ledger_intact": self.ledger.verify(),
        }


# ── the worked example ───────────────────────────────────────────────────
class Advisories:
    """A stand-in for the outside world: a security feed the auditors read.

    The runner never writes here, and neither does the demo script's
    invalidation step. That is the point. The CVE is not a flag flipped on a
    node by the code that wants the demo to be dramatic — it is a fact that
    appears outside the graph, and the hashing auditor *discovers* it on the
    next recheck. If the auditor were removed, nothing would fall over, which
    is exactly the property a fake demo cannot have.
    """

    def __init__(self, open_advisories: Iterable[Tuple[str, str]] = ()) -> None:
        self._open: Dict[str, str] = dict(open_advisories)

    def publish(self, package: str, advisory: str) -> None:
        self._open[package] = advisory

    def withdraw(self, package: str) -> None:
        self._open.pop(package, None)

    def clear(self, package: str) -> bool:
        return package not in self._open

    def advisory(self, package: str) -> Optional[str]:
        return self._open.get(package)


@dataclass
class DemoRun:
    runner: Runner
    feed: Advisories
    invalidations: List[InvalidationReport] = field(default_factory=list)


def _hasher(package: str, params: str, artefact: str) -> Worker:
    def run(ctx: RunContext) -> Dict[str, Any]:
        return {
            "package": package,
            "params": params,
            "artefact": artefact,
            "verified": True,
        }

    return run


def _audit_hashing(feed: Advisories) -> Auditor:
    """The only auditor that reads the world instead of just the output."""

    def audit(ctx: AuditContext) -> Verdict:
        package = ctx.output["package"]
        advisory = feed.advisory(package)
        if advisory is not None:
            return ctx.disproved(f"{package} has an open advisory — {advisory}")
        return ctx.ok(f"{package} carries no open advisory")

    return audit


def demo(feed: Optional[Advisories] = None) -> DemoRun:
    """Build a service, break its ground from outside, rebuild only what fell.

    Five steps, two of which are independent of the hashing choice. A CVE lands
    against the hashing package *after* everything is green; the next recheck
    finds it; exactly the three steps that stood on it rerun.
    """
    feed = feed if feed is not None else Advisories()
    runner = Runner("secure authentication service")

    runner.task(
        "schema",
        lambda ctx: {"tables": ["users", "sessions"], "migrations": 2},
        title="Define the auth schema",
        assumes="the user store is reachable and empty",
        produces=("Auth Schema",),
        audit=lambda ctx: (
            ctx.ok(f"{len(ctx.output['tables'])} tables created")
            if ctx.output["tables"]
            else ctx.fail("no tables were created")
        ),
    )
    runner.task(
        "hashing",
        _hasher("argon2-cffi", "t=3,m=64MiB,p=4", "Argon2 Parameters"),
        title="Implement password hashing",
        assumes="argon2-cffi has no open advisory",
        produces=("Password Hasher", "Argon2 Parameters"),
        audit=_audit_hashing(feed),
    )
    runner.task(
        "tokens",
        lambda ctx: {"algorithm": "EdDSA", "key_bits": 256},
        title="Implement the token generator",
        assumes="the signing key is provisioned from the environment",
        produces=("Token Generator",),
        audit=lambda ctx: (
            ctx.ok(f"{ctx.output['key_bits']}-bit {ctx.output['algorithm']} key")
            if ctx.output["key_bits"] >= 256
            else ctx.fail("signing key is under 256 bits")
        ),
    )
    runner.task(
        "routes",
        lambda ctx: {
            "endpoints": ["POST /login", "POST /register"],
            "hasher": ctx.inputs["hashing"]["package"],
        },
        title="Implement the auth routes",
        assumes="the hasher and token generator expose stable interfaces",
        needs=("hashing", "tokens"),
        produces=("Login Endpoint", "Register Endpoint"),
        audit=lambda ctx: (
            ctx.ok(f"{len(ctx.output['endpoints'])} endpoints over {ctx.output['hasher']}")
            if len(ctx.output["endpoints"]) == 2
            else ctx.fail("expected a login and a register endpoint")
        ),
    )
    runner.task(
        "rate_limit",
        lambda ctx: {"window_seconds": 60, "limit": 10},
        title="Implement rate limiting",
        assumes="the rate limit store is thread safe",
        needs=("routes",),
        produces=("Rate Limiter",),
        audit=lambda ctx: (
            ctx.ok(f"{ctx.output['limit']} per {ctx.output['window_seconds']}s")
            if ctx.output["limit"] > 0
            else ctx.fail("rate limit is not enforcing anything")
        ),
    )

    runner.run()

    # The world moves. Nothing in the graph is touched by this line.
    feed.publish("argon2-cffi", "CVE-2026-9999: memory-hard parameters silently downgraded")

    invalidations = runner.recheck()

    runner.repair(
        "hashing",
        assumes="bcrypt has no open advisory",
        run=_hasher("bcrypt", "cost=12", "Bcrypt Parameters"),
        produces=("Password Hasher", "Bcrypt Parameters"),
    )
    runner.run()

    return DemoRun(runner=runner, feed=feed, invalidations=invalidations)
