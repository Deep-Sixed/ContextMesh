"""The graph an MCP server serves, and how it survives a restart.

One session per process is right for a stdio server talking to one local
client. It is not right for a shared service, and this file is where that
assumption is written down rather than implied.

A session on disk is a *directory*, not a file::

    session/
      session-000003.json    the manifest, and the commit point
      session.json           latest-manifest pointer for humans and older tools
      graph-000003.json      contextmesh.graph    v1 (contextmesh/graph.py)
      resolver-000003.json   contextmesh.resolver v1 (contextmesh/resolve.py)
      execution-000003.json  contextmesh.execution v1 (contextmesh/execute.py)
      runledger-000003.json  contextmesh.runledger v1 (contextmesh/execute.py)

The generation manifest is the join, and it is the only place that can check the
invariants the companions cannot see on their own. Session v1 held graph and
resolver; v2 adds the execution plan and its run ledger.

**Generations, because a session gets overwritten.** A save never overwrites a
live companion. It writes a whole new generation under new names, and only then
publishes ``session-000NNN.json``. The generation manifest is the commit:

    crash before publication  ->  the previous generation is still named, intact
    crash during publication  ->  one manifest or the other, never half of one
    crash after publication   ->  the new generation is named, and complete

Superseded generations are deleted after publication, so an interrupted save costs
disk rather than correctness.
"""

from __future__ import annotations

import json
import os
import re
import socket
import stat
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from contextmesh.assumptions import AssumptionLedger
from contextmesh.demo import run as demo_run
from contextmesh.execute import (
    ExecutionError,
    RunLedger,
    Runner,
    TaskRegistry,
)
from contextmesh.graph import ContextGraph
from contextmesh.ingest import IngestError, build_folder
from contextmesh.model import NodeType
from contextmesh.resolve import Resolver
from contextmesh.traverse import DEFAULT_POLICY, EdgeType, Walker

DEFAULT_ROUNDS = 8

#: The session directory's own format. Bump for any change that makes a
#: directory this build writes unreadable to an older one, or vice versa.
SESSION_SCHEMA = "contextmesh.session"
SESSION_VERSION = 2

#: Versions this build can *read*. It writes ``SESSION_VERSION`` and only that.
#: v1 held a graph and a resolver; v2 adds the execution plan and its run
#: ledger. A v1 directory is a real thing this project shipped, so it is read
#: rather than orphaned — as a session with no execution — and the next save
#: upgrades it. A version from the *future* is still refused: this build cannot
#: know what it would be dropping.
READABLE_VERSIONS = (1, 2)

SESSION_FILE = "session.json"
SESSION_MANIFEST_STEM = "session"
LOCK_FILE = "session.lock"
LOCK_META_FILE = "session.lock.meta"
#: Prefix for the short-lived file every write lands in first. Recognisable so
#: an abandoned one can be swept, random so it cannot be pre-empted by a name
#: planted in the directory.
STAGING_PREFIX = ".staging-"
STAGING_SUFFIX = ".tmp"
GRAPH_STEM = "graph"
RESOLVER_STEM = "resolver"
EXECUTION_STEM = "execution"
RUNLEDGER_STEM = "runledger"

#: The companions a manifest can name, and the stem each generation uses. Order
#: matters only for reading errors; the manifest is the commit point either way.
COMPANIONS = (
    ("graph", GRAPH_STEM),
    ("resolver", RESOLVER_STEM),
    ("execution", EXECUTION_STEM),
    ("ledger", RUNLEDGER_STEM),
)

#: The two a session cannot be without.
REQUIRED_COMPANIONS = ("graph", "resolver")

#: A versioned durable format has an exact vocabulary. v1 is kept separately so
#: the compatibility promise does not quietly teach an old format new fields.
_SESSION_V1_KEYS = frozenset(
    {"schema", "version", "generation", "rounds", "walker", "graph", "resolver"}
)
_SESSION_V2_KEYS = frozenset(
    _SESSION_V1_KEYS | {"execution", "ledger", "ledger_head"}
)
_LEDGER_HEAD_RE = re.compile(r"^[0-9a-f]{64}$")
_GENERATION_MANIFEST_RE = re.compile(r"^session-(?P<generation>[0-9]{6})[.]json$")

#: How often a served session is written back. ``every-ask`` is the default
#: because a question is a write here and a lost write is silent; the cost is
#: one full serialisation per question, which is real and bounded.
CHECKPOINT_POLICIES = ("every-ask", "on-exit", "never")
DEFAULT_CHECKPOINT = "every-ask"


def generation_name(stem: str, generation: int) -> str:
    """The one filename this build writes for a stem and a generation.

    Zero-padded so a directory sorts readably, and checked on load: a manifest
    whose names disagree with its generation is refused, because the next save
    would compute a name that is already live and overwrite it in place — which
    is the exact failure generations exist to prevent.
    """
    return f"{stem}-{generation:06d}.json"


def manifest_name(generation: int) -> str:
    return generation_name(SESSION_MANIFEST_STEM, generation)


class SessionLockedError(Exception):
    """Raised when another process is already writing this session directory.

    Not a ``SessionError``: the directory is fine, and so is the session. The
    only problem is timing, and a caller that wants to wait and retry needs to
    tell that apart from "this directory cannot be restored". ``Checkpointer``
    treats it as a skipped commit; the mutation stays pending for the next one.
    """


#: How many times ``load`` re-reads a directory that moved under it. Small on
#: purpose: each retry only happens when a writer demonstrably committed during
#: the read, and a directory being rewritten faster than this is a problem the
#: reader cannot fix by waiting.
LOAD_ATTEMPTS = 4
LOAD_BACKOFF = 0.05


class _Swept(Exception):
    """Internal: a file the manifest named was gone by the time we read it.

    Not an error on its own. Committed companions are immutable — once a
    manifest names them they are never rewritten, only deleted by a later
    sweep — so the *only* way a read of them fails is that a writer moved the
    directory on mid-read. ``load`` re-reads; the caller never sees this.
    """

    def __init__(self, name: str, field_name: str) -> None:
        super().__init__(name)
        self.name = name
        self.field_name = field_name


class SessionError(ValueError):
    """Raised when a directory is not a session this build can restore.

    Loading fails closed. A session that half-loads is worse than one that
    refuses: the reads still answer, they just answer differently, and nothing
    in the output says so.
    """


def _same_session_directory(left: Path, right: Path) -> bool:
    """Return whether two spellings name the same existing session directory.

    ``Path.__eq__`` is lexical, so ``session`` and ``session/../session`` are
    different values even though the filesystem sends both to the same place.
    The stale-writer CAS is a statement about that place, not its spelling.
    Prefer the filesystem's own identity comparison and keep a realpath/normcase
    fallback for races where one spelling disappears between checks.
    """
    try:
        return os.path.samefile(left, right)
    except OSError:
        left_real = os.path.normcase(os.path.realpath(os.fspath(left)))
        right_real = os.path.normcase(os.path.realpath(os.fspath(right)))
        return left_real == right_real


def _directory_identity(path: Path) -> Tuple[int, int]:
    """Stable in-process identity of one existing directory object.

    A resolved pathname is not an identity: the directory can be renamed and a
    different directory can be created at the same spelling. ``stat`` gives us
    the filesystem object's device/inode identity on the supported Windows and
    POSIX filesystems, which remains stable across a rename.
    """
    try:
        info = path.stat()
    except OSError as exc:
        raise SessionError(
            f"the loaded session directory {path} is no longer available: {exc}"
        ) from None
    if not stat.S_ISDIR(info.st_mode):
        raise SessionError(f"the loaded session target {path} is no longer a directory")
    return int(info.st_dev), int(info.st_ino)


def _no_session_constants(value: str) -> float:
    raise SessionError(f"session.json contains the non-JSON constant {value!r}")


def _strict_session_object(pairs: Any) -> Dict[str, Any]:
    """Build one JSON object, refusing a name that appears twice at any depth."""
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SessionError(f"session.json contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _exact_session_keys(data: Dict[str, Any], version: int) -> None:
    expected = _SESSION_V1_KEYS if version == 1 else _SESSION_V2_KEYS
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise SessionError(
            f"{SESSION_FILE} is missing " + ", ".join(repr(name) for name in missing)
        )
    if unknown:
        raise SessionError(
            f"session version {version} does not define "
            + ", ".join(repr(name) for name in unknown)
        )


def _declared_ledger_head(data: Dict[str, Any], version: int) -> Optional[str]:
    """Validate the v2 execution/ledger/head unit and return its trusted head."""
    if version == 1:
        return None

    execution = data["execution"]
    ledger = data["ledger"]
    head = data["ledger_head"]
    has_execution = execution is not None
    has_ledger = ledger is not None
    has_head = head is not None

    if has_execution != has_ledger:
        present = "execution" if has_execution else "ledger"
        missing = "ledger" if has_execution else "execution"
        raise SessionError(
            f"{SESSION_FILE} names the {present} but no {missing}; a plan "
            "and the record of it running are committed together or not at all"
        )

    if has_execution and not has_head:
        raise SessionError(
            "session.ledger_head must not be null when execution and ledger are present"
        )
    if not has_execution and has_head:
        raise SessionError(
            "session execution, ledger and ledger_head are one unit; the three "
            "fields travel together"
        )
    if has_head and (
        not isinstance(head, str) or _LEDGER_HEAD_RE.fullmatch(head) is None
    ):
        raise SessionError(
            "session.ledger_head must be exactly 64 lowercase hexadecimal "
            f"characters, got {head!r}"
        )
    return head


def write_in_place(directory: Path, name: str, payload: str) -> Path:
    """Put ``payload`` at ``directory/name``, through a descriptor we own.

    Writing by name follows whatever is already there. A session directory is
    untrusted input on the way *out* as well as in: a directory handed over
    with ``graph-000006.json`` already present as a symlink would have the next
    save write through it, and with ``--checkpoint every-ask`` merely *asking a
    question* is enough to trigger that. Reproduced before it was fixed; four
    names were reachable that way, including the lock file.

    So every write lands in a fresh file created with ``O_EXCL`` under a random
    name — which cannot already exist, and so cannot already be a link — and is
    moved into place with ``os.replace``. Rename replaces a symlink *itself*
    rather than following it, so a planted link is destroyed rather than
    honoured, and the file it pointed at is never touched.

    The rename is also what makes the write atomic, which the manifest needs
    anyway. One mechanism, both properties.
    """
    handle, staged = tempfile.mkstemp(
        prefix=STAGING_PREFIX, suffix=STAGING_SUFFIX, dir=str(directory)
    )
    staged_path = Path(staged)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as writer:
            writer.write(payload)
            writer.flush()
            os.fsync(writer.fileno())
        target = directory / name
        os.replace(staged_path, target)
        return target
    except BaseException:
        try:
            staged_path.unlink()
        except OSError:  # pragma: no cover - already gone
            pass
        raise


def _open_shared_text(path: Path) -> Any:
    """Open a text file without blocking a Windows rename over the same name.

    POSIX lets a writer replace a file while another process has it open. On
    Windows that works only if the reader opted into delete sharing. Context
    Mesh readers are lock-free by design, so the open takes that share mode
    explicitly when the platform needs it. The caller owns the returned file.
    """
    if os.name != "nt":
        return open(path, "r", encoding="utf-8")

    import ctypes
    import msvcrt

    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    FILE_SHARE_DELETE = 0x00000004
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x00000080
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = ctypes.c_void_p
    handle = kernel32.CreateFileW(
        str(path),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        None,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        None,
    )
    if handle == INVALID_HANDLE_VALUE:
        error = ctypes.get_last_error()
        raise OSError(error, os.strerror(error), str(path))

    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
    except BaseException:
        kernel32.CloseHandle(handle)
        raise
    return os.fdopen(fd, "r", encoding="utf-8")


def read_text_shared(path: Path) -> str:
    """Read a whole file under the shared mode described above."""
    with _open_shared_text(path) as reader:
        return reader.read()


def live_manifest_path(directory: Path) -> Optional[Path]:
    manifests: List[Tuple[int, Path]] = []
    for entry in directory.iterdir():
        match = _GENERATION_MANIFEST_RE.fullmatch(entry.name)
        if match and entry.is_symlink():
            raise SessionError(
                f"{entry.name} is a symbolic link; a committed session manifest "
                "must be a regular file inside the session directory"
            )
        if match and entry.is_file():
            manifests.append((int(match.group("generation")), entry))
    if manifests:
        return max(manifests, key=lambda item: item[0])[1]
    legacy = directory / SESSION_FILE
    return legacy if legacy.is_file() else None


def read_live_manifest_text(directory: Path) -> str:
    path = live_manifest_path(directory)
    if path is None:
        raise FileNotFoundError(directory / SESSION_FILE)
    return read_text_shared(path)


def open_session_manifest_reader(directory: Path) -> Any:
    """Open the committed manifest the same way ``Session.load`` chooses it."""
    path = live_manifest_path(directory)
    if path is None:
        raise FileNotFoundError(directory / SESSION_FILE)
    return _open_shared_text(path)


def _open_lock_file(path: Path) -> Any:
    """Open the lock file without following a symlink, and check what it is.

    The lock is the one artifact that cannot use ``write_in_place``: it has to
    keep a single inode for its whole life, because that inode is what the
    kernel lock is attached to. Replacing it on every save would hand two
    processes locks on two different inodes and let both believe they had won.

    So it is opened directly, and defended directly. ``O_NOFOLLOW`` refuses a
    final component that is a symlink, and the ``fstat`` afterwards refuses a
    fifo or a device — the two other things a hostile directory can leave under
    a name you are about to write to.
    """
    flags = os.O_RDWR | os.O_CREAT
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:  # pragma: no cover - Windows
        # No O_NOFOLLOW: refuse anything that is already a link, then open.
        # A narrower guarantee than the POSIX path, and said so out loud rather
        # than left to look equivalent.
        if path.is_symlink():
            raise SessionError(
                f"{path} is a symbolic link; the session lock must be a regular "
                "file inside the session directory"
            )
    try:
        descriptor = os.open(str(path), flags | nofollow, 0o644)
    except OSError as exc:
        raise SessionError(
            f"{path} could not be opened as the session lock ({exc.strerror}); "
            "it must be a regular file inside the session directory"
        ) from None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SessionError(
                f"{path} is not a regular file; the session lock cannot be a "
                "link, a device or a fifo"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return os.fdopen(descriptor, "r+", encoding="utf-8")


def _fsync_file(path: Path) -> None:
    """Get a written file onto the disk before anything points at it.

    Without this the manifest swap can reach the platter first, leaving a
    manifest that names a file the filesystem has not finished writing — which
    is precisely the mixed state generations are here to rule out.
    """
    handle = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(handle)
    except OSError:  # pragma: no cover - some filesystems refuse; not fatal
        pass
    finally:
        os.close(handle)


def _fsync_dir(path: Path) -> None:
    """Make the manifest rename itself durable, where the platform allows it."""
    try:
        handle = os.open(str(path), os.O_RDONLY)
    except OSError:  # pragma: no cover - Windows cannot open a directory
        return
    try:
        os.fsync(handle)
    except OSError:  # pragma: no cover - nor fsync one
        pass
    finally:
        os.close(handle)


def _lock_exclusive(handle: Any) -> None:
    """Take an exclusive lock on an open file, without waiting.

    Kernel-held rather than advisory-by-convention, which matters more than the
    portability cost: a lock the kernel owns is released when the holder dies,
    however it dies. A lock file with a pid in it has to guess whether a stale
    holder is dead or merely slow, and it guesses wrong exactly when a machine
    is under the load that made the save slow in the first place.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: Any) -> None:
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _holder(path: Path) -> str:
    """Best-effort description of whoever holds the lock, for the error."""
    try:
        data = json.loads((path.parent / LOCK_META_FILE).read_text(encoding="utf-8"))
        return f"pid {data['pid']} on {data['host']}"
    except (OSError, ValueError, KeyError, TypeError):
        return "another process"


@contextmanager
def writer_lock(directory: Any) -> Iterator[Path]:
    """Serialise writers over one session directory.

    Generations make a save atomic *against a crash*. They do nothing against a
    second writer, and the failure there is worse than a torn file because it
    ends with a valid-looking manifest. Two processes reading generation 5 both
    choose 6 and overwrite each other's companions; worse, one can commit 6 and
    sweep while the other has already written 7 but not yet swapped — leaving a
    manifest naming files that the first process just deleted.

    So the lock has to span the whole transaction, from reading the current
    generation to sweeping the superseded one. Locking only the swap would
    still allow both of those.

    The lock file is created once and never deleted. Deleting it on release
    would let a third process create a *new* file at the same name while a
    fourth still holds the old inode, and both would believe they had the lock.
    """
    target = Path(directory)
    path = target / LOCK_FILE
    handle = _open_lock_file(path)
    try:
        try:
            _lock_exclusive(handle)
        except OSError:
            raise SessionLockedError(
                f"{target} is already being written by {_holder(path)}; "
                "one writer at a time"
            ) from None
        try:
            handle.seek(0)
            handle.truncate()
            holder = json.dumps({"pid": os.getpid(), "host": socket.gethostname()}) + "\n"
            handle.write(holder)
            handle.flush()
            write_in_place(
                target,
                LOCK_META_FILE,
                holder,
            )
            yield path
        finally:
            _unlock(handle)
    finally:
        handle.close()


def _bare_name(value: Any, field_name: str) -> str:
    """A filename, not a path.

    ``session.json`` names its two companion files so a reader can see the
    generation without parsing anything. Names are all it may do: a session
    directory is something you can be handed, and ``"graph":
    "../../etc/passwd"`` is not a graph.
    """
    if not isinstance(value, str):
        raise SessionError(
            f"session.{field_name} must be a string, got {type(value).__name__}"
        )
    if value != Path(value).name or value in ("", ".", ".."):
        raise SessionError(
            f"session.{field_name} must be a plain filename inside the session "
            f"directory, got {value!r}"
        )
    return value


def _contained(directory: Path, name: str, field_name: str) -> Path:
    """Resolve a companion filename, refusing one that leaves the directory.

    ``_bare_name`` stops the path from containing a traversal. It cannot stop
    the *file* from being a symlink to somewhere else, and a session directory
    is something you can be handed. Resolving both sides and requiring
    containment closes that: the loader reads what is in the directory, or it
    reads nothing.
    """
    root = Path(os.path.realpath(str(directory)))
    resolved = Path(os.path.realpath(str(directory / name)))
    if resolved != root / name:
        raise SessionError(
            f"session.{field_name} names {name!r}, which resolves to "
            f"{resolved} — outside the session directory {root}"
        )
    return directory / name


def _session_int(value: Any, field_name: str, *, minimum: int) -> int:
    # bool is an int in Python, so ``"rounds": true`` would otherwise load as 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise SessionError(
            f"session.{field_name} must be an integer, got {value!r}"
        )
    if value < minimum:
        raise SessionError(
            f"session.{field_name} must be >= {minimum}, got {value}"
        )
    return value


def _walker_policy(value: Any) -> Tuple[EdgeType, ...]:
    if not isinstance(value, list):
        raise SessionError(
            f"session.walker.policy must be a list, got {type(value).__name__}"
        )
    policy: List[EdgeType] = []
    for index, name in enumerate(value):
        if not isinstance(name, str):
            raise SessionError(
                f"session.walker.policy[{index}] must be a string, got {name!r}"
            )
        try:
            edge = EdgeType(name)
        except ValueError:
            raise SessionError(
                f"session.walker.policy[{index}] names the edge type {name!r}, "
                "which this build's ontology does not have"
            ) from None
        if edge in policy:
            raise SessionError(
                f"session.walker.policy lists {name!r} twice; the policy is an "
                "ordered preference, so a repeat is ambiguous"
            )
        policy.append(edge)
    return tuple(policy)


@dataclass
class WalkerConfig:
    """The walker settings that decide what a read returns.

    Persisted because they are not derivable and they are not cosmetic: restore
    a session saved with ``hop_budget=3`` into the default 6 and the same
    question comes back with a different answer, with nothing to say why. The
    walk *history* is not here — see ``Session.load``.
    """

    hop_budget: int = 6
    policy: Tuple[EdgeType, ...] = DEFAULT_POLICY
    flat_k: int = 40
    max_expand: int = 260

    @classmethod
    def of(cls, walker: Walker) -> "WalkerConfig":
        return cls(
            hop_budget=walker.hop_budget,
            policy=tuple(walker.policy),
            flat_k=walker.flat_k,
            max_expand=walker.max_expand,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hop_budget": self.hop_budget,
            "policy": [edge.value for edge in self.policy],
            "flat_k": self.flat_k,
            "max_expand": self.max_expand,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "WalkerConfig":
        if not isinstance(data, dict):
            raise SessionError(
                f"session.walker must be an object, got {type(data).__name__}"
            )
        for key in ("hop_budget", "policy", "flat_k", "max_expand"):
            if key not in data:
                raise SessionError(f"session.walker is missing {key!r}")
        return cls(
            hop_budget=_session_int(data["hop_budget"], "walker.hop_budget", minimum=1),
            policy=_walker_policy(data["policy"]),
            flat_k=_session_int(data["flat_k"], "walker.flat_k", minimum=1),
            max_expand=_session_int(data["max_expand"], "walker.max_expand", minimum=1),
        )

    def walker(self, graph: ContextGraph, resolver: Resolver) -> Walker:
        return Walker(
            graph,
            resolver,
            hop_budget=self.hop_budget,
            policy=self.policy,
            flat_k=self.flat_k,
            max_expand=self.max_expand,
        )


def _reconcile_execution(
    runner: Runner, execution: str, ledger: str, graph: ContextGraph
) -> None:
    """The plan and its ledger have to be about each other.

    Each is checked on its own before this — the ledger's chain verifies, the
    plan's references close against the graph — and both can pass while
    describing different runs. These are the facts that only make sense across
    the pair::

        every ledger entry names a task the plan holds
        no entry is recorded in a round the plan never reached

    Reference existence, and no more. Which *events* carry which ids is not a
    universal the engine guarantees — a DISPROVED entry names an assumption, an
    EXECUTED one names a decision, and neither shape is enforced anywhere — so
    only the ids that are present are required to point at something.

    Kept to what the engine actually guarantees in the other direction too. The
    ledger does not hold an entry for every task (a plan can be saved before
    anything runs) and a task can appear many times (attempts, audits, repairs),
    so neither count is an invariant and neither is asserted.
    """
    names = {task.name for task in runner.tasks}
    for entry in runner.ledger.entries:
        if entry.node_id is not None and entry.node_id not in graph.nodes:
            raise SessionError(
                f"{ledger} entry {entry.seq} cites node {entry.node_id!r}, "
                "which is not in this session's graph. The history refers to "
                "provenance this graph does not hold"
            )
        if (
            entry.assumption_id is not None
            and entry.assumption_id not in graph.assumptions
        ):
            raise SessionError(
                f"{ledger} entry {entry.seq} cites assumption "
                f"{entry.assumption_id!r}, which is not in this session's graph"
            )
        if entry.task not in names:
            raise SessionError(
                f"{ledger} records {entry.task!r} at entry {entry.seq}, which is "
                f"not a task in {execution}. The plan and its history are from "
                "different runs"
            )
        if entry.round > runner.round:
            raise SessionError(
                f"{ledger} records entry {entry.seq} in round {entry.round}, but "
                f"{execution} has only reached round {runner.round}. The history "
                "runs ahead of the plan it belongs to"
            )


def check_agreement(graph: ContextGraph, resolver: Resolver) -> None:
    """The invariants that span both files, and so live in neither."""
    for entity_id, label in sorted(resolver.canonical.items()):
        node = graph.nodes.get(entity_id)
        if node is None:
            raise SessionError(
                f"the resolver resolves {label!r} to {entity_id!r}, which is not "
                "a node in this session's graph"
            )
        if node.type is not NodeType.ENTITY:
            raise SessionError(
                f"the resolver resolves {label!r} to {entity_id!r}, which is a "
                f"{node.type.value} rather than an entity"
            )
        if node.label != label:
            raise SessionError(
                f"the resolver calls {entity_id!r} {label!r} but the graph calls "
                f"it {node.label!r}; the resolver's label is matched against "
                "mentions, so the two disagreeing changes what resolves"
            )


@dataclass
class Session:
    graph: ContextGraph
    resolver: Resolver
    walker: Walker
    ledger: AssumptionLedger
    rounds: int = DEFAULT_ROUNDS
    source: str = "bundled demo corpus"
    path: Optional[Path] = field(default=None, repr=False)
    # Stable filesystem target captured at load time. ``path`` keeps the caller's
    # absolute spelling for diagnostics/compatibility; this one is where the
    # loaded session actually lives, even if a symlink/junction is retargeted.
    _checkpoint_path: Optional[Path] = field(
        default=None, repr=False, compare=False
    )
    # Device/inode identity of the directory object restored at load time.
    # The path can later be rebound to a different directory; this cannot.
    _checkpoint_identity: Optional[Tuple[int, int]] = field(
        default=None, repr=False, compare=False
    )
    generation: int = 0
    runner: Optional[Runner] = None

    @classmethod
    def build(cls, rounds: int = DEFAULT_ROUNDS) -> "Session":
        result = demo_run(rounds=rounds)
        return cls(
            graph=result.graph,
            resolver=result.resolver,
            walker=result.walker,
            ledger=result.ledger,
            rounds=rounds,
        )

    @classmethod
    def from_folder(cls, folder: Any) -> "Session":
        """A session over a folder of the user's own Markdown and text files.

        Built fresh and not persistent until saved with ``--save``; reload it
        afterwards with ``--session``. ``rounds`` is 0 because no demo
        question stream was walked to warm it.
        """
        built = build_folder(folder)
        return cls(
            graph=built.graph,
            resolver=built.resolver,
            walker=built.walker,
            ledger=built.ledger,
            rounds=0,
            source=f"folder {built.root}",
        )

    def manifest(self, generation: int) -> Dict[str, Any]:
        """The commit point. Names every companion this generation holds."""
        keyed = {
            "schema": SESSION_SCHEMA,
            "version": SESSION_VERSION,
            "generation": generation,
            "rounds": self.rounds,
            "walker": WalkerConfig.of(self.walker).to_dict(),
        }
        for field_name, stem in COMPANIONS:
            if field_name in REQUIRED_COMPANIONS or self.runner is not None:
                keyed[field_name] = generation_name(stem, generation)
            else:
                keyed[field_name] = None
        keyed["ledger_head"] = (
            self.runner.ledger.head if self.runner is not None else None
        )
        return keyed

    @staticmethod
    def _live_generation(directory: Path) -> int:
        generations = []
        for entry in directory.iterdir():
            match = _GENERATION_MANIFEST_RE.fullmatch(entry.name)
            if match and not entry.is_symlink() and entry.is_file():
                generations.append(int(match.group("generation")))
        if generations:
            return max(generations)

        path = live_manifest_path(directory)
        if path is None:
            return 0
        try:
            data = json.loads(read_text_shared(path))
            generation = data["generation"]
        except (OSError, ValueError, KeyError, TypeError):
            return 0
        if isinstance(generation, bool) or not isinstance(generation, int):
            return 0
        return max(generation, 0)

    @staticmethod
    def _path_spelling(path: Path) -> str:
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    def _is_loaded_home(self, target: Path) -> bool:
        """Whether ``target`` is an attempt to write this restored session home."""
        candidates = [p for p in (self.path, self._checkpoint_path) if p is not None]
        spelling = self._path_spelling(target)
        if any(self._path_spelling(candidate) == spelling for candidate in candidates):
            return True
        if self._checkpoint_identity is not None:
            try:
                return _directory_identity(target) == self._checkpoint_identity
            except SessionError:
                return False
        return any(_same_session_directory(target, candidate) for candidate in candidates)

    def _assert_loaded_home_identity(self, target: Path) -> None:
        if self._checkpoint_identity is None:
            return
        current = _directory_identity(target)
        if current != self._checkpoint_identity:
            raise SessionError(
                f"{target} no longer names the directory this session restored; "
                "refusing to checkpoint into a replacement filesystem object"
            )

    def _checkpoint_target(self) -> Path:
        target = self._checkpoint_path or self.path
        if target is None:
            raise SessionError(
                "this session was built rather than loaded, so it has no "
                "directory to check point into — save it somewhere first, then "
                "load it from there"
            )
        self._assert_loaded_home_identity(target)
        return target

    def save(self, directory: Any) -> Path:
        target = Path(directory)
        # Refuse a replaced/missing loaded home before mkdir or lock metadata can
        # create anything at the replacement spelling. Re-check under the writer
        # lock in _commit to narrow the remaining name-to-object race.
        if self._is_loaded_home(target):
            self._assert_loaded_home_identity(target)
        target.mkdir(parents=True, exist_ok=True)
        with writer_lock(target):
            return self._commit(target)

    def _commit(self, target: Path) -> Path:
        is_home = self._is_loaded_home(target)
        if is_home:
            self._assert_loaded_home_identity(target)
        live = self._live_generation(target)
        if is_home and live != self.generation:
            raise SessionError(
                f"{target} is at generation {live} but this session last "
                f"committed generation {self.generation}; another writer has "
                "committed since, and saving now would discard their work"
            )
        generation = live + 1
        payload = self.manifest(generation)
        write_in_place(target, payload["graph"], self.graph.to_json())
        write_in_place(target, payload["resolver"], self.resolver.to_json())
        if self.runner is not None:
            write_in_place(target, payload["execution"], self.runner.to_json())
            write_in_place(target, payload["ledger"], self.runner.ledger.to_json())
        manifest_payload = (
            json.dumps(
                payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
            )
            + "\n"
        )
        write_in_place(target, manifest_name(generation), manifest_payload)
        try:
            write_in_place(target, SESSION_FILE, manifest_payload)
        except PermissionError:
            if os.name != "nt":
                raise
        _fsync_dir(target)
        self._sweep(
            target,
            keep={SESSION_FILE, LOCK_FILE, LOCK_META_FILE, manifest_name(generation)}
            | {payload[name] for name, _ in COMPANIONS if payload[name] is not None},
        )
        self.generation = generation
        return target

    @staticmethod
    def _sweep(directory: Path, keep: Any) -> None:
        for entry in directory.iterdir():
            name = entry.name
            if name in keep or not entry.is_file():
                continue
            superseded = name.endswith(".json") and (
                any(name.startswith(f"{stem}-") for _, stem in COMPANIONS)
                or name.startswith(f"{SESSION_MANIFEST_STEM}-")
            )
            abandoned = name.startswith(STAGING_PREFIX) and name.endswith(STAGING_SUFFIX)
            if superseded or abandoned:
                try:
                    entry.unlink()
                except OSError:  # pragma: no cover - cleanup is best effort
                    pass

    def checkpoint(self) -> Path:
        target = self._checkpoint_target()
        self.save(target)
        # Keep the established caller-visible return contract even when I/O is
        # pinned to a resolved symlink, junction or Windows short-path target.
        return self.path or target

    @classmethod
    def load(
        cls, directory: Any, *, registry: Optional[TaskRegistry] = None
    ) -> "Session":
        target = Path(directory)
        if not target.is_dir():
            raise SessionError(f"{target} is not a session directory")
        # Keep the caller-visible absolute spelling, but pin I/O to the directory
        # it names *now*. A symlink/junction can be retargeted later just as CWD
        # can change later; neither may redirect a restored session's checkpoint.
        display_target = target.absolute()
        target = display_target.resolve()
        for attempt in range(LOAD_ATTEMPTS):
            identity_before = _directory_identity(target)
            before = cls._live_generation(target)
            try:
                loaded = cls._load_once(target, registry)
                identity_after = _directory_identity(target)
                if identity_after != identity_before:
                    raise SessionError(
                        f"{target} was replaced while this session was loading; "
                        "refusing a mixed filesystem identity"
                    )
                loaded.path = display_target
                loaded._checkpoint_path = target
                loaded._checkpoint_identity = identity_before
                loaded.source = f"session directory {display_target}"
                return loaded
            except _Swept as swept:
                if cls._live_generation(target) == before:
                    raise SessionError(
                        f"{SESSION_FILE} names {swept.name!r} as the "
                        f"{swept.field_name}, but no such file is in the "
                        "session directory"
                    ) from None
                if attempt + 1 == LOAD_ATTEMPTS:
                    raise SessionError(
                        f"{target} was committed to {LOAD_ATTEMPTS} times while "
                        "this read was in progress; giving up rather than spinning"
                    ) from None
                time.sleep(LOAD_BACKOFF * (attempt + 1))
        raise AssertionError("unreachable")  # pragma: no cover

    @classmethod
    def _load_once(
        cls, target: Path, registry: Optional[TaskRegistry] = None
    ) -> "Session":
        session_path = live_manifest_path(target)
        if session_path is None:
            raise SessionError(
                f"{target} has no {SESSION_FILE}, so it is not a session "
                "directory this build can read"
            )
        try:
            raw = read_text_shared(session_path)
        except FileNotFoundError:
            raise _Swept(SESSION_FILE, "manifest") from None
        try:
            data = json.loads(
                raw,
                parse_constant=_no_session_constants,
                object_pairs_hook=_strict_session_object,
            )
        except json.JSONDecodeError as exc:
            raise SessionError(f"{session_path} is not valid JSON: {exc}") from None

        if not isinstance(data, dict):
            raise SessionError(
                f"{SESSION_FILE} must contain an object, got {type(data).__name__}"
            )
        for key in ("schema", "version"):
            if key not in data:
                raise SessionError(f"{SESSION_FILE} is missing {key!r}")
        if data["schema"] != SESSION_SCHEMA:
            raise SessionError(
                f"not a {SESSION_SCHEMA} directory: schema is {data['schema']!r}"
            )
        version = data["version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise SessionError(f"session version must be an integer, got {version!r}")
        if version not in READABLE_VERSIONS:
            raise SessionError(
                f"session version {version!r} cannot be read by this build, "
                f"which writes {SESSION_VERSION} and reads "
                f"{', '.join(str(v) for v in READABLE_VERSIONS)}"
            )
        _exact_session_keys(data, version)
        declared_head = _declared_ledger_head(data, version)

        rounds = _session_int(data["rounds"], "rounds", minimum=0)
        generation = _session_int(data["generation"], "generation", minimum=1)
        config = WalkerConfig.from_dict(data["walker"])

        expected_fields = [name for name, _ in COMPANIONS]
        if version == 1:
            expected_fields = list(REQUIRED_COMPANIONS)

        paths: Dict[str, Optional[Path]] = {name: None for name, _ in COMPANIONS}
        for field_name, stem in COMPANIONS:
            if field_name not in expected_fields:
                continue
            if data[field_name] is None:
                if field_name in REQUIRED_COMPANIONS:
                    raise SessionError(
                        f"{SESSION_FILE} names no {field_name}; a session "
                        "without one is not a session"
                    )
                continue
            name = _bare_name(data[field_name], field_name)
            expected = generation_name(stem, generation)
            if name != expected:
                raise SessionError(
                    f"{SESSION_FILE} is at generation {generation} but names "
                    f"{name!r} as the {field_name}; this build writes "
                    f"{expected!r} for that generation"
                )
            path = _contained(target, name, field_name)
            if not path.is_file():
                raise _Swept(name, field_name)
            paths[field_name] = path

        graph_path, resolver_path = paths["graph"], paths["resolver"]
        assert graph_path is not None and resolver_path is not None
        try:
            graph = ContextGraph.load_json(graph_path)
        except FileNotFoundError:
            raise _Swept(graph_path.name, "graph") from None
        except ValueError as exc:
            raise SessionError(f"{graph_path.name}: {exc}") from exc
        try:
            resolver = Resolver.load_json(resolver_path)
        except FileNotFoundError:
            raise _Swept(resolver_path.name, "resolver") from None
        except ValueError as exc:
            raise SessionError(f"{resolver_path.name}: {exc}") from exc

        check_agreement(graph, resolver)

        runner = None
        if paths["execution"] is not None:
            assert paths["ledger"] is not None
            assert declared_head is not None
            runner = cls._load_execution(
                paths["execution"],
                paths["ledger"],
                graph=graph,
                registry=registry,
                expect_head=declared_head,
            )

        return cls(
            graph=graph,
            resolver=resolver,
            walker=config.walker(graph, resolver),
            ledger=AssumptionLedger(graph),
            rounds=rounds,
            source=f"session directory {target}",
            path=target,
            generation=generation,
            runner=runner,
        )

    @staticmethod
    def _load_execution(
        execution_path: Path,
        ledger_path: Path,
        *,
        graph: ContextGraph,
        registry: Optional[TaskRegistry],
        expect_head: Optional[str] = None,
    ) -> Runner:
        if registry is None:
            raise SessionError(
                f"{execution_path.name} holds an execution plan, but no "
                "TaskRegistry was given. A checkpoint names its workers and "
                "this process has to say what those names mean; pass "
                "registry= to Session.load"
            )
        try:
            ledger = RunLedger.load_json(ledger_path, expect_head=expect_head)
        except FileNotFoundError:
            raise _Swept(ledger_path.name, "ledger") from None
        except (ExecutionError, ValueError) as exc:
            raise SessionError(f"{ledger_path.name}: {exc}") from exc
        try:
            runner = Runner.load_json(
                execution_path, graph=graph, registry=registry, ledger=ledger
            )
        except FileNotFoundError:
            raise _Swept(execution_path.name, "execution") from None
        except (ExecutionError, ValueError) as exc:
            raise SessionError(f"{execution_path.name}: {exc}") from exc
        _reconcile_execution(runner, execution_path.name, ledger_path.name, graph)
        return runner

    def assumption_ids(self) -> list:
        return sorted(self.graph.assumptions)

    def describe(self) -> dict:
        live_counts = self.graph.type_counts()
        persistent = self.path is not None
        return {
            "source": self.source,
            "persistent": persistent,
            "note": (
                "Restored from disk. Structure, belief and resolution survive a "
                "restart; walk history and the ledger's event log do not."
                if persistent
                else "Rebuilt per process. Save it with --session to keep it."
            ),
            "build": self.graph.build,
            "generation": self.generation,
            "rounds": self.rounds,
            "nodes_live": sum(live_counts.values()),
            "nodes_total": len(self.graph.nodes),
            "node_types_live": live_counts,
            "node_types_total": self.graph.type_counts(live_only=False),
            "edges_live": sum(1 for e in self.graph.edges.values() if e.live),
            "edges_total": len(self.graph.edges),
            "assumptions": len(self.graph.assumptions),
        }


class Checkpointer:
    """Decides when a served session is written back to its directory."""

    def __init__(self, session: Session, policy: str = DEFAULT_CHECKPOINT) -> None:
        if policy not in CHECKPOINT_POLICIES:
            raise SessionError(
                f"unknown checkpoint policy {policy!r}; expected one of "
                + ", ".join(repr(p) for p in CHECKPOINT_POLICIES)
            )
        self.session = session
        self.policy = policy
        self.pending = 0
        self.commits = 0
        self.contended = 0

    @property
    def durable(self) -> bool:
        return self.policy != "never" and self.session.path is not None

    def record_mutation(self) -> Optional[Path]:
        self.pending += 1
        if self.durable and self.policy == "every-ask":
            return self.commit()
        return None

    def commit(self) -> Optional[Path]:
        if not self.durable or not self.pending:
            return None
        try:
            written = self.session.checkpoint()
        except SessionLockedError:
            self.contended += 1
            return None
        self.pending = 0
        self.commits += 1
        return written

    def close(self) -> Optional[Path]:
        if self.policy == "never":
            return None
        return self.commit()


_SESSION: Optional[Session] = None


def session(
    rounds: int = DEFAULT_ROUNDS, *, directory: Optional[Any] = None
) -> Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = (
            Session.build(rounds=rounds)
            if directory is None
            else Session.load(directory)
        )
    return _SESSION


def adopt(opened: Session) -> Session:
    global _SESSION
    _SESSION = opened
    return _SESSION


def reset() -> None:
    global _SESSION
    _SESSION = None


# ── command line ─────────────────────────────────────────────────────────
def add_source_arguments(parser: Any) -> Any:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--demo",
        action="store_true",
        help="build the bundled demo corpus fresh (not persistent)",
    )
    source.add_argument(
        "--folder",
        metavar="DIR",
        help=(
            "build from your own Markdown/text files under DIR "
            "(not persistent; add --save to keep it)"
        ),
    )
    source.add_argument(
        "--session",
        metavar="DIR",
        help=f"load a session directory written by --save ({SESSION_SCHEMA} v{SESSION_VERSION})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help=f"walk rounds to build the demo with (default: {DEFAULT_ROUNDS}); --demo only",
    )
    parser.add_argument(
        "--save",
        metavar="DIR",
        help="write the session to DIR and exit without serving",
    )
    parser.add_argument(
        "--checkpoint",
        choices=CHECKPOINT_POLICIES,
        default=DEFAULT_CHECKPOINT,
        help=(
            "when to write a served session back to its directory "
            f"(default: {DEFAULT_CHECKPOINT}); --session only"
        ),
    )
    return parser


def open_session(
    args: Any, *, registry: Optional[TaskRegistry] = None
) -> Session:
    if args.session is not None:
        if args.rounds is not None:
            raise SessionError(
                "--rounds builds a demo graph and has no meaning for --session; "
                "a restored session carries the rounds it was built with"
            )
        return Session.load(args.session, registry=registry)
    if getattr(args, "checkpoint", DEFAULT_CHECKPOINT) != DEFAULT_CHECKPOINT:
        raise SessionError(
            "--checkpoint decides when a served session is written back and has "
            "no meaning for --demo or --folder, which have no directory to write "
            "to; --save one first, then serve it with --session"
        )
    if getattr(args, "folder", None) is not None:
        if args.rounds is not None:
            raise SessionError(
                "--rounds builds a demo graph and has no meaning for --folder"
            )
        try:
            return Session.from_folder(args.folder)
        except IngestError as exc:
            raise SessionError(str(exc)) from exc
    return Session.build(rounds=DEFAULT_ROUNDS if args.rounds is None else args.rounds)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    registry: Optional[TaskRegistry] = None,
) -> int:
    import argparse

    parser = add_source_arguments(
        argparse.ArgumentParser(
            prog="python -m contextmesh_mcp",
            description="Write or inspect a Context Mesh session directory.",
        )
    )
    args = parser.parse_args(argv)
    try:
        opened = open_session(args, registry=registry)
        if args.save is not None:
            target = opened.save(args.save)
            print(
                json.dumps(
                    {
                        "saved": str(target),
                        "files": sorted(p.name for p in target.iterdir()),
                        **opened.manifest(opened.generation),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        print(json.dumps(opened.describe(), indent=2, sort_keys=True, default=str))
        return 0
    except SessionError as exc:
        print(f"contextmesh session: {exc}", file=__import__("sys").stderr)
        return 2
