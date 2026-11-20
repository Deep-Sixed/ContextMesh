"""Entity resolution — one id per real-world thing.

The RESOLVE stage is where most spans die. A mention that cannot be tied to a
canonical entity is dropped rather than admitted as a near-duplicate node, and
the count of what was dropped is exactly the dashboard's DROPPED AT RESOLVE.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from .model import (
    _expect_number,
    _expect_str,
    _require,
)

_NOISE = re.compile(r"[^a-z0-9 ]+")
# Only genuine corporate forms are noise. Stripping "service", "api" or a
# version number destroys identity instead of normalising it: "Retrieval
# Service" is the name of a thing, and a v2 encoder is emphatically not a v3
# encoder — the corpus this ships with says so in as many words.
_LEGAL_SUFFIX = re.compile(r"\b(inc|llc|ltd|corp|corporation|co|gmbh|plc)\b")
_ARTICLES = frozenset({"the", "a", "an", "our", "their", "its"})

#: Words long enough to look distinctive but common enough to mean nothing on
#: their own. A mention is not resolved by one of these.
_GENERIC = frozenset(
    """index service model system builder engine layer server client rebuild
    target policy data record request response cluster storage database
    pipeline account project version release feature summary context
    decision claim source entity evidence assumption encoder""".split()
)


def _is_distinctive(token: str) -> bool:
    """A single token identifies a thing only if it is long and not generic."""
    return len(token) >= 7 and token not in _GENERIC


# Pure functions of their argument, called on the same strings many times per
# resolve; a real folder can mean millions of calls without the cache.
@lru_cache(maxsize=65536)
def normalise(text: str) -> str:
    """Casefold, strip punctuation, drop articles and corporate noise."""
    lowered = _NOISE.sub(" ", text.lower())
    lowered = _LEGAL_SUFFIX.sub(" ", lowered)
    parts = [p for p in lowered.split() if p and p not in _ARTICLES]
    return " ".join(parts)


@lru_cache(maxsize=65536)
def acronym(text: str) -> str:
    parts = normalise(text).split()
    return "".join(p[0] for p in parts) if len(parts) > 1 else ""


@lru_cache(maxsize=65536)
def block_keys(name: str) -> Tuple[str, ...]:
    """The blocks a name is filed under; a mention is only ever compared with
    entities sharing one. Pure, so cached like ``normalise``."""
    norm = normalise(name)
    if not norm:
        return ()
    toks = norm.split()
    keys = {toks[0][:4], toks[-1][:4]}
    acr = acronym(name)
    if acr:
        keys.add(acr[:4])
    if len(norm) >= 4:
        keys.add(norm.replace(" ", "")[:4])
    return tuple(keys)


def _token_set(text: str) -> Set[str]:
    return set(normalise(text).split())


def similarity(a: str, b: str) -> float:
    """Jaccard over normalised tokens, with an acronym and containment bonus."""
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    na, nb = normalise(a), normalise(b)
    # "PG Vector" and "pgvector" are the same string with the spaces argued
    # about. Token overlap scores that pair at zero, so check it first.
    if na.replace(" ", "") == nb.replace(" ", ""):
        return 0.95

    inter = len(ta & tb)
    union = len(ta | tb)
    score = inter / union
    # Containment is strong evidence ("the pgvector extension" ⊃ "pgvector"),
    # but a single shared *generic* word ("index") is not. Trust containment
    # when the smaller side carries two tokens, or when the one token it does
    # carry is distinctive enough to be an identifier on its own.
    if ta <= tb or tb <= ta:
        smaller = ta if len(ta) <= len(tb) else tb
        distinctive = len(smaller) == 1 and _is_distinctive(next(iter(smaller)))
        if len(smaller) >= 2 or distinctive:
            score = max(score, 0.85)
    if na and (na == acronym(b) or nb == acronym(a)):
        score = max(score, 0.9)
    return score


@dataclass
class ResolutionRecord:
    """One mention's fate at RESOLVE. Kept whether it resolved or not."""

    mention: str
    canonical_id: Optional[str]
    canonical_label: Optional[str]
    score: float
    reason: str

    @property
    def resolved(self) -> bool:
        return self.canonical_id is not None

    def to_dict(self) -> Dict[str, object]:
        return {
            "mention": self.mention,
            "canonical_id": self.canonical_id,
            "canonical_label": self.canonical_label,
            "score": self.score,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ResolutionRecord":
        where = "resolution record"
        if not isinstance(data, dict):
            raise ResolverSnapshotError(
                f"{where} must be an object, got {type(data).__name__}"
            )
        with _resolver_errors():
            canonical_id = _require(data, "canonical_id", where)
            canonical_label = _require(data, "canonical_label", where)
            record = cls(
                mention=_expect_str(_require(data, "mention", where), f"{where}.mention"),
                # ``null`` here is a resolution that *failed*, which is a fact
                # worth keeping. Only a missing key is an error.
                canonical_id=(
                    None if canonical_id is None
                    else _expect_str(canonical_id, f"{where}.canonical_id")
                ),
                canonical_label=(
                    None if canonical_label is None
                    else _expect_str(canonical_label, f"{where}.canonical_label")
                ),
                score=_expect_number(_require(data, "score", where), f"{where}.score"),
                reason=_expect_str(_require(data, "reason", where), f"{where}.reason"),
            )
        # An id without a label, or a label without an id, is neither a
        # resolution nor a miss. ``resolved`` keys off the id alone, so a
        # half-null record reads as resolved and then hands back a label that
        # is not there.
        if (record.canonical_id is None) != (record.canonical_label is None):
            raise ResolverSnapshotError(
                f"{where} for {record.mention!r} has canonical_id="
                f"{record.canonical_id!r} and canonical_label="
                f"{record.canonical_label!r}; a resolution has both, a miss has "
                "neither"
            )
        return record


class ResolverSnapshotError(ValueError):
    """Raised when a resolver snapshot is not one this build can restore."""


def _no_resolver_constants(value: str) -> float:
    raise ResolverSnapshotError(
        f"resolver snapshot contains the non-JSON constant {value!r}"
    )


@contextmanager
def _resolver_errors() -> Iterator[None]:
    """One exception type at the format boundary.

    The field validators are shared with the graph's loader and raise plain
    ``ValueError``. Letting those through would mean a caller had to catch two
    types to mean one thing — "this file is not a resolver I can restore" —
    and would make ``except ResolverSnapshotError`` a filter that silently
    passes half the bad files. ``ResolverSnapshotError`` is itself a
    ``ValueError``, so callers already catching that are unaffected.
    """
    try:
        yield
    except ResolverSnapshotError:
        raise
    except ValueError as exc:
        raise ResolverSnapshotError(str(exc)) from exc


def _expect_str_map(value: Any, field: str) -> Dict[str, str]:
    if not isinstance(value, dict):
        raise ResolverSnapshotError(
            f"{field} must be an object, got {type(value).__name__}"
        )
    return {
        _expect_str(k, f"{field} key"): _expect_str(v, f"{field}[{k!r}]")
        for k, v in value.items()
    }


#: The resolver's own durable format, separate from the graph's. Keeping them
#: apart means graph snapshot v1 stays closed: query-resolution state is not
#: graph state, and a later runner or ledger format can be added the same way.
SNAPSHOT_SCHEMA = "contextmesh.resolver"
SNAPSHOT_VERSION = 1


@dataclass
class Resolver:
    """Alias-table + blocking + scoring resolver.

    ``threshold`` is deliberately high: admitting a wrong merge corrupts every
    walk that later crosses the entity, while dropping a mention only costs one
    span, and the drop is recorded.

    Three of these four fields are *learned*, which is why the resolver has to
    be persisted rather than rebuilt from the graph's entities:

    - ``resolve`` writes back into ``aliases`` when a mention clears the
      threshold by score, so a surface form the graph never stored resolves
      instantly next time.
    - ``blocks`` is built by ``register`` from labels *and* explicitly
      registered aliases, so it is not a function of ``canonical`` alone.
    - ``log`` is the record every health signal counts.
    """

    threshold: float = 0.62
    canonical: Dict[str, str] = field(default_factory=dict)  # id -> label
    aliases: Dict[str, str] = field(default_factory=dict)  # normalised -> id
    blocks: Dict[str, Set[str]] = field(default_factory=dict)  # key -> ids
    log: List[ResolutionRecord] = field(default_factory=list)

    # ── registration ─────────────────────────────────────────────────────
    def register(self, entity_id: str, label: str, aliases: Iterable[str] = ()) -> None:
        self.canonical[entity_id] = label
        for name in (label, *aliases):
            key = normalise(name)
            if key:
                self.aliases[key] = entity_id
            for block in self._block_keys(name):
                self.blocks.setdefault(block, set()).add(entity_id)

    def _block_keys(self, name: str) -> Sequence[str]:
        return block_keys(name)

    # ── resolution ───────────────────────────────────────────────────────
    def match(self, mention: str) -> Optional[Tuple[str, float, str]]:
        """Best canonical id for a mention, without logging or learning.

        Used when registering a new name: a name that already resolves to a
        known entity is an alias of it, not a second entity for the same thing.
        """
        key = normalise(mention)
        if not key:
            return None
        if key in self.aliases:
            eid = self.aliases[key]
            return eid, 1.0, "alias table"

        best_id, best_score = self._best_candidate(mention)
        if best_id is not None and best_score >= self.threshold:
            return best_id, best_score, "scored match"
        return None

    def near_miss(self, mention: str) -> Tuple[Optional[str], float]:
        """The candidate that came closest without clearing the threshold."""
        return self._best_candidate(mention)

    def _best_candidate(self, mention: str) -> Tuple[Optional[str], float]:
        """Highest-scoring candidate for a mention, resolved deterministically.

        Blocking collects candidates into a set, and iterating a set of strings
        follows Python's per-process hash salt. Two mentions that tie on score
        would then resolve to different entities in different processes — so the
        same corpus built twice produced different edges. Sorting the candidates
        makes the tie-break the entity id, which is stable everywhere.
        """
        candidates: Set[str] = set()
        for block in self._block_keys(mention):
            candidates |= self.blocks.get(block, set())

        best_id, best_score = None, 0.0
        for eid in sorted(candidates):
            score = similarity(mention, self.canonical[eid])
            if score > best_score:
                best_id, best_score = eid, score
        return best_id, best_score

    # ── persistence ──────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        """The resolver's durable state.

        ``blocks`` is persisted rather than rebuilt, which is where the analogy
        with the graph's ``_out``/``_in`` breaks. Those are a function of the
        edge list; ``blocks`` is a function of *how* aliases arrived. A learned
        alias adds no block key, a registered one does, and the alias table
        does not record which is which — so rebuilding from ``canonical`` loses
        keys, and rebuilding from ``canonical`` plus ``aliases`` invents keys
        the resolver never had. Either way candidate sets change, and with them
        what resolves.

        Sets are written as sorted lists so the file is byte-stable.
        """
        return {
            "schema": SNAPSHOT_SCHEMA,
            "version": SNAPSHOT_VERSION,
            "threshold": self.threshold,
            "canonical": dict(self.canonical),
            "aliases": dict(self.aliases),
            "blocks": {key: sorted(ids) for key, ids in self.blocks.items()},
            "log": [record.to_dict() for record in self.log],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Resolver":
        """Rebuild a resolver from a snapshot, refusing one it cannot restore.

        Same discipline as the graph loader: required fields, checked types, no
        coercion, no null-to-empty, and references that have to point at
        something. Every refusal leaves as ``ResolverSnapshotError``.
        """
        with _resolver_errors():
            return cls._restore(data)

    @classmethod
    def _restore(cls, data: Dict[str, Any]) -> "Resolver":
        if not isinstance(data, dict):
            raise ResolverSnapshotError(
                f"resolver snapshot must be an object, got {type(data).__name__}"
            )
        for key in ("schema", "version", "threshold", "canonical", "aliases", "blocks", "log"):
            if key not in data:
                raise ResolverSnapshotError(f"resolver snapshot is missing {key!r}")
        if data["schema"] != SNAPSHOT_SCHEMA:
            raise ResolverSnapshotError(
                f"not a {SNAPSHOT_SCHEMA} snapshot: schema is {data['schema']!r}"
            )
        version = data["version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise ResolverSnapshotError(
                f"resolver version must be an integer, got {version!r}"
            )
        if version != SNAPSHOT_VERSION:
            raise ResolverSnapshotError(
                f"resolver snapshot version {version!r} cannot be read by this "
                f"build, which writes and reads version {SNAPSHOT_VERSION}"
            )

        threshold = _expect_number(data["threshold"], "resolver.threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ResolverSnapshotError(
                f"resolver.threshold must be between 0 and 1, got {threshold}"
            )

        canonical = _expect_str_map(data["canonical"], "resolver.canonical")
        aliases = _expect_str_map(data["aliases"], "resolver.aliases")

        raw_blocks = data["blocks"]
        if not isinstance(raw_blocks, dict):
            raise ResolverSnapshotError(
                f"resolver.blocks must be an object, got {type(raw_blocks).__name__}"
            )
        blocks: Dict[str, Set[str]] = {}
        for key, ids in raw_blocks.items():
            _expect_str(key, "resolver.blocks key")
            if not isinstance(ids, list):
                raise ResolverSnapshotError(
                    f"resolver.blocks[{key!r}] must be a list, got {type(ids).__name__}"
                )
            blocks[key] = {
                _expect_str(i, f"resolver.blocks[{key!r}][{n}]") for n, i in enumerate(ids)
            }

        raw_log = data["log"]
        if not isinstance(raw_log, list):
            raise ResolverSnapshotError(
                f"resolver.log must be a list, got {type(raw_log).__name__}"
            )
        log = [ResolutionRecord.from_dict(row) for row in raw_log]

        # References, once every table exists. An alias or block pointing at an
        # entity the resolver does not know would resolve a mention to an id
        # that is not there — a KeyError on the next ask rather than on load.
        for alias, eid in aliases.items():
            if eid not in canonical:
                raise ResolverSnapshotError(
                    f"resolver.aliases[{alias!r}] names {eid!r}, which is not canonical"
                )
        for key, ids in blocks.items():
            for eid in ids:
                if eid not in canonical:
                    raise ResolverSnapshotError(
                        f"resolver.blocks[{key!r}] names {eid!r}, which is not canonical"
                    )
        for record in log:
            if record.canonical_id is None:
                continue
            if record.canonical_id not in canonical:
                raise ResolverSnapshotError(
                    f"resolver.log names {record.canonical_id!r}, which is not canonical"
                )
            # ``resolve`` reads the label straight out of ``canonical``, so the
            # two can only disagree in a file someone else wrote. Letting it
            # through would put a name in the log that the resolver would never
            # have produced for that id, and the log is what the health signals
            # and the borderline report are computed from.
            if record.canonical_label != canonical[record.canonical_id]:
                raise ResolverSnapshotError(
                    f"resolver.log records {record.mention!r} resolving to "
                    f"{record.canonical_id!r} as {record.canonical_label!r}, but "
                    f"the resolver calls that entity "
                    f"{canonical[record.canonical_id]!r}"
                )

        return cls(
            threshold=threshold,
            canonical=canonical,
            aliases=aliases,
            blocks=blocks,
            log=log,
        )

    def to_json(self) -> str:
        """The snapshot as text, so a caller can place the bytes itself."""
        return (
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )

    def save_json(self, path: Any) -> Any:
        target = Path(path)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, path: Any) -> "Resolver":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text, parse_constant=_no_resolver_constants))

    def resolve(self, mention: str) -> ResolutionRecord:
        if not normalise(mention):
            record = ResolutionRecord(mention, None, None, 0.0, "empty after normalisation")
            self.log.append(record)
            return record

        hit = self.match(mention)
        if hit is not None:
            eid, score, reason = hit
            if reason == "scored match":
                self.aliases[normalise(mention)] = eid  # learn it for next time
            record = ResolutionRecord(mention, eid, self.canonical[eid], score, reason)
            self.log.append(record)
            return record

        best_id, best_score = self.near_miss(mention)
        reason = (
            f"best candidate {self.canonical[best_id]!r} scored "
            f"{best_score:.2f} < {self.threshold:.2f}"
            if best_id
            else "no candidate in any block"
        )
        record = ResolutionRecord(mention, None, None, best_score, reason)
        self.log.append(record)
        return record

    # ── reporting ────────────────────────────────────────────────────────
    @property
    def resolved_count(self) -> int:
        return sum(1 for r in self.log if r.resolved)

    @property
    def dropped_count(self) -> int:
        return sum(1 for r in self.log if not r.resolved)

    def unresolved(self) -> List[ResolutionRecord]:
        """Distinct mentions that never resolved, first attempt each.

        The log records every attempt, and the same noisy word is tried on every
        question — counting attempts would make the health signal a measure of
        traffic rather than of the graph.
        """
        seen: Set[str] = set()
        out: List[ResolutionRecord] = []
        for record in self.log:
            if record.resolved:
                continue
            key = normalise(record.mention)
            if key in seen:
                continue
            seen.add(key)
            out.append(record)
        return out

    def borderline(self, floor: float = 0.25) -> List[ResolutionRecord]:
        """Unresolved mentions that came close to a real entity.

        A mention with no candidate at all is just prose — the extractor offers
        the resolver every phrase in a sentence, and most of them are nothing.
        The mentions worth a human's attention are the near misses: something in
        the graph nearly matched, and the alias table or the threshold is why it
        did not.
        """
        return [r for r in self.unresolved() if r.score >= floor]

    def collapsed(self) -> Dict[str, List[str]]:
        """canonical id -> the distinct surface forms that folded into it."""
        out: Dict[str, List[str]] = {}
        for record in self.log:
            if record.canonical_id:
                forms = out.setdefault(record.canonical_id, [])
                if record.mention not in forms:
                    forms.append(record.mention)
        return out
