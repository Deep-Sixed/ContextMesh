"""Build a graph from a folder of real files instead of the bundled corpus.

The bundled corpus hands the pipeline three things a real folder does not have:
a list of entity names, relation hints, and clean prose. This module supplies
the first and third from the files themselves and leaves the second empty —
a ``contradicts`` or ``supersedes`` edge is an interpretation, and nothing here
is allowed to invent one.

Two rules hold for everything it writes:

* **Every claim is a verbatim span of a file, and says where.** The text the
  pipeline reads is the file with its markup blanked out *in place*, so a
  claim's provenance span is a byte-for-byte offset into the original file.
  Each claim also carries ``path``, ``line_start``, ``line_end`` and
  ``location`` (``docs/a.md:12-13``).
* **Nothing extracted is presented as verified.** Each claim carries
  ``verified: False``, ``extraction: "verbatim-rule"`` and a ``confidence`` of
  ``high``, ``medium`` or ``low``. Confidence grades how cleanly the span was
  lifted (a whole sentence from a paragraph, a list item, a heading), *not*
  whether the author was right. Entities carry a ``confidence`` grading how
  they were found: marked up explicitly by the author, or inferred from
  capitalisation.

Phase 1 reads Markdown and plain text only, with no new dependencies. It does
not follow symlinks out of the folder, and it skips hidden, vendored and build
directories. A build reads at most ``DEFAULT_MAX_TOTAL_BYTES`` of text in all
(4 MB unless raised), because the whole graph lives in memory; files past the
cap are skipped and reported, never silently dropped.
"""

from __future__ import annotations

import bisect
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .assumptions import AssumptionLedger
from .graph import ContextGraph
from .model import NodeType, slug
from .pipeline import BuildReport, Document, Pipeline
from .resolve import Resolver
from .traverse import Walker

MARKDOWN_SUFFIXES = frozenset({".md", ".markdown", ".mdx"})
TEXT_SUFFIXES = frozenset({".txt", ".rst", ".text"})
DEFAULT_SUFFIXES = MARKDOWN_SUFFIXES | TEXT_SUFFIXES

#: Directory names never descended into, on top of every hidden directory.
SKIP_DIRS = frozenset({
    "node_modules", "__pycache__", "venv", "env", "site-packages",
    "dist", "build", "target", "vendor", "bower_components",
})

DEFAULT_MAX_FILES = 2000
DEFAULT_MAX_BYTES = 1_000_000
#: The most text one build reads, across all files. A build holds the whole
#: graph in memory, and measured cost is roughly 350 MB of RAM and 15 seconds
#: per MB of Markdown (embedding vectors and the resolver's log grow with
#: every claim). 4 MB is about 600,000 words: roughly 1.4 GB and a minute.
DEFAULT_MAX_TOTAL_BYTES = 4_000_000
#: The most entity names one build registers. Past this the weakest go first.
DEFAULT_MAX_ENTITIES = 1500

EXTRACTION_METHOD = "verbatim-rule"

# Confidence vocabulary. A label, not a probability: there is no calibration
# behind these that would justify a number.
HIGH, MEDIUM, LOW = "high", "medium", "low"


class IngestError(Exception):
    """The folder cannot be ingested at all (missing, empty, not a directory)."""


# ── reading ──────────────────────────────────────────────────────────────
@dataclass
class Block:
    start: int
    end: int
    kind: str  # paragraph | list | heading | quote | text


@dataclass
class ParsedFile:
    path: str  # relative, forward slashes
    raw: str
    clean: str  # same length as raw; markup replaced by spaces
    blocks: List[Block]
    title: str
    marked: List[str]  # names the author marked up: headings, code, links
    mtime: float

    def __post_init__(self) -> None:
        self._line_starts = [0] + [m.end() for m in re.finditer("\n", self.raw)]

    def line_of(self, offset: int) -> int:
        """1-based line number of a character offset."""
        return bisect.bisect_right(self._line_starts, offset)


@dataclass
class Skipped:
    path: str
    reason: str


@dataclass
class FolderBuild:
    """What a folder ingest produced, ready to walk or wrap in a session."""

    root: Path
    graph: ContextGraph
    resolver: Resolver
    walker: Walker
    ledger: AssumptionLedger
    report: BuildReport
    documents: List[Document]
    files: List[ParsedFile]
    skipped: List[Skipped] = field(default_factory=list)
    entity_confidence: Dict[str, str] = field(default_factory=dict)
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES

    def summary(self) -> Dict[str, Any]:
        live = [n for n in self.graph.nodes.values() if n.live]
        by_type = Counter(n.type.value for n in live)
        confidence = Counter(
            n.attrs.get("confidence")
            for n in live
            if n.type is NodeType.CLAIM and "confidence" in n.attrs
        )
        return {
            "root": str(self.root),
            "files_read": len(self.files),
            "files_skipped": len(self.skipped),
            "bytes_read": sum(len(f.raw.encode("utf-8")) for f in self.files),
            "max_total_bytes": self.max_total_bytes,
            "nodes_live": len(live),
            "edges_live": sum(1 for e in self.graph.edges.values() if e.live),
            "by_type": dict(sorted(by_type.items())),
            "claim_confidence": {k: confidence.get(k, 0) for k in (HIGH, MEDIUM, LOW)},
        }


def iter_files(
    root: Path,
    *,
    suffixes: Iterable[str] = DEFAULT_SUFFIXES,
    max_files: int = DEFAULT_MAX_FILES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> Tuple[List[Path], List[Skipped]]:
    """Files under ``root`` worth reading, in a stable order, plus what was skipped.

    Files are admitted in path order while they fit under ``max_total_bytes``;
    one that would take the total past it is skipped and the walk goes on, so
    a single large file early in the listing does not shut out everything
    after it. The result is still fixed by the folder's contents alone.

    Symlinks are never followed: a link can point outside the folder the user
    chose, and the graph would then hold text they did not ask to share.
    """
    wanted = {s.lower() for s in suffixes}
    found: List[Path] = []
    skipped: List[Skipped] = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".") and d not in SKIP_DIRS
            and not os.path.islink(os.path.join(dirpath, d))
        )
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if name.startswith(".") or path.suffix.lower() not in wanted:
                continue
            try:
                info = path.lstat()
            except OSError as exc:
                skipped.append(Skipped(rel, f"unreadable: {exc.strerror or exc}"))
                continue
            if stat.S_ISLNK(info.st_mode):
                skipped.append(Skipped(rel, "symlink"))
                continue
            if not stat.S_ISREG(info.st_mode):
                # A FIFO, socket or device named *.md would block or stream
                # forever when read.
                skipped.append(Skipped(rel, "not a regular file"))
                continue
            if info.st_size > max_bytes:
                skipped.append(Skipped(rel, f"larger than {max_bytes} bytes"))
                continue
            if len(found) >= max_files:
                skipped.append(Skipped(rel, f"over the {max_files}-file limit"))
                continue
            if total + info.st_size > max_total_bytes:
                skipped.append(Skipped(rel, total_limit_reason(max_total_bytes)))
                continue
            total += info.st_size
            found.append(path)
    return found, skipped


# ── markdown, blanked in place ───────────────────────────────────────────
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
# Only the opening hashes are matched; the caller strips a closing run itself.
# Matching the text lazily against a trailing `\s*#*\s*$` is quadratic on a
# heading line padded with whitespace.
_HEADING = re.compile(r"^(\s{0,3}#{1,6})(?=\s|$)")
_SETEXT = re.compile(r"^\s{0,3}(=+|-+)\s*$")
_LIST = re.compile(r"^(\s*(?:[-*+]|\d{1,3}[.)])\s+(?:\[[ xX]\]\s+)?)")
_QUOTE = re.compile(r"^(\s*(?:>\s?)+)")
_TABLE = re.compile(r"^\s*\|")
_RULE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_HTML_TAG = re.compile(r"<[^>\n]{1,200}>")
# Inline patterns are bounded: unbounded, each opening marker on a long line
# scans to the end of it, and one 200KB line of "**x " takes minutes. Real
# emphasis, link text and code spans are far shorter than these caps; markup
# longer than a cap is left in the text rather than parsed.
_IMAGE = re.compile(r"!\[([^\]\n]{0,120})\]\([^)\n]{0,500}\)")
_LINK = re.compile(r"\[([^\]\n]{1,120})\]\([^)\n]{0,500}\)")
_REF_LINK = re.compile(r"\[([^\]\n]{1,120})\]\[[^\]\n]{0,120}\]")
_WIKILINK = re.compile(r"\[\[([^\]|\n]{1,120})(\|[^\]\n]{0,120})?\]\]")
_CODE_SPAN = re.compile(r"(`{1,10})([^`\n]{1,200}?)\1")
_EMPHASIS = re.compile(r"(?<![\w*])(\*{1,3}|_{2,3})(?=\S)(.{1,120}?)(?<=\S)\1(?![\w*])")
_LINK_DEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s+\S")


def _blank(chars: List[str], start: int, end: int) -> None:
    for i in range(start, end):
        if chars[i] != "\n":
            chars[i] = " "


#: Lines longer than this keep their inline markup. Real prose never has one;
#: a minified blob or generated file does, and the inline patterns cost
#: line length times their cap.
MAX_INLINE_LINE = 20_000


def _blank_inline(line: str, marked: List[str]) -> str:
    """Strip inline markup from one line without changing its length."""
    if len(line) > MAX_INLINE_LINE:
        return line
    chars = list(line)

    def keep_group(pattern: re.Pattern, group: int, record: bool) -> None:
        for m in pattern.finditer("".join(chars)):
            if record:
                marked.append(m.group(group))
            _blank(chars, m.start(), m.start(group))
            _blank(chars, m.end(group), m.end())

    for m in _HTML_TAG.finditer(line):
        _blank(chars, m.start(), m.end())
    keep_group(_IMAGE, 1, False)
    keep_group(_WIKILINK, 1, True)
    keep_group(_LINK, 1, True)
    keep_group(_REF_LINK, 1, True)
    keep_group(_CODE_SPAN, 2, True)
    keep_group(_EMPHASIS, 2, False)
    return "".join(chars)


def parse_markdown(rel: str, raw: str, mtime: float = 0.0) -> ParsedFile:
    lines = raw.splitlines(keepends=True)
    out: List[str] = []
    blocks: List[Block] = []
    marked: List[str] = []
    title = ""
    offset = 0
    fence: Optional[str] = None
    current: Optional[Block] = None

    def close() -> None:
        nonlocal current
        if current is not None and current.end > current.start:
            blocks.append(current)
        current = None

    # Front matter: a leading --- block is metadata, not prose.
    front_end = -1
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() in ("---", "..."):
                front_end = i
                break

    for index, line in enumerate(lines):
        body = line.rstrip("\r\n")
        newline = line[len(body):]
        start = offset
        offset += len(line)

        def emit(text: str) -> None:
            out.append(text + newline)

        if index <= front_end:
            emit(" " * len(body))
            continue
        if fence is not None:
            emit(" " * len(body))
            if body.strip().startswith(fence):
                fence = None
            continue
        m = _FENCE.match(body)
        if m:
            close()
            fence = m.group(1)[0] * 3
            emit(" " * len(body))
            continue
        if not body.strip():
            close()
            emit(body)
            continue
        if _TABLE.match(body) or _RULE.match(body) or _LINK_DEF.match(body):
            # Table cells are not sentences; still let their code spans and
            # links count as names the author marked up.
            close()
            _blank_inline(body, marked)
            emit(" " * len(body))
            continue
        if _SETEXT.match(body) and current is not None and current.kind == "paragraph":
            # The paragraph above was a heading all along.
            current.kind = "heading"
            heading = raw[current.start:current.end].strip()
            marked.append(heading)
            title = title or heading
            close()
            emit(" " * len(body))
            continue

        h = _HEADING.match(body)
        if h:
            close()
            text = _blank_inline(body, marked)
            text = " " * len(h.group(1)) + text[len(h.group(1)):]
            trail = re.search(r"\s#+\s*$", text)
            if trail:
                text = text[: trail.start()] + " " * (len(text) - trail.start())
            clean_heading = " ".join(text.split())
            if clean_heading:
                marked.append(clean_heading)
                title = title or clean_heading
                blocks.append(Block(start + len(h.group(1)), start + len(body), "heading"))
            emit(text)
            continue

        prefix = ""
        kind = "paragraph"
        q = _QUOTE.match(body)
        if q:
            prefix, kind = q.group(1), "quote"
        rest = body[len(prefix):]
        li = _LIST.match(rest)
        if li:
            prefix, kind = prefix + li.group(1), "list"
        text = " " * len(prefix) + _blank_inline(body[len(prefix):], marked)
        emit(text)

        if kind == "list" or current is None or current.kind != kind and kind != "paragraph":
            close()
            current = Block(start + len(prefix), start + len(body), kind)
        else:
            # A continuation line joins the block above (lazy continuation).
            current.end = start + len(body)
    close()

    clean = "".join(out)
    assert len(clean) == len(raw), "markdown blanking must preserve length"
    return ParsedFile(rel, raw, clean, blocks, title or Path(rel).stem, marked, mtime)


def parse_text(rel: str, raw: str, mtime: float = 0.0) -> ParsedFile:
    blocks: List[Block] = []
    for m in re.finditer(r"\S(?:.*\S)?(?:\n(?!\s*\n)\s*\S.*)*", raw):
        span = m.group(0).rstrip()
        blocks.append(Block(m.start(), m.start() + len(span), "text"))
    first = next((b for b in blocks), None)
    title = raw[first.start:first.end].split("\n", 1)[0].strip()[:80] if first else ""
    return ParsedFile(rel, raw, raw, blocks, title or Path(rel).stem, [], mtime)


def total_limit_reason(max_total_bytes: int) -> str:
    return f"over the {max_total_bytes:,}-byte total limit"


class _Unreadable(Exception):
    """A file the walk admitted that turned out not to be safe to read."""


def _read_bounded(path: Path, max_bytes: int) -> Tuple[bytes, float]:
    """Read one regular file, never more than ``max_bytes``, never via a link.

    The walk already checked all of this, but the file can change between the
    walk and the read: swapped for a symlink, replaced by a FIFO, or grown.
    So the checks are repeated on the open descriptor, which is the thing
    actually read. ``O_NONBLOCK`` keeps opening a FIFO from hanging, and
    ``O_NOFOLLOW`` makes opening a symlink fail where the OS supports it.
    """
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise _Unreadable(f"unreadable: {exc.strerror or exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise _Unreadable("not a regular file")
        chunks: List[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 1 << 16))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise _Unreadable(f"larger than {max_bytes} bytes")
        return data, info.st_mtime
    except OSError as exc:
        raise _Unreadable(f"unreadable: {exc.strerror or exc}") from exc
    finally:
        os.close(fd)


def read_folder(
    root: Any,
    *,
    suffixes: Iterable[str] = DEFAULT_SUFFIXES,
    max_files: int = DEFAULT_MAX_FILES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> Tuple[Path, List[ParsedFile], List[Skipped]]:
    base = Path(root).expanduser()
    if not base.exists():
        raise IngestError(f"{base} does not exist")
    if not base.is_dir():
        raise IngestError(f"{base} is not a directory")
    base = base.resolve()
    if max_total_bytes < 1:
        raise IngestError("the total size limit must be at least 1 byte")
    paths, skipped = iter_files(
        base, suffixes=suffixes, max_files=max_files, max_bytes=max_bytes,
        max_total_bytes=max_total_bytes,
    )
    files: List[ParsedFile] = []
    total = 0
    for path in paths:
        rel = path.relative_to(base).as_posix()
        try:
            data, mtime = _read_bounded(path, max_bytes)
            raw = data.decode("utf-8-sig")
        except _Unreadable as exc:
            skipped.append(Skipped(rel, str(exc)))
            continue
        except UnicodeDecodeError:
            skipped.append(Skipped(rel, "not UTF-8 text"))
            continue
        # The walk budgeted by size on disk; a file can grow before it is read.
        if total + len(data) > max_total_bytes:
            skipped.append(Skipped(rel, total_limit_reason(max_total_bytes)))
            continue
        total += len(data)
        if "\x00" in raw:
            skipped.append(Skipped(rel, "binary content"))
            continue
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        parse = parse_markdown if path.suffix.lower() in MARKDOWN_SUFFIXES else parse_text
        parsed = parse(rel, raw, mtime)
        if not parsed.blocks:
            skipped.append(Skipped(rel, "no prose"))
            continue
        files.append(parsed)
    if not files:
        over = sum(s.reason == total_limit_reason(max_total_bytes) for s in skipped)
        raise IngestError(
            f"no readable Markdown or text files under {base}"
            + (f"; {over} files are each larger than the "
               f"{max_total_bytes:,}-byte total limit" if over else "")
            + (f" ({len(skipped)} skipped)" if skipped else "")
        )
    return base, files, skipped


# ── entities ─────────────────────────────────────────────────────────────
_CAPITALISED = re.compile(r"\b([A-Z][A-Za-z0-9+.-]*(?:\s+[A-Z][A-Za-z0-9+.-]*){0,4})\b")
_IDENT_OK = re.compile(r"^[A-Za-z][\w.+/-]*(?: [\w.+/-]+){0,3}$")

#: Capitalised words that start phrases without naming anything.
_STOP = frozenset("""
a an and or but the this that these those it its if then when while where what
which who why how all any each every no not none some such there here we you
they he she i our your their my is are was were be been note see also yes ok
for from with into onto of on in at to by as so do does did can may must
should would could will shall one two three first second next last new
""".split())


def _usable_name(name: str) -> Optional[str]:
    name = " ".join(name.strip().strip(".,;:!?()[]{}\"'").split())
    if not (3 <= len(name) <= 60):
        return None
    words = name.split()
    if len(words) > 5 or not re.search(r"[A-Za-z]", name):
        return None
    if all(w.lower() in _STOP for w in words):
        return None
    if words[0].lower() in _STOP and len(words) > 1:
        name = " ".join(words[1:])
        return _usable_name(name)
    return name


def discover_entities(
    files: Sequence[ParsedFile], *, max_entities: int = DEFAULT_MAX_ENTITIES
) -> Dict[str, str]:
    """Entity name -> confidence, found from the files' own text.

    ``high``: the author marked it up (heading, code span, link text) and it is
    also mentioned in prose somewhere. ``medium``: a capitalised phrase that
    appears in at least two files. A name seen once and never marked up is not
    an entity yet — it would connect nothing.
    """
    marked: Counter = Counter()
    capitalised_df: Counter = Counter()
    display: Dict[str, str] = {}

    for f in files:
        for raw_name in f.marked:
            name = _usable_name(raw_name)
            if name and _IDENT_OK.match(name):
                marked[name.lower()] += 1
                display.setdefault(name.lower(), name)
        seen: Set[str] = set()
        for block in f.blocks:
            text = f.clean[block.start:block.end]
            for m in _CAPITALISED.finditer(text):
                phrase = m.group(1)
                # The first word of a sentence is capitalised for grammar, not
                # because it names something.
                sentence_start = m.start() == 0 or re.search(r"[.!?:]\s+$", text[: m.start()])
                if sentence_start and " " not in phrase:
                    continue
                name = _usable_name(phrase)
                if name and name.lower() not in seen:
                    seen.add(name.lower())
                    display.setdefault(name.lower(), name)
        for key in seen:
            capitalised_df[key] += 1

    wanted = {_words(key) for key in display}
    counts: Counter = Counter()
    for f in files:
        counts.update(_name_counts(f.clean, wanted))
    scored: List[Tuple[int, str, str]] = []
    for key, name in display.items():
        occurrences = counts.get(_words(key), 0)
        if marked[key] and occurrences >= 1:
            scored.append((2 * occurrences + 3 * marked[key], key, HIGH))
        elif capitalised_df[key] >= 2:
            scored.append((occurrences, key, MEDIUM))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return {display[key]: conf for _, key, conf in scored[:max_entities]}


_WORD = re.compile(r"\w+")
_MAX_NGRAM = 8


def _words(text: str) -> Tuple[str, ...]:
    return tuple(_WORD.findall(text.lower()))


def _name_counts(text: str, wanted: Set[Tuple[str, ...]]) -> Counter:
    """How often each word sequence in ``wanted`` occurs in ``text``.

    One pass per file instead of one regex scan per entity name: a name occurs
    where its word sequence does, which is the same test a word-bounded search
    makes, punctuation between the words aside. Only the wanted sequences are
    kept; counting every run of words instead cost about 130 MB of memory per
    MB of text.
    """
    words = _words(text)
    counts: Counter = Counter()
    for size in sorted({len(w) for w in wanted if 0 < len(w) <= _MAX_NGRAM}):
        for i in range(len(words) - size + 1):
            gram = words[i : i + size]
            if gram in wanted:
                counts[gram] += 1
    return counts


# ── claims ───────────────────────────────────────────────────────────────
def _claim_confidence(kind: str, sentence: str) -> str:
    words = len(sentence.split())
    if kind == "heading" or words > 60:
        return LOW
    if kind in ("paragraph", "text") and sentence.rstrip()[-1:] in ".!?":
        return HIGH
    return MEDIUM


def _annotator(by_doc: Dict[str, ParsedFile]):
    kinds: Dict[str, List[Block]] = {doc_id: f.blocks for doc_id, f in by_doc.items()}

    def annotate(doc: Document, start: int, end: int, sentence: str) -> Optional[Dict[str, Any]]:
        parsed = by_doc.get(doc.id)
        if parsed is None:
            return None
        kind = "paragraph"
        for block in kinds[doc.id]:
            if block.start <= start < block.end:
                kind = block.kind
                break
        line_start = parsed.line_of(start)
        line_end = parsed.line_of(max(start, end - 1))
        where = f"{line_start}" if line_start == line_end else f"{line_start}-{line_end}"
        return {
            "path": parsed.path,
            "line_start": line_start,
            "line_end": line_end,
            "location": f"{parsed.path}:{where}",
            "block": kind,
            "extraction": EXTRACTION_METHOD,
            "confidence": _claim_confidence(kind, sentence),
            "verified": False,
        }

    return annotate


def source_id_for(rel: str) -> str:
    return slug(rel, NodeType.SOURCE.value)


def documents_from(
    files: Sequence[ParsedFile], entities: Dict[str, str]
) -> List[Document]:
    docs: List[Document] = []
    wanted = {_words(name) for name in entities}
    for f in files:
        present = _name_counts(f.clean, wanted)
        names = [name for name in entities if _words(name) in present]
        docs.append(
            Document(
                id=source_id_for(f.path),
                title=f.title,
                origin=f.path,
                text=f.clean,
                retrieved_at=datetime.fromtimestamp(f.mtime, tz=timezone.utc).date().isoformat()
                if f.mtime
                else "",
                entities=names,
                segments=[(b.start, b.end) for b in f.blocks],
            )
        )
    return docs


def build_folder(
    root: Any,
    *,
    suffixes: Iterable[str] = DEFAULT_SUFFIXES,
    max_files: int = DEFAULT_MAX_FILES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_entities: int = DEFAULT_MAX_ENTITIES,
    hop_budget: int = 6,
) -> FolderBuild:
    """Read ``root``, discover its entities, and build a walkable graph."""
    base, files, skipped = read_folder(
        root, suffixes=suffixes, max_files=max_files, max_bytes=max_bytes,
        max_total_bytes=max_total_bytes,
    )
    entities = discover_entities(files, max_entities=max_entities)
    docs = documents_from(files, entities)
    by_doc = {doc.id: f for doc, f in zip(docs, files)}

    graph = ContextGraph()
    resolver = Resolver()
    pipeline = Pipeline(graph, resolver, annotate=_annotator(by_doc))
    report = pipeline.build(docs)

    # Grade each entity by how it was found. Aliases folded into one id take
    # the best grade any of their surface forms earned.
    rank = {HIGH: 2, MEDIUM: 1, LOW: 0}
    entity_confidence: Dict[str, str] = {}
    for name, conf in entities.items():
        record = resolver.match(name)
        if record is None:
            continue
        entity_id = record[0]
        prior = entity_confidence.get(entity_id)
        if prior is None or rank[conf] > rank[prior]:
            entity_confidence[entity_id] = conf
    for entity_id, conf in entity_confidence.items():
        node = graph.nodes.get(entity_id)
        if node is not None and node.type is NodeType.ENTITY:
            node.attrs["confidence"] = conf
            node.attrs["extraction"] = "marked-up" if conf == HIGH else "capitalised-phrase"

    walker = Walker(graph, resolver, hop_budget=hop_budget)
    return FolderBuild(
        root=base,
        graph=graph,
        resolver=resolver,
        walker=walker,
        ledger=AssumptionLedger(graph),
        report=report,
        documents=docs,
        files=files,
        skipped=skipped,
        entity_confidence=entity_confidence,
        max_total_bytes=max_total_bytes,
    )
