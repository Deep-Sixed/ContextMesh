"""Build a graph from a folder of real files instead of the bundled corpus.

The bundled corpus hands the pipeline three things a real folder does not have:
a list of entity names, relation hints, and clean prose. This module supplies
the first and third from the files themselves and leaves the second empty —
a ``contradicts`` or ``supersedes`` edge is an interpretation, and nothing here
is allowed to invent one.

Two rules hold for everything it writes:

* **Every claim is a verbatim span of a file, and says where.** The text the
  pipeline reads is the file's decoded text with its markup blanked out *in
  place*, so a claim's provenance span is a character offset into that text
  (newlines normalised to ``\\n``; the source records its ``encoding`` and
  original ``newlines``). For a UTF-8 file with ``\\n`` line endings that is
  the file itself. Each claim also carries ``path``, ``line_start``,
  ``line_end`` and ``location`` (``docs/a.md:12-13``), which match the file
  whatever its encoding.
* **Nothing extracted is presented as verified.** Each claim carries
  ``verified: False``, ``extraction: "verbatim-rule"`` and a ``confidence`` of
  ``high``, ``medium`` or ``low``. Confidence grades how cleanly the span was
  lifted (a whole sentence from a paragraph, a list item, a heading), *not*
  whether the author was right. Entities carry a ``confidence`` grading how
  they were found: marked up explicitly by the author, or inferred from
  capitalisation.

A source's ``retrieved_at`` (temporal.py's *source time*) is the file's
modification date, labelled ``source_time_basis: "file-modified"`` with the
full ``modified_at`` beside it: the best evidence a folder offers of when its
content existed, and not a claim about when it was written or verified. When
the build read the folder is ``FolderBuild.ingested_at``, reported but kept
out of the graph so a rebuild of the same files is identical; inside the
graph, processing time is the build counter.

Phase 1 reads Markdown and plain text only, with no new dependencies. Text is
UTF-8 by default; UTF-16 is read when a byte-order mark declares it. It does
not follow symlinks out of the folder, and it skips hidden, vendored and build
directories. A build reads at most ``DEFAULT_MAX_TOTAL_BYTES`` of text in all
(4 MB unless raised), because the whole graph lives in memory; files past the
cap are skipped and reported, never silently dropped.
"""

from __future__ import annotations

import ast
import bisect
import hashlib
import io
import os
import re
import stat
import tokenize
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .assumptions import AssumptionLedger
from .graph import ContextGraph
from .model import AssumptionStatus, EdgeType, NodeType, slug
from .pipeline import BuildReport, Document, Pipeline
from .resolve import Resolver
from .traverse import Walker

MARKDOWN_SUFFIXES = frozenset({".md", ".markdown", ".mdx"})
TEXT_SUFFIXES = frozenset({".txt", ".rst", ".text"})
#: Source files read for their docstrings and comments, never their code.
CODE_SUFFIXES = frozenset({".py"})
PROSE_SUFFIXES = MARKDOWN_SUFFIXES | TEXT_SUFFIXES
DEFAULT_SUFFIXES = PROSE_SUFFIXES | CODE_SUFFIXES

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
    kind: str  # paragraph | list | heading | quote | text | docstring | comment
    #: For code: the qualified name the block documents or sits in
    #: ("Pipeline.build"), or "" at module level.
    symbol: str = ""


@dataclass
class ParsedFile:
    path: str  # relative, forward slashes
    raw: str
    clean: str  # same length as raw; markup replaced by spaces
    blocks: List[Block]
    title: str
    marked: List[str]  # names the author marked up: headings, code, links
    mtime: float
    encoding: str = "utf-8"
    newlines: str = "lf"  # lf | crlf | cr | mixed, before normalising to \n
    size_bytes: int = 0  # on disk, as read
    sha256: str = ""  # of the bytes on disk, as read; names this file version

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
    #: Wall-clock time this build read the folder. Reported, never written
    #: into the graph: temporal.py keeps the graph free of wall-clock time so a
    #: rebuild is reproducible, and processing time there is the build counter.
    ingested_at: str = ""

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
            "ingested_at": self.ingested_at,
            "files_read": len(self.files),
            "files_skipped": len(self.skipped),
            "bytes_read": sum(f.size_bytes for f in self.files),
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


_BOMS = (
    # UTF-32 first: its little-endian BOM begins with UTF-16's.
    (b"\xff\xfe\x00\x00", None, "UTF-32"),
    (b"\x00\x00\xfe\xff", None, "UTF-32"),
    (b"\xef\xbb\xbf", "utf-8-sig", "utf-8 (BOM)"),
    (b"\xff\xfe", "utf-16-le", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be", "utf-16-be"),
)


def decode_text(data: bytes) -> Tuple[str, str]:
    """``(text, encoding label)`` for a file's bytes, or raise ``_Unreadable``.

    UTF-8 is the default. UTF-16 is accepted only when a byte-order mark says
    so: that is the one signal that is reliable, and it is what Windows tools
    that save UTF-16 (Notepad, PowerShell's ``Out-File``) write. A file that
    looks like UTF-16 but has no BOM is skipped with a reason that says how to
    fix it, rather than guessed at.
    """
    for bom, codec, label in _BOMS:
        if data.startswith(bom):
            if codec is None:
                raise _Unreadable(f"{label} text is not supported; save it as UTF-8")
            try:
                return data.decode(codec), label
            except UnicodeDecodeError as exc:
                raise _Unreadable(f"declares {label} but does not decode as it") from exc
    bomless_utf16 = (
        "looks like UTF-16 without a byte-order mark; save it as UTF-8 "
        "or as UTF-16 with a BOM"
    )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        if _looks_like_bomless_utf16(data):
            raise _Unreadable(bomless_utf16) from exc
        raise _Unreadable("not UTF-8 text") from exc
    # NUL is valid UTF-8, so mostly-ASCII UTF-16 decodes "successfully".
    if "\x00" in text:
        raise _Unreadable(bomless_utf16 if _looks_like_bomless_utf16(data) else "binary content")
    return text, "utf-8"


def _looks_like_bomless_utf16(data: bytes) -> bool:
    """Mostly-ASCII UTF-16 has a NUL in every other byte. Used only to word
    the skip reason; such a file is never decoded on this guess."""
    sample = data[:4096]
    if len(sample) < 16:
        return False
    even, odd = sample[0::2], sample[1::2]
    return max(even.count(0) / len(even), odd.count(0) / len(odd)) > 0.4


def _newline_style(text: str) -> str:
    crlf = text.count("\r\n")
    cr = text.count("\r") - crlf
    lf = text.count("\n") - crlf
    styles = [name for name, n in (("crlf", crlf), ("cr", cr), ("lf", lf)) if n]
    if not styles:
        return "lf"
    return styles[0] if len(styles) == 1 else "mixed"


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


# ── python: docstrings and comments, blanked in place ──────────────────
#: Comments that instruct a tool rather than tell a reader something.
_PRAGMA = re.compile(
    r"^\s*(?:noqa|type:|pragma|pylint:|fmt:|isort:|ruff:|mypy:|pyright:|nosec|coding[:=])",
    re.I,
)
_CODE_CHARS = frozenset("=()[]{}")
#: A reStructuredText role in front of a code span: ":meth:`x`", ":py:func:`y`".
_REST_ROLE = re.compile(r":[A-Za-z]+(?::[A-Za-z]+)?:(?=`)")
_TRIPLE_QUOTES = ('"' * 3, "'" * 3)


def _looks_like_code(text: str) -> bool:
    """A comment run that parses as Python and is shaped like it: commented-out
    code, not prose. Prose sentences rarely parse; ones that do lack these."""
    stripped = text.strip()
    if not stripped or not (_CODE_CHARS & set(stripped)):
        return False
    try:
        ast.parse(stripped)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    return True


def _definitions(tree: ast.AST) -> List[Tuple[str, ast.AST]]:
    """(qualified name, node) for the module and every class and function."""
    out: List[Tuple[str, ast.AST]] = [("", tree)]

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}.{child.name}" if prefix else child.name
                out.append((name, child))
                visit(child, name)
            else:
                visit(child, prefix)

    visit(tree, "")
    return out


def parse_python(rel: str, raw: str, mtime: float = 0.0) -> ParsedFile:
    """A Python file's prose: its docstrings and comments, never its code.

    Like ``parse_markdown``, everything that is not prose is blanked in place,
    so the clean text has the file's length and every claim's span and line
    numbers stay true to the file. Docstring paragraphs and comment runs become
    blocks that record the definition they document or sit in. Definitions,
    imports and the module's dotted path count as names the author marked up;
    as for Markdown, they become entities only where prose mentions them.
    Pragmas (``# noqa``, ``# type:``...), shebangs and commented-out code are
    not prose. A file that is not valid Python is refused, not guessed at.
    """
    try:
        tree = ast.parse(raw, filename=rel)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        line = getattr(exc, "lineno", None)
        raise _Unreadable("not valid Python" + (f" (line {line})" if line else "")) from exc

    lines = raw.split("\n")
    line_starts = [0]
    for text in lines[:-1]:
        line_starts.append(line_starts[-1] + len(text) + 1)

    def offset(lineno: int, col_bytes: int) -> int:
        # ast columns count UTF-8 bytes; the clean text counts characters.
        text = lines[lineno - 1]
        col = len(text.encode("utf-8")[:col_bytes].decode("utf-8", errors="ignore"))
        return line_starts[lineno - 1] + col

    chars = [" " if ch != "\n" else "\n" for ch in raw]
    blocks: List[Block] = []
    marked: List[str] = []
    title = ""
    definitions = _definitions(tree)

    def keep(start: int, end: int) -> None:
        """Copy raw[start:end] into the clean text, inline markup blanked."""
        cursor = start
        while cursor < end:
            line_end = raw.find("\n", cursor, end)
            line_end = end if line_end < 0 else line_end
            line = _REST_ROLE.sub(lambda m: " " * len(m.group(0)), raw[cursor:line_end])
            chars[cursor:line_end] = list(_blank_inline(line, marked))
            cursor = line_end + 1

    def paragraphs(start: int, end: int, kind: str, symbol: str) -> None:
        """A docstring's prose paragraphs as blocks, one per list item.

        Paragraphs are separated by blank lines. A paragraph that is code — a
        doctest, the indented block after a ``::``, an example that parses as
        Python — is left blanked: it is not something the author asserts.
        """
        after_literal = False
        for m in re.finditer(r"\S(?:.*\S)?(?:\n(?![ \t]*\n)[ \t]*\S.*)*", raw[start:end]):
            text = m.group(0).rstrip()
            first = start + m.start()
            indent = first - (raw.rfind("\n", 0, first) + 1)
            literal, after_literal = after_literal and indent > 0, text.endswith("::")
            dedented = "\n".join(line.strip() for line in text.split("\n"))
            if (
                literal
                or dedented.startswith((">>>", "..."))
                or _looks_like_code(dedented)
            ):
                for i in range(first, first + len(text)):
                    if chars[i] != "\n":
                        chars[i] = " "
                continue
            # One block per list item, as in Markdown.
            cursor = first
            for item in re.finditer(r"(?m)^[ \t]*(?:[-*+]|\d{1,3}[.)])[ \t]+", text):
                item_start = first + item.start()
                if item_start > cursor:
                    blocks.append(Block(cursor, item_start - 1, kind, symbol))
                for i in range(item_start, first + item.end()):
                    chars[i] = " "
                cursor = first + item.end()
            if cursor < first + len(text):
                blocks.append(Block(cursor, first + len(text), kind, symbol))

    # Docstrings, and the names each definition gives.
    module = rel[:-3].replace("/", ".") if rel.endswith(".py") else rel
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    marked.append(module)
    for symbol, node in definitions:
        if symbol and not symbol.split(".")[-1].startswith("_"):
            marked.append(symbol)
            marked.append(symbol.split(".")[-1])
        body = getattr(node, "body", None)
        if not body or not isinstance(body[0], ast.Expr):
            continue
        value = body[0].value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        if getattr(value, "end_lineno", None) is None:
            continue
        start = offset(value.lineno, value.col_offset)
        end = offset(value.end_lineno, value.end_col_offset)
        literal = raw[start:end]
        prefix = len(literal) - len(literal.lstrip("rRbBuUfF"))
        triple = literal[prefix:prefix + 3]
        quote = triple if triple in _TRIPLE_QUOTES else literal[prefix:prefix + 1]
        inner_start, inner_end = start + prefix + len(quote), end - len(quote)
        if inner_end <= inner_start:
            continue
        keep(inner_start, inner_end)
        paragraphs(inner_start, inner_end, "docstring", symbol)
        if not symbol and not title:
            title = raw[inner_start:inner_end].strip().split("\n", 1)[0].strip()[:80]

    # Imports name things this file uses.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            marked.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            marked.extend(alias.name for alias in node.names if alias.name != "*")

    # line -> innermost definition containing it. Outer definitions are laid
    # down first and inner ones over them, so one pass serves every comment
    # (scanning every definition per comment is quadratic on large files).
    innermost = [""] * (len(lines) + 2)
    for symbol, node in sorted(
        (d for d in definitions if d[0]), key=lambda d: d[0].count(".")
    ):
        first = getattr(node, "lineno", 0)
        last = getattr(node, "end_lineno", None) or first
        for row in range(first, min(last, len(lines)) + 1):
            innermost[row] = symbol

    def enclosing(lineno: int) -> str:
        """The innermost definition whose lines include ``lineno``."""
        return innermost[lineno] if 0 <= lineno < len(innermost) else ""

    # Comments: a run of full-line comments at one indent is one block; an
    # inline comment is a block of its own.
    run: List[Tuple[int, int, int, int]] = []  # (row, col, text start, text end)

    def flush() -> None:
        if not run:
            return
        text = "\n".join(raw[a:b] for _, _, a, b in run)
        if not _looks_like_code(text):
            for _, _, a, b in run:
                keep(a, b)
            blocks.append(Block(run[0][2], run[-1][3], "comment", enclosing(run[0][0])))
        run.clear()

    try:
        found = list(tokenize.generate_tokens(io.StringIO(raw).readline))
    except (tokenize.TokenError, SyntaxError):
        found = []
    for tok in found:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        if row == 1 and tok.string.startswith("#!"):
            continue
        body = tok.string.lstrip("#")
        if body.startswith(":"):
            body = body[1:]  # Sphinx's "#:" attribute comment
        if not body.strip() or _PRAGMA.match(body):
            flush()
            continue
        hashes = len(tok.string) - len(body)
        text_start = line_starts[row - 1] + col + hashes + (len(body) - len(body.lstrip()))
        text_end = line_starts[row - 1] + col + len(tok.string)
        if lines[row - 1][:col].strip():
            flush()
            run.append((row, col, text_start, text_end))
            flush()
            continue
        if run and (run[-1][0] != row - 1 or run[-1][1] != col):
            flush()
        run.append((row, col, text_start, text_end))
    flush()

    blocks.sort(key=lambda b: b.start)
    clean = "".join(chars)
    assert len(clean) == len(raw), "python blanking must preserve length"
    return ParsedFile(rel, raw, clean, blocks, title or Path(rel).stem, marked, mtime)


def read_folder(
    root: Any,
    *,
    suffixes: Iterable[str] = DEFAULT_SUFFIXES,
    max_files: int = DEFAULT_MAX_FILES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    allow_empty: bool = False,
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
            raw, encoding = decode_text(data)
        except _Unreadable as exc:
            skipped.append(Skipped(rel, str(exc)))
            continue
        # The walk budgeted by size on disk; a file can grow before it is read.
        if total + len(data) > max_total_bytes:
            skipped.append(Skipped(rel, total_limit_reason(max_total_bytes)))
            continue
        total += len(data)
        newlines = _newline_style(raw)
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        suffix = path.suffix.lower()
        if suffix in CODE_SUFFIXES:
            parse = parse_python
        elif suffix in MARKDOWN_SUFFIXES:
            parse = parse_markdown
        else:
            parse = parse_text
        try:
            parsed = parse(rel, raw, mtime)
        except _Unreadable as exc:
            skipped.append(Skipped(rel, str(exc)))
            continue
        parsed.encoding, parsed.newlines, parsed.size_bytes = encoding, newlines, len(data)
        parsed.sha256 = hashlib.sha256(data).hexdigest()
        if not parsed.blocks:
            skipped.append(Skipped(rel, "no prose"))
            continue
        files.append(parsed)
    if not files and not allow_empty:
        over = sum(s.reason == total_limit_reason(max_total_bytes) for s in skipped)
        raise IngestError(
            f"no readable Markdown or text files under {base}"
            + (f"; {over} files are each larger than the "
               f"{max_total_bytes:,}-byte total limit" if over else "")
            + (f" ({len(skipped)} skipped)" if skipped else "")
        )
    return base, files, skipped


# ── entities ─────────────────────────────────────────────────────────────
# A capitalised phrase never crosses a sentence: words are separated by spaces
# or tabs only (not newlines), and a dot is allowed inside a word ("Node.js")
# but not at its end. Without both, "…the Shard Store.\nThe Index Builder…"
# read as one name, and the resolver then folded the two real names into it.
_CAP_WORD = r"[A-Z][A-Za-z0-9+-]*(?:\.[A-Za-z0-9+-]+)*"
_CAPITALISED = re.compile(rf"\b({_CAP_WORD}(?:[ \t]+{_CAP_WORD}){{0,4}})\b")
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
    # A comment is graded no higher than medium: comments drift from the code
    # they describe more readily than prose meant to be read.
    if kind in ("paragraph", "text", "docstring") and sentence.rstrip()[-1:] in ".!?":
        return HIGH
    return MEDIUM


def _annotator(by_doc: Dict[str, ParsedFile]):
    kinds: Dict[str, List[Block]] = {doc_id: f.blocks for doc_id, f in by_doc.items()}

    def annotate(doc: Document, start: int, end: int, sentence: str) -> Optional[Dict[str, Any]]:
        parsed = by_doc.get(doc.id)
        if parsed is None:
            return None
        kind, symbol = "paragraph", ""
        for block in kinds[doc.id]:
            if block.start <= start < block.end:
                kind, symbol = block.kind, block.symbol
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
            **({"symbol": symbol} if symbol else {}),
        }

    return annotate


def source_id_for(rel: str, sha256: str) -> str:
    """One source per *version* of a file: its path and its content hash.

    A refresh adds a new version when the content changes and keeps the old
    one as history, so evidence and provenance can name the exact text.
    """
    return slug(f"{rel}@{sha256[:16]}", NodeType.SOURCE.value)


#: Who ``AssumptionLedger.assume`` records as having taken each claim's
#: "file still says" assumption.
CLAIM_ASSUMPTION_AUTHOR = "folder-ingest"


def claim_assumption_statement(
    rel: str, sha256: str, sentence: str, generation: int = 0
) -> str:
    """What every folder claim rests on: that its file still says it.

    The statement names the file version the claim was first read from and,
    for a sentence that returned after being retired, the claim's generation.
    The version alone is not enough: a file reverted to earlier content is
    that earlier version again, so a returning sentence would otherwise name
    its dead predecessor's statement, and ``assume`` would hand back the
    rejected assumption instead of new ground.
    """
    statement = f'{rel}@{sha256[:12]} says: "{sentence}"'
    if generation:
        statement += f" (returned, generation {generation})"
    return statement


def _claim_generation(claim_id: str, sentence: str, rel: str) -> int:
    """Which generation of ``sentence`` in ``rel`` the claim ``claim_id`` is."""
    from .pipeline import claim_id_for

    generation = 0
    while claim_id_for(sentence, rel, generation) != claim_id:
        generation += 1
        if generation > 10_000:
            raise ValueError(f"{claim_id!r} is not a claim id for this sentence in {rel!r}")
    return generation


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
                id=source_id_for(f.path, f.sha256),
                title=f.title,
                origin=f.path,
                text=f.clean,
                retrieved_at=datetime.fromtimestamp(f.mtime, tz=timezone.utc).date().isoformat()
                if f.mtime
                else "",
                entities=names,
                segments=[(b.start, b.end) for b in f.blocks],
                claim_scope=f.path,
            )
        )
    return docs


def _utc_iso(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def grade_entities(
    graph: ContextGraph, resolver: Resolver, entities: Dict[str, str]
) -> Dict[str, str]:
    """Grade each entity by how it was found; returns entity id -> grade.

    Aliases folded into one id take the best grade any of their surface forms
    earned. A refresh calls this again over the whole current corpus, so a
    grade can rise but an entity is never removed.
    """
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
            current = node.attrs.get("confidence")
            if current is not None and rank.get(current, 0) >= rank[conf]:
                continue
            node.attrs["confidence"] = conf
            node.attrs["extraction"] = "marked-up" if conf == HIGH else "capitalised-phrase"
    return entity_confidence


def label_sources(
    graph: ContextGraph, docs: Sequence[Document], files: Sequence[ParsedFile]
) -> None:
    """Say what each new source version's date and offsets actually rest on."""
    for doc, parsed in zip(docs, files):
        node = graph.nodes.get(doc.id)
        if node is None:
            continue
        node.attrs.update({
            # retrieved_at is temporal.py's *source time*. For a file the best
            # evidence of when its content existed is when it was last
            # modified, so that is what it holds — labelled as such, because a
            # modification time does not show when the content was written,
            # checked or verified.
            "source_time_basis": "file-modified",
            "modified_at": _utc_iso(parsed.mtime),
            "encoding": parsed.encoding,
            "newlines": parsed.newlines,
            # Claim spans index this text, not the raw bytes.
            "span_basis": "characters of the decoded text, newlines normalised to \\n",
            "path": parsed.path,
            "sha256": parsed.sha256,
            "size_bytes": parsed.size_bytes,
            # The version a refresh compares the folder against. Older
            # versions stay in the graph as history with this set False.
            "current": True,
        })
        # A file reverted to an earlier content is that earlier version again.
        node.attrs.pop("replaced_at_build", None)
        node.attrs.pop("deleted_at_build", None)


def assume_claims(
    graph: ContextGraph,
    ledger: AssumptionLedger,
    docs: Sequence[Document],
    files: Sequence[ParsedFile],
) -> List[str]:
    """Give every claim lifted from these source versions its assumption.

    Every claim rests on its file still saying it. A refresh retires a
    sentence a file stopped saying by rejecting this assumption with re-read
    evidence (GRAPH.md rule 7), so exactly what was built on that claim falls
    with it and nothing else does (rule 2). Claims are taken in graph order,
    which is document order, so ids and the ledger's history come out the same
    on every build. Returns the claim ids it covered.
    """
    by_source = {doc.id: parsed for doc, parsed in zip(docs, files)}
    covered: List[str] = []
    for node in list(graph.nodes.values()):
        if node.type is not NodeType.CLAIM or node.provenance is None:
            continue
        parsed = by_source.get(node.provenance.source_id)
        if parsed is None or graph.out_edges(node.id, [EdgeType.DEPENDS_ON], live_only=False):
            continue
        assumption = ledger.assume(
            claim_assumption_statement(
                parsed.path,
                parsed.sha256,
                node.label,
                _claim_generation(node.id, node.label, parsed.path),
            ),
            created_by=CLAIM_ASSUMPTION_AUTHOR,
        )
        # A claim must rest on live ground. ``assume`` returns an existing
        # assumption for an identical statement whatever its status, so a
        # collision with a retired claim's statement would silently attach a
        # rejected one: refuse instead.
        if assumption.status is not AssumptionStatus.ACTIVE:
            raise ValueError(
                f"claim {node.id!r} would rest on {assumption.status.value} "
                f"assumption {assumption.id!r}"
            )
        ledger.depends(node.id, assumption.id)
        covered.append(node.id)
    return covered


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
    ingested_at = _utc_iso(datetime.now(timezone.utc).timestamp())
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

    entity_confidence = grade_entities(graph, resolver, entities)
    label_sources(graph, docs, files)
    ledger = AssumptionLedger(graph)
    assume_claims(graph, ledger, docs, files)

    walker = Walker(graph, resolver, hop_budget=hop_budget)
    return FolderBuild(
        root=base,
        graph=graph,
        resolver=resolver,
        walker=walker,
        ledger=ledger,
        report=report,
        documents=docs,
        files=files,
        skipped=skipped,
        entity_confidence=entity_confidence,
        max_total_bytes=max_total_bytes,
        ingested_at=ingested_at,
    )
