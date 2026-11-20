"""Folder ingestion: real files in, located and graded claims out."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path

from contextmesh import cli
from contextmesh.ingest import (
    DEFAULT_MAX_TOTAL_BYTES,
    HIGH,
    LOW,
    MEDIUM,
    IngestError,
    _read_bounded,
    _Unreadable,
    build_folder,
    discover_entities,
    parse_markdown,
    read_folder,
    source_id_for,
)
from contextmesh.model import NodeType

REPO = Path(__file__).resolve().parent.parent

NOTES = """---
title: front matter is metadata
tags: [not, prose]
---
# Index Builder

The **Index Builder** writes shards to `shard_store` every night.
It retries a failed shard three times before it gives up.

- The `shard_store` keeps every shard for thirty days.
- See [Retention Policy](policy.md) for why.

```python
# Code inside a fence never becomes a claim.
shard_store.write(everything)
```

| Column | Means |
|---|---|
| `shard_store` | the table rows are not sentences either |
"""

POLICY = """# Retention Policy

The Retention Policy keeps shards for thirty days because audits look back a month.
The Index Builder reads the Retention Policy before it deletes anything.
"""


def _write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class FolderFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write(self.root, "notes.md", NOTES)
        _write(self.root, "policy.md", POLICY)

    def tearDown(self):
        self._tmp.cleanup()

    def claims(self, built):
        return [n for n in built.graph.nodes.values() if n.type is NodeType.CLAIM]


class MarkdownParsingTest(unittest.TestCase):
    def test_blanking_preserves_every_offset(self):
        parsed = parse_markdown("notes.md", NOTES)
        self.assertEqual(len(parsed.clean), len(parsed.raw))
        for raw_ch, clean_ch in zip(parsed.raw, parsed.clean):
            if raw_ch == "\n":
                self.assertEqual(clean_ch, "\n")

    def test_fences_front_matter_and_tables_are_not_prose(self):
        parsed = parse_markdown("notes.md", NOTES)
        prose = " ".join(parsed.clean[b.start:b.end] for b in parsed.blocks)
        self.assertNotIn("Code inside a fence", prose)
        self.assertNotIn("front matter", prose)
        self.assertNotIn("table rows", prose)

    def test_marked_up_names_are_collected(self):
        parsed = parse_markdown("notes.md", NOTES)
        self.assertIn("Index Builder", parsed.marked)
        self.assertIn("shard_store", parsed.marked)
        self.assertIn("Retention Policy", parsed.marked)

    def test_headings_and_list_items_are_separate_blocks(self):
        parsed = parse_markdown("notes.md", NOTES)
        kinds = [b.kind for b in parsed.blocks]
        self.assertEqual(kinds.count("heading"), 1)
        self.assertEqual(kinds.count("list"), 2)


class EncodingAndTimeTest(unittest.TestCase):
    """UTF-16 by BOM only, and source time that says what it rests on."""

    WINDOWS_NOTE = "# Index Builder\r\n\r\nThe Index Builder writes shards every night.\r\n"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _source(self, built, rel):
        return built.graph.nodes[source_id_for(rel)]

    def test_utf16_with_a_bom_is_read_and_located(self):
        (self.root / "le.md").write_bytes(b"\xff\xfe" + self.WINDOWS_NOTE.encode("utf-16-le"))
        (self.root / "be.md").write_bytes(b"\xfe\xff" + self.WINDOWS_NOTE.encode("utf-16-be"))
        built = build_folder(self.root)
        self.assertEqual(sorted(f.path for f in built.files), ["be.md", "le.md"])
        for rel, encoding in (("le.md", "utf-16-le"), ("be.md", "utf-16-be")):
            source = self._source(built, rel)
            self.assertEqual(source.attrs["encoding"], encoding)
            self.assertEqual(source.attrs["newlines"], "crlf")
        claim = next(
            n for n in built.graph.nodes.values()
            if n.type is NodeType.CLAIM and "writes shards" in n.label
        )
        self.assertEqual(claim.attrs["line_start"], 3)

    def test_utf16_without_a_bom_is_skipped_with_a_fix(self):
        (self.root / "keep.md").write_text(POLICY, encoding="utf-8")
        (self.root / "nobom.md").write_bytes(self.WINDOWS_NOTE.encode("utf-16-le"))
        (self.root / "wide.md").write_bytes(b"\xff\xfe\x00\x00" + "The text.".encode("utf-32-le"))
        _, files, skipped = read_folder(self.root)
        self.assertEqual([f.path for f in files], ["keep.md"])
        reasons = {s.path: s.reason for s in skipped}
        self.assertIn("UTF-16 without a byte-order mark", reasons["nobom.md"])
        self.assertIn("UTF-32", reasons["wide.md"])

    def test_the_cap_counts_bytes_on_disk_not_characters(self):
        data = b"\xff\xfe" + POLICY.encode("utf-16-le")
        (self.root / "a.md").write_bytes(data)
        summary = build_folder(self.root).summary()
        self.assertEqual(summary["bytes_read"], len(data))

    def test_source_time_is_the_labelled_modification_time(self):
        path = _write(self.root, "a.md", POLICY)
        stamp = 1767225600  # 2026-01-01T00:00:00Z
        os.utime(path, (stamp, stamp))
        built = build_folder(self.root)
        attrs = self._source(built, "a.md").attrs
        self.assertEqual(attrs["retrieved_at"], "2026-01-01")
        self.assertEqual(attrs["modified_at"], "2026-01-01T00:00:00Z")
        self.assertEqual(attrs["source_time_basis"], "file-modified")
        self.assertEqual(attrs["encoding"], "utf-8")
        self.assertEqual(attrs["newlines"], "lf")

    def test_as_of_queries_see_only_files_modified_by_then(self):
        from contextmesh.temporal import as_of_graph

        old = _write(self.root, "old.md", "The Index Builder writes shards every night.\n")
        new = _write(self.root, "new.md", "The Index Builder now keys shards on tenant.\n")
        os.utime(old, (1767225600, 1767225600))  # 2026-01-01
        os.utime(new, (1780272000, 1780272000))  # 2026-06-01
        built = build_folder(self.root)
        march = as_of_graph(built.graph, "2026-03-01")
        visible = sorted(
            n.label for n in march.nodes.values() if n.type is NodeType.CLAIM and n.live
        )
        self.assertEqual(visible, ["The Index Builder writes shards every night."])

    def test_ingestion_time_is_reported_but_not_in_the_graph(self):
        path = _write(self.root, "a.md", POLICY)
        os.utime(path, (1767225600, 1767225600))  # so modified_at != now
        built = build_folder(self.root)
        self.assertRegex(built.ingested_at, r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual(built.summary()["ingested_at"], built.ingested_at)
        self.assertNotIn(built.ingested_at, built.graph.to_json())


class PathologicalInputTest(unittest.TestCase):
    """Inputs that once made a parser regex scan quadratically, or forever."""

    CASES = {
        "unclosed emphasis": "**x " * 50_000,
        "unclosed links": "[a](" * 50_000,
        "unclosed wikilinks": "[[a " * 50_000,
        "padded heading": "# a" + " " * 200_000 + "b",
        "many long lines": "\n".join(["**x " * 4_999] * 10),
    }

    def test_each_parses_in_bounded_time(self):
        # In a subprocess with a timeout, so a regression fails instead of
        # hanging the suite.
        script = (
            "import sys,json;from contextmesh.ingest import parse_markdown;"
            "[parse_markdown('x.md', t) for t in json.load(sys.stdin).values()]"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(self.CASES), text=True, cwd=REPO,
            timeout=60, check=True,
        )

    def test_a_padded_heading_is_still_a_heading(self):
        parsed = parse_markdown("x.md", "# Index Builder   ##   \n\nThe Index Builder runs.\n")
        self.assertEqual(parsed.title, "Index Builder")
        self.assertEqual(parsed.blocks[0].kind, "heading")


class ClaimProvenanceTest(FolderFixture):
    def test_every_claim_says_where_it_came_from(self):
        built = build_folder(self.root)
        claims = self.claims(built)
        self.assertTrue(claims)
        for claim in claims:
            attrs = claim.attrs
            self.assertEqual(attrs["extraction"], "verbatim-rule")
            self.assertIs(attrs["verified"], False)
            self.assertIn(attrs["confidence"], (HIGH, MEDIUM, LOW))
            self.assertEqual(claim.provenance.source_id, source_id_for(attrs["path"]))
            self.assertEqual(
                attrs["location"].split(":")[0], attrs["path"]
            )

    def test_span_and_line_point_at_the_original_file(self):
        built = build_folder(self.root)
        raw = (self.root / "policy.md").read_text(encoding="utf-8")
        lines = raw.splitlines()
        sentence = "The Index Builder reads the Retention Policy before it deletes anything."
        claim = next(c for c in self.claims(built) if c.label == sentence)
        start, end = claim.provenance.span
        self.assertEqual(raw[start:end], sentence)
        self.assertEqual(claim.attrs["line_start"], 4)
        self.assertEqual(lines[claim.attrs["line_start"] - 1], sentence)
        self.assertEqual(claim.attrs["location"], "policy.md:4")
        self.assertEqual(claim.attrs["confidence"], HIGH)

    def test_markup_is_gone_from_the_label_but_not_from_the_span(self):
        built = build_folder(self.root)
        raw = (self.root / "notes.md").read_text(encoding="utf-8")
        claim = next(c for c in self.claims(built) if "writes shards" in c.label)
        self.assertEqual(
            claim.label, "The Index Builder writes shards to shard_store every night."
        )
        start, end = claim.provenance.span
        self.assertIn("**Index Builder**", raw[start:end])

    def test_list_items_are_graded_below_paragraph_sentences(self):
        built = build_folder(self.root)
        item = next(c for c in self.claims(built) if "thirty days." in c.label
                    and c.attrs["path"] == "notes.md")
        self.assertEqual(item.attrs["block"], "list")
        self.assertEqual(item.attrs["confidence"], MEDIUM)

    def test_no_relation_edges_are_invented(self):
        built = build_folder(self.root)
        invented = {"contradicts", "supersedes", "supports", "depends_on", "cites"}
        self.assertFalse(
            [e for e in built.graph.edges.values() if e.type.value in invented]
        )

    def test_a_repeated_sentence_keeps_its_first_location(self):
        _write(self.root, "zcopy.md", POLICY)
        built = build_folder(self.root)
        sentence = "The Index Builder reads the Retention Policy before it deletes anything."
        claim = next(c for c in self.claims(built) if c.label == sentence)
        self.assertEqual(claim.provenance.source_id, source_id_for("policy.md"))
        self.assertEqual(claim.attrs["path"], "policy.md")


class EntityDiscoveryTest(FolderFixture):
    def test_marked_and_repeated_names_become_entities(self):
        _, files, _ = read_folder(self.root)
        found = discover_entities(files)
        self.assertEqual(found.get("Index Builder"), HIGH)
        self.assertEqual(found.get("Retention Policy"), HIGH)
        self.assertEqual(found.get("shard_store"), HIGH)

    def test_entity_nodes_carry_their_confidence(self):
        built = build_folder(self.root)
        entities = [n for n in built.graph.nodes.values() if n.type is NodeType.ENTITY]
        self.assertTrue(entities)
        graded = [n for n in entities if "confidence" in n.attrs]
        self.assertTrue(graded)
        for node in graded:
            self.assertIn(node.attrs["confidence"], (HIGH, MEDIUM))

    def test_files_connect_through_a_shared_entity(self):
        built = build_folder(self.root)
        walk = built.walker.ask("What does the Index Builder read before deleting?")
        self.assertTrue(walk.resolved, walk.detail)
        answer = built.graph.node(walk.answer_id)
        self.assertEqual(answer.attrs["path"], "policy.md")


class FolderSafetyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_or_empty_folder_is_refused(self):
        with self.assertRaises(IngestError):
            build_folder(self.root / "nope")
        with self.assertRaises(IngestError):
            build_folder(self.root)

    def test_hidden_vendored_binary_and_oversize_files_are_skipped(self):
        _write(self.root, "keep.md", POLICY)
        _write(self.root, ".git/HEAD.md", POLICY)
        _write(self.root, "node_modules/pkg/README.md", POLICY)
        (self.root / "latin1.txt").write_bytes("caf\xe9 is closed today.".encode("latin-1"))
        _write(self.root, "big.md", "The thing is large. " * 100)
        base, files, skipped = read_folder(self.root, max_bytes=1000)
        self.assertEqual([f.path for f in files], ["keep.md"])
        reasons = {s.path: s.reason for s in skipped}
        self.assertIn("not UTF-8", reasons["latin1.txt"])
        self.assertIn("larger than", reasons["big.md"])
        self.assertNotIn(".git/HEAD.md", reasons)

    @unittest.skipUnless(hasattr(os, "symlink"), "no symlinks")
    def test_symlinks_are_not_followed_out_of_the_folder(self):
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        _write(Path(outside.name), "secret.md", "The secret is kept here today.\n")
        _write(self.root, "keep.md", POLICY)
        try:
            os.symlink(Path(outside.name) / "secret.md", self.root / "link.md")
            os.symlink(outside.name, self.root / "linkdir")
        except OSError:
            self.skipTest("symlinks not permitted")
        _, files, skipped = read_folder(self.root)
        self.assertEqual([f.path for f in files], ["keep.md"])
        self.assertIn("link.md", [s.path for s in skipped])

    @unittest.skipUnless(hasattr(os, "mkfifo"), "no FIFOs")
    def test_a_fifo_named_md_is_skipped_not_read(self):
        # Reading a FIFO blocks until a writer appears, which is never.
        _write(self.root, "keep.md", POLICY)
        os.mkfifo(self.root / "pipe.md")
        _, files, skipped = read_folder(self.root)
        self.assertEqual([f.path for f in files], ["keep.md"])
        self.assertIn(("pipe.md", "not a regular file"),
                      [(s.path, s.reason) for s in skipped])

    def test_the_read_itself_enforces_the_size_limit(self):
        # The walk checks size, but a file can grow before it is read.
        path = _write(self.root, "grows.md", "x" * 50)
        data, _ = _read_bounded(path, 50)
        self.assertEqual(len(data), 50)
        with self.assertRaises(_Unreadable):
            _read_bounded(path, 49)

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "no O_NOFOLLOW")
    def test_the_read_itself_refuses_a_symlink(self):
        # A file swapped for a link after the walk must still not be followed.
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        secret = _write(Path(outside.name), "secret.md", "The secret is here.\n")
        link = self.root / "swapped.md"
        try:
            os.symlink(secret, link)
        except OSError:
            self.skipTest("symlinks not permitted")
        with self.assertRaises(_Unreadable):
            _read_bounded(link, 1000)

    def test_file_limit_is_reported_not_silent(self):
        for i in range(3):
            _write(self.root, f"n{i}.md", POLICY)
        _, files, skipped = read_folder(self.root, max_files=2)
        self.assertEqual(len(files), 2)
        self.assertEqual(len(skipped), 1)
        self.assertIn("limit", skipped[0].reason)

    def test_total_size_cap_admits_what_fits_in_path_order(self):
        # Three 100-byte files and a tiny one; a 250-byte cap admits a and b,
        # skips c, and still admits d, which fits in what is left.
        for name in ("a.md", "b.md", "c.md"):
            _write(self.root, name, ("The Index Builder runs. " * 5)[:99] + "\n")
        _write(self.root, "d.md", "Tiny note here.\n")
        _, files, skipped = read_folder(self.root, max_total_bytes=250)
        self.assertEqual([f.path for f in files], ["a.md", "b.md", "d.md"])
        self.assertEqual([s.path for s in skipped], ["c.md"])
        for item in skipped:
            self.assertEqual(item.reason, "over the 250-byte total limit")

    def test_total_size_cap_counts_bytes_actually_read(self):
        # The walk budgets by size on disk; a file that grew before the read
        # must still not take the total past the cap.
        from contextmesh import ingest

        _write(self.root, "a.md", "The Index Builder runs every night.\n")
        _write(self.root, "b.md", "The Retention Policy keeps shards.\n")
        real = ingest._read_bounded

        def grown(path, max_bytes):
            data, mtime = real(path, max_bytes)
            return (data * 10 if path.name == "a.md" else data), mtime

        ingest._read_bounded = grown
        try:
            _, files, skipped = read_folder(self.root, max_total_bytes=200)
        finally:
            ingest._read_bounded = real
        self.assertEqual([f.path for f in files], ["b.md"])
        self.assertEqual([s.path for s in skipped], ["a.md"])
        self.assertIn("total limit", skipped[0].reason)

    def test_a_cap_smaller_than_the_first_file_is_explained(self):
        _write(self.root, "a.md", POLICY)
        with self.assertRaises(IngestError) as caught:
            build_folder(self.root, max_total_bytes=10)
        self.assertIn("total limit", str(caught.exception))
        with self.assertRaises(IngestError):
            build_folder(self.root, max_total_bytes=0)

    def test_the_default_cap_is_reported(self):
        _write(self.root, "a.md", POLICY)
        summary = build_folder(self.root).summary()
        self.assertEqual(summary["max_total_bytes"], DEFAULT_MAX_TOTAL_BYTES)
        self.assertEqual(summary["bytes_read"], len(POLICY.encode("utf-8")))


class RealRepositoryTest(unittest.TestCase):
    """The repository's own docs, not a synthetic corpus."""

    @classmethod
    def setUpClass(cls):
        cls.built = build_folder(REPO / "docs")

    def test_reads_the_docs_folder(self):
        paths = {f.path for f in self.built.files}
        self.assertIn("ARCHITECTURE.md", paths)
        self.assertGreater(self.built.summary()["by_type"]["claim"], 50)

    def test_answers_with_a_located_claim(self):
        walk = self.built.walker.ask("How are controlled writes committed?")
        self.assertTrue(walk.resolved, walk.detail)
        node = self.built.graph.node(walk.answer_id)
        self.assertTrue(node.attrs["location"].startswith(node.attrs["path"] + ":"))
        raw = (REPO / "docs" / node.attrs["path"]).read_text(encoding="utf-8")
        start, end = node.provenance.span
        self.assertEqual(raw.count("\n", 0, start) + 1, node.attrs["line_start"])
        self.assertTrue(raw[start:end].strip())
        self.assertTrue((REPO / "docs" / node.attrs["path"]).is_file())

    def test_unknown_topic_is_a_named_dead_end(self):
        walk = self.built.walker.ask("What is the refund policy for annual plans?")
        self.assertFalse(walk.resolved)
        self.assertIsNotNone(walk.dead_end)

    def test_build_is_deterministic(self):
        # Two fresh builds: the shared one has been walked, which counts.
        first, second = build_folder(REPO / "docs"), build_folder(REPO / "docs")
        self.assertEqual(first.graph.to_json(), second.graph.to_json())
        self.assertEqual(first.entity_confidence, second.entity_confidence)

    def test_build_does_not_depend_on_the_hash_seed(self):
        script = (
            "import hashlib,sys;from contextmesh.ingest import build_folder;"
            "b=build_folder(sys.argv[1]);"
            "print(hashlib.sha256(b.graph.to_json().encode()).hexdigest(),"
            "sorted(b.entity_confidence.items()))"
        )
        outputs = set()
        for seed in ("0", "4242"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            result = subprocess.run(
                [sys.executable, "-c", script, str(REPO / "docs")],
                cwd=REPO, env=env, capture_output=True, text=True, check=True,
            )
            outputs.add(result.stdout)
        self.assertEqual(len(outputs), 1)


class FolderSessionTest(FolderFixture):
    def test_save_and_reload_keeps_locations_and_confidence(self):
        import argparse

        from contextmesh_mcp.session import (
            Session,
            SessionError,
            add_source_arguments,
            open_session,
        )

        parser = add_source_arguments(argparse.ArgumentParser())
        session = open_session(parser.parse_args(["--folder", str(self.root)]))
        self.assertTrue(session.source.startswith("folder "))
        target = Path(self._tmp.name) / "saved"
        session.save(target)

        restored = Session.load(target)
        before = {n.id: n.attrs for n in session.graph.nodes.values()}
        after = {n.id: n.attrs for n in restored.graph.nodes.values()}
        self.assertEqual(before, after)

        with self.assertRaises(SessionError):
            open_session(parser.parse_args(["--folder", str(self.root), "--rounds", "3"]))
        with self.assertRaises(SessionError):
            open_session(parser.parse_args(["--folder", str(self.root / "missing")]))
        with self.assertRaises(SessionError):
            open_session(parser.parse_args(["--demo", "--max-total-mb", "2"]))
        with self.assertRaises(SessionError):
            open_session(parser.parse_args(["--folder", str(self.root), "--max-total-mb", "0"]))

        capped = open_session(
            parser.parse_args(["--folder", str(self.root), "--max-total-mb", "0.0003"])
        )
        sources = [n for n in capped.graph.nodes.values() if n.type is NodeType.SOURCE]
        self.assertEqual(len(sources), 1)


class FolderCliTest(FolderFixture):
    def run_cli(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(list(argv))
        return code, out.getvalue()

    def test_ingest_json_summary(self):
        code, out = self.run_cli("ingest", str(self.root), "--json")
        self.assertEqual(code, 0)
        summary = json.loads(out)
        self.assertEqual(summary["files_read"], 2)
        self.assertEqual(summary["skipped"], [])
        self.assertGreater(summary["claim_confidence"]["high"], 0)

    def test_ingest_reports_files_left_out_by_the_cap(self):
        code, out = self.run_cli("ingest", str(self.root), "--max-total-mb", "0.0003")
        self.assertEqual(code, 0)
        self.assertIn("over the 300-byte total limit", out)
        self.assertIn("--max-total-mb", out)

    def test_cap_flag_rejects_nonsense(self):
        for bad in ("0", "-1", "lots"):
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
                with unittest.mock.patch("sys.stderr", io.StringIO()):
                    cli.main(["ingest", str(self.root), "--max-total-mb", bad])

    def test_ask_prints_where_the_answer_is(self):
        code, out = self.run_cli(
            "ask", "--folder", str(self.root),
            "What does the Index Builder read before deleting?",
        )
        self.assertEqual(code, 0)
        self.assertIn("policy.md:", out)
        self.assertIn("unverified", out)


if __name__ == "__main__":
    unittest.main()
