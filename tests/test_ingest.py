"""Folder ingestion: real files in, located and graded claims out."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from contextmesh import cli
from contextmesh.ingest import (
    HIGH,
    LOW,
    MEDIUM,
    IngestError,
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

    def test_file_limit_is_reported_not_silent(self):
        for i in range(3):
            _write(self.root, f"n{i}.md", POLICY)
        _, files, skipped = read_folder(self.root, max_files=2)
        self.assertEqual(len(files), 2)
        self.assertEqual(len(skipped), 1)
        self.assertIn("limit", skipped[0].reason)


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
        again = build_folder(REPO / "docs")
        self.assertEqual(sorted(self.built.graph.nodes), sorted(again.graph.nodes))
        self.assertEqual(sorted(self.built.graph.edges), sorted(again.graph.edges))


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
