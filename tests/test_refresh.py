"""Phase 2, slice 2: refreshing a folder graph applies only what changed."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from contextmesh.decisions import DecisionLog
from contextmesh.ingest import build_folder
from contextmesh.model import EdgeType, NodeType
from contextmesh.refresh import RefreshError, refresh_folder
from contextmesh.temporal import as_of_graph

JAN, JUN = 1767225600, 1780272000  # 2026-01-01, 2026-06-01 (UTC)

CACHE = """# Cache

The cache is keyed per tenant.
The cache holds entries for an hour.
"""

INDEX = """# Index

The Index Builder writes shards every night.
The cache is keyed per tenant.
"""


def write(root: Path, name: str, text: str, mtime: int = JAN) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    os.utime(path, (mtime, mtime))
    return path


class RefreshFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "notes"
        write(self.root, "a.md", CACHE)
        write(self.root, "b.md", INDEX)
        self.built = build_folder(self.root)
        self.graph = self.built.graph

    def tearDown(self):
        self._tmp.cleanup()

    def refresh(self, **kwargs):
        return refresh_folder(
            self.graph, self.built.resolver, self.built.ledger, self.root, **kwargs
        )

    def claim(self, path, sentence, *, live=True):
        found = [
            n for n in self.graph.nodes.values()
            if n.type is NodeType.CLAIM and n.attrs.get("path") == path
            and n.label == sentence and n.live == live
        ]
        self.assertEqual(len(found), 1, (path, sentence, live))
        return found[0]

    def live_claims(self, graph=None):
        graph = graph or self.graph
        return sorted(
            (n.attrs["path"], n.label) for n in graph.nodes.values()
            if n.type is NodeType.CLAIM and n.live
        )

    def decide_on(self, claim, title):
        return DecisionLog(self.graph).decide(
            title, "test", source_id=claim.provenance.source_id, supported_by=[claim.id]
        )

    def current_source(self, path):
        found = [
            n for n in self.graph.nodes.values()
            if n.type is NodeType.SOURCE and n.attrs.get("path") == path
            and n.attrs.get("current")
        ]
        self.assertEqual(len(found), 1, path)
        return found[0]


class NoChangeTest(RefreshFixture):
    def test_nothing_changed_touches_nothing(self):
        before, build = self.graph.to_json(), self.graph.build
        report = self.refresh()
        self.assertFalse(report.changed)
        self.assertEqual(report.files["unchanged"], ["a.md", "b.md"])
        self.assertEqual(self.graph.to_json(), before)
        self.assertEqual(self.graph.build, build)

    def test_touching_a_file_without_changing_it_is_no_change(self):
        write(self.root, "a.md", CACHE, mtime=JUN)
        self.assertFalse(self.refresh().changed)


class EditTest(RefreshFixture):
    EDITED = """# Cache

Entries are evicted on every deploy.

The cache holds entries for an hour.
"""

    def edit(self):
        write(self.root, "a.md", self.EDITED, mtime=JUN)

    def test_an_edit_keeps_what_stays_retires_what_went_and_adds_what_is_new(self):
        old_kept = self.claim("a.md", "The cache holds entries for an hour.")
        self.edit()
        report = self.refresh()
        self.assertEqual(report.files["changed"], ["a.md"])
        self.assertEqual(report.files["unchanged"], ["b.md"])
        self.assertEqual(report.claims_kept, 1)
        self.assertEqual(len(report.claims_added), 1)
        self.assertEqual(len(report.claims_retired), 1)
        # The kept claim is the same node, still live.
        self.assertIs(self.claim("a.md", "The cache holds entries for an hour."), old_kept)
        self.claim("a.md", "Entries are evicted on every deploy.")
        self.claim("a.md", "The cache is keyed per tenant.", live=False)

    def test_only_what_stood_on_the_removed_sentence_falls(self):
        gone = self.claim("a.md", "The cache is keyed per tenant.")
        stays = self.claim("a.md", "The cache holds entries for an hour.")
        elsewhere = self.claim("b.md", "The cache is keyed per tenant.")
        on_gone = self.decide_on(gone, "Shard by tenant")
        on_stays = self.decide_on(stays, "Hourly expiry")
        on_elsewhere = self.decide_on(elsewhere, "Index per tenant")
        self.edit()
        report = self.refresh()
        self.assertFalse(self.graph.nodes[on_gone.id].live)
        self.assertTrue(self.graph.nodes[on_stays.id].live)
        # The same sentence in another file is another claim, untouched.
        self.assertTrue(self.graph.nodes[elsewhere.id].live)
        self.assertTrue(self.graph.nodes[on_elsewhere.id].live)
        self.assertEqual(list(report.downstream), [on_gone.id])
        chain = " ".join(report.downstream[on_gone.id])
        self.assertIn("depends_on", chain)
        self.assertIn("derived_from", chain)

    def test_the_retirement_is_explained_by_evidence_naming_both_versions(self):
        gone = self.claim("a.md", "The cache is keyed per tenant.")
        old_sha = self.current_source("a.md").attrs["sha256"]
        self.edit()
        report = self.refresh()
        evidence = self.graph.nodes[report.evidence[0]]
        self.assertIs(evidence.type, NodeType.EVIDENCE)
        self.assertEqual(evidence.attrs["kind"], "file-reread")
        self.assertEqual(evidence.attrs["previous_sha256"], old_sha)
        self.assertEqual(evidence.attrs["sha256"], self.current_source("a.md").attrs["sha256"])
        assumption_id = self.graph.out_edges(gone.id, [EdgeType.DEPENDS_ON], live_only=False)[0].dst
        contradicting = self.graph.in_edges(assumption_id, [EdgeType.CONTRADICTS], live_only=False)
        self.assertEqual([e.src for e in contradicting], [evidence.id])

    def test_versions_are_kept_and_only_the_new_one_is_current(self):
        old = self.current_source("a.md")
        self.edit()
        report = self.refresh()
        new = self.current_source("a.md")
        self.assertNotEqual(old.id, new.id)
        self.assertFalse(old.attrs["current"])
        self.assertEqual(old.attrs["replaced_at_build"], report.build)
        self.assertTrue(old.live and new.live)

    def test_a_kept_claim_says_where_it_now_is_and_keeps_its_first_provenance(self):
        kept = self.claim("a.md", "The cache holds entries for an hour.")
        span, line = kept.provenance.span, kept.attrs["line_start"]
        self.edit()
        self.refresh()
        self.assertEqual(kept.attrs["line_start"], 5)
        self.assertNotEqual(kept.attrs["line_start"], line)
        self.assertEqual(kept.attrs["location"], "a.md:5")
        self.assertEqual(kept.attrs["location_sha256"], self.current_source("a.md").attrs["sha256"])
        self.assertEqual(kept.provenance.span, span)

    def test_new_claims_rest_on_the_new_version(self):
        self.edit()
        self.refresh()
        new = self.claim("a.md", "Entries are evicted on every deploy.")
        self.assertEqual(new.provenance.source_id, self.current_source("a.md").id)
        dep = self.graph.out_edges(new.id, [EdgeType.DEPENDS_ON])
        self.assertEqual(len(dep), 1)
        sha = self.current_source("a.md").attrs["sha256"]
        self.assertIn(f"a.md@{sha[:12]} says:", self.graph.assumptions[dep[0].dst].statement)

    def test_the_live_view_matches_a_fresh_build_of_the_new_folder(self):
        self.edit()
        self.refresh()
        self.assertEqual(self.live_claims(), self.live_claims(build_folder(self.root).graph))

    def test_as_of_queries_see_the_text_of_their_time(self):
        self.edit()
        self.refresh()
        march = as_of_graph(self.graph, "2026-03-01")
        july = as_of_graph(self.graph, "2026-07-01")
        self.assertIn(("a.md", "The cache is keyed per tenant."), self.live_claims(march))
        self.assertNotIn(("a.md", "The cache is keyed per tenant."), self.live_claims(july))
        self.assertNotIn(("a.md", "Entries are evicted on every deploy."), self.live_claims(march))
        self.assertIn(("a.md", "Entries are evicted on every deploy."), self.live_claims(july))


class AddDeleteSkipTest(RefreshFixture):
    def test_an_added_file_is_extracted_with_assumptions(self):
        write(self.root, "c.md", "The Retention Policy keeps shards for thirty days.\n")
        report = self.refresh()
        self.assertEqual(report.files["added"], ["c.md"])
        claim = self.claim("c.md", "The Retention Policy keeps shards for thirty days.")
        self.assertEqual(len(self.graph.out_edges(claim.id, [EdgeType.DEPENDS_ON])), 1)

    def test_a_deleted_file_retires_its_claims_but_not_its_history(self):
        old = self.current_source("b.md")
        (self.root / "b.md").unlink()
        report = self.refresh()
        self.assertEqual(report.files["deleted"], ["b.md"])
        self.assertEqual(len(report.claims_retired), 2)
        self.assertNotIn("b.md", {p for p, _ in self.live_claims()})
        self.assertTrue(old.live)
        self.assertFalse(old.attrs["current"])
        self.assertEqual(old.attrs["deleted_at_build"], report.build)
        entities = [n for n in self.graph.nodes.values() if n.type is NodeType.ENTITY]
        self.assertTrue(all(not n.invalidated for n in entities))

    def test_a_deletion_is_undated_unless_given_a_date(self):
        (self.root / "b.md").unlink()
        report = self.refresh()
        evidence = self.graph.nodes[report.evidence[0]]
        tomb = self.graph.nodes[evidence.provenance.source_id]
        self.assertEqual(evidence.attrs["kind"], "file-missing")
        self.assertEqual(tomb.attrs["retrieved_at"], "undated")

    def test_a_dated_deletion_falls_on_that_date(self):
        (self.root / "b.md").unlink()
        self.refresh(as_of="2026-06-01")
        sentence = ("b.md", "The Index Builder writes shards every night.")
        self.assertIn(sentence, self.live_claims(as_of_graph(self.graph, "2026-05-01")))
        self.assertNotIn(sentence, self.live_claims(as_of_graph(self.graph, "2026-07-01")))

    def test_a_file_skipped_this_time_is_left_as_it_was(self):
        write(self.root, "b.md", INDEX + "x" * 2000, mtime=JUN)
        report = refresh_folder(
            self.graph, self.built.resolver, self.built.ledger, self.root, max_bytes=1000
        )
        self.assertEqual(report.files["left_as_is"], ["b.md"])
        self.assertEqual(report.claims_retired, [])
        self.claim("b.md", "The Index Builder writes shards every night.")

    def test_an_emptied_folder_retires_everything(self):
        for name in ("a.md", "b.md"):
            (self.root / name).unlink()
        report = self.refresh()
        self.assertEqual(report.files["deleted"], ["a.md", "b.md"])
        self.assertEqual(self.live_claims(), [])


class RestoreTest(RefreshFixture):
    def test_a_returning_sentence_is_a_new_claim_that_supersedes_the_dead_one(self):
        first = self.claim("a.md", "The cache is keyed per tenant.")
        write(self.root, "a.md", "# Cache\n\nThe cache holds entries for an hour.\n", mtime=JUN)
        self.refresh()
        write(self.root, "a.md", CACHE, mtime=JUN + 86400)
        report = self.refresh()
        second = self.claim("a.md", "The cache is keyed per tenant.")
        self.assertNotEqual(first.id, second.id)
        self.assertTrue(first.invalidated)
        self.assertEqual(report.claims_restored, [second.id])
        supersedes = self.graph.out_edges(second.id, [EdgeType.SUPERSEDES], live_only=False)
        self.assertEqual([e.dst for e in supersedes], [first.id])

    def test_a_sentence_returning_by_revert_rests_on_new_live_ground(self):
        # Reverting a file makes its earlier version current again, so the
        # returning sentence's version is the one its dead predecessor was
        # read from. It must still get its own, active assumption.
        first = self.claim("a.md", "The cache is keyed per tenant.")
        write(self.root, "a.md", "# Cache\n\nThe cache holds entries for an hour.\n", mtime=JUN)
        self.refresh()
        write(self.root, "a.md", CACHE)
        self.refresh()
        second = self.claim("a.md", "The cache is keyed per tenant.")
        self.assertNotEqual(first.id, second.id)
        deps = self.graph.out_edges(second.id, [EdgeType.DEPENDS_ON])
        self.assertEqual(len(deps), 1)
        assumption = self.graph.assumptions[deps[0].dst]
        self.assertEqual(assumption.status.value, "active")
        self.assertIn("(returned, generation 1)", assumption.statement)

    def test_reverting_a_file_makes_its_earlier_version_current_again(self):
        original = self.current_source("a.md")
        write(self.root, "a.md", "# Cache\n\nThe cache holds entries for an hour.\n", mtime=JUN)
        self.refresh()
        write(self.root, "a.md", CACHE)
        self.refresh()
        self.assertIs(self.current_source("a.md"), original)
        self.assertNotIn("replaced_at_build", original.attrs)


class EntityRediscoveryTest(unittest.TestCase):
    """Phase 2, slice 3: a name that starts to qualify reaches unchanged files."""

    # "Retention Window" is capitalised mid-sentence in one file only, so a
    # full build does not make it an entity; it needs a second file.
    POLICY = "# Policy\n\nShards follow the Retention Window set by audit.\n"
    NEWS = "# News\n\nAudit shortened the Retention Window this quarter.\n"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "notes"
        write(self.root, "policy.md", self.POLICY)
        write(self.root, "cache.md", CACHE)
        self.built = build_folder(self.root)
        self.graph = self.built.graph

    def tearDown(self):
        self._tmp.cleanup()

    def refresh(self, **kwargs):
        return refresh_folder(
            self.graph, self.built.resolver, self.built.ledger, self.root, **kwargs
        )

    def entity(self, graph, label):
        found = [
            n for n in graph.nodes.values()
            if n.type is NodeType.ENTITY and n.label.lower() == label.lower()
        ]
        return found[0] if found else None

    def mentions(self, graph):
        """(path, claim sentence, entity) for every live claim's mentions."""
        out = set()
        for node in graph.nodes.values():
            if node.type is NodeType.CLAIM and node.live:
                for edge in graph.out_edges(node.id, [EdgeType.MENTIONS]):
                    out.add((node.attrs["path"], node.label, graph.nodes[edge.dst].label))
        return out

    def test_a_newly_qualifying_name_is_linked_into_unchanged_files(self):
        self.assertIsNone(self.entity(self.graph, "Retention Window"))
        write(self.root, "news.md", self.NEWS)
        report = self.refresh()
        entity = self.entity(self.graph, "Retention Window")
        self.assertIsNotNone(entity)
        self.assertIn(entity.id, report.entities_added)
        self.assertGreater(report.entity_links_added, 0)
        self.assertTrue(entity.live)
        self.assertIsNotNone(entity.embedding)

        policy_claim = next(
            n for n in self.graph.nodes.values()
            if n.type is NodeType.CLAIM and n.attrs.get("path") == "policy.md"
        )
        mentioned = [e.dst for e in self.graph.out_edges(policy_claim.id, [EdgeType.MENTIONS])]
        self.assertIn(entity.id, mentioned)
        policy_source = next(
            n for n in self.graph.nodes.values()
            if n.type is NodeType.SOURCE and n.attrs.get("path") == "policy.md"
        )
        self.assertIn(
            entity.id,
            [e.dst for e in self.graph.out_edges(policy_source.id, [EdgeType.MENTIONS])],
        )
        self.assertIn(
            policy_source.id,
            [e.dst for e in self.graph.out_edges(entity.id, [EdgeType.DERIVED_FROM])],
        )

    def test_the_unchanged_file_now_answers_through_the_new_entity(self):
        write(self.root, "news.md", self.NEWS)
        self.refresh()
        walker = self.built.walker
        walk = walker.ask("What do shards follow under the Retention Window?")
        self.assertTrue(walk.resolved, walk.detail)

    def test_mentions_match_a_fresh_build(self):
        write(self.root, "news.md", self.NEWS)
        self.refresh()
        fresh = build_folder(self.root).graph
        self.assertEqual(self.mentions(self.graph), self.mentions(fresh))

    def edge_weights_at(self, graph, entity_label):
        """(src, type, dst) -> weight for every live edge touching an entity,
        with nodes named by path/label so two graphs can be compared."""
        def name(node):
            if node.type is NodeType.SOURCE:
                return ("source", node.attrs["path"])
            if node.type is NodeType.CLAIM:
                return ("claim", node.attrs["path"], node.label)
            return (node.type.value, node.label.lower())

        entity = self.entity(graph, entity_label)
        out = {}
        for edge in graph.edges.values():
            if edge.live and entity.id in (edge.src, edge.dst):
                key = (name(graph.nodes[edge.src]), edge.type.value, name(graph.nodes[edge.dst]))
                out[key] = edge.weight
        return out

    def test_edges_to_a_new_entity_weigh_what_a_fresh_build_gives_them(self):
        # LINK weights an edge by how often it is mentioned; relinking must
        # repeat those calls, not de-duplicate them, or walks over a refreshed
        # graph would cost differently from walks over a fresh build.
        write(self.root, "news.md", self.NEWS)
        self.refresh()
        fresh = build_folder(self.root).graph
        self.assertEqual(
            self.edge_weights_at(self.graph, "Retention Window"),
            self.edge_weights_at(fresh, "Retention Window"),
        )

    def test_relinking_adds_no_weight_to_edges_that_already_existed(self):
        weights = {eid: e.weight for eid, e in self.graph.edges.items()}
        write(self.root, "news.md", self.NEWS)
        self.refresh()
        for eid, weight in weights.items():
            self.assertEqual(self.graph.edges[eid].weight, weight, eid)

    def test_a_deletion_only_refresh_still_registers_and_links(self):
        # With room for few entities, deleting the file whose names ranked
        # highest lets a lower-ranked name in, with nothing to re-extract.
        for name in ("a.md", "b.md", "c.md"):
            write(self.root, name, "")
        (self.root / "cache.md").unlink()
        heavy = "# Heavy\n\n" + "The Index Builder and the Shard Store run the Rebuild.\n" * 3
        write(self.root, "heavy.md", heavy)
        write(self.root, "also.md", "The Index Builder and the Shard Store run the Rebuild.\n")
        write(self.root, "news.md", self.NEWS)
        built = build_folder(self.root, max_entities=3)
        graph = built.graph
        self.assertIsNone(self.entity(graph, "Retention Window"))
        for name in ("heavy.md", "also.md"):
            (self.root / name).unlink()
        report = refresh_folder(
            graph, built.resolver, built.ledger, self.root, max_entities=3
        )
        self.assertEqual(report.files["changed"] + report.files["added"], [])
        entity = self.entity(graph, "Retention Window")
        self.assertIsNotNone(entity)
        self.assertTrue(entity.live)
        self.assertGreater(report.entity_links_added, 0)

    def test_a_dry_run_names_the_entities_it_would_add(self):
        write(self.root, "news.md", self.NEWS)
        dry = self.refresh(dry_run=True)
        real = self.refresh()
        self.assertEqual(dry.entities_added, real.entities_added)


class DryRunTest(RefreshFixture):
    def test_a_dry_run_changes_nothing_and_predicts_the_real_run(self):
        self.decide_on(self.claim("a.md", "The cache is keyed per tenant."), "Shard by tenant")
        write(self.root, "a.md", EditTest.EDITED, mtime=JUN)
        (self.root / "b.md").unlink()
        write(self.root, "c.md", "The Retention Policy keeps shards for thirty days.\n")
        before = self.graph.to_json()
        dry = self.refresh(dry_run=True)
        self.assertEqual(self.graph.to_json(), before)
        real = self.refresh()
        for key in ("files", "claims_retired", "claims_restored"):
            self.assertEqual(getattr(dry, key), getattr(real, key), key)
        self.assertEqual(sorted(dry.claims_added), sorted(real.claims_added))
        self.assertEqual(sorted(dry.downstream), sorted(real.downstream))
        self.assertEqual(dry.claims_kept, real.claims_kept)


class DeterminismTest(unittest.TestCase):
    def run_sequence(self, base: Path) -> str:
        root = base / "notes"
        write(root, "a.md", CACHE)
        write(root, "b.md", INDEX)
        built = build_folder(root)
        steps = [
            lambda: write(root, "a.md", EditTest.EDITED, mtime=JUN),
            lambda: (root / "b.md").unlink(),
            lambda: write(root, "a.md", CACHE, mtime=JUN + 86400),
        ]
        for step in steps:
            step()
            refresh_folder(built.graph, built.resolver, built.ledger, root, as_of="2026-06-02")
        return built.graph.to_json()

    def test_the_same_edits_give_the_same_graph(self):
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            self.assertEqual(self.run_sequence(Path(one)), self.run_sequence(Path(two)))


class RefusalTest(unittest.TestCase):
    def test_a_graph_not_built_from_a_folder_is_refused(self):
        from contextmesh.demo import run

        result = run(rounds=1)
        with tempfile.TemporaryDirectory() as tmp:
            write(Path(tmp), "a.md", CACHE)
            with self.assertRaises(RefreshError):
                refresh_folder(result.graph, result.resolver, result.ledger, tmp)

    def test_a_graph_from_before_versioned_sources_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(Path(tmp), "a.md", CACHE)
            built = build_folder(tmp)
            for node in built.graph.nodes.values():
                if node.type is NodeType.SOURCE:
                    node.attrs.pop("current", None)
            with self.assertRaises(RefreshError) as caught:
                refresh_folder(built.graph, built.resolver, built.ledger, tmp)
            self.assertIn("rebuild", str(caught.exception))

    def test_a_bad_as_of_is_refused_before_anything_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(Path(tmp), "a.md", CACHE)
            built = build_folder(tmp)
            before = built.graph.to_json()
            with self.assertRaises(ValueError):
                refresh_folder(built.graph, built.resolver, built.ledger, tmp, as_of="soon")
            self.assertEqual(built.graph.to_json(), before)


class SessionRefreshTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.notes = self.base / "notes"
        write(self.notes, "a.md", CACHE)
        write(self.notes, "b.md", INDEX)
        self.session = self.base / "session"
        from contextmesh_mcp.session import main

        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--folder", str(self.notes), "--save", str(self.session)]), 0)

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, *argv):
        from contextmesh_mcp.session import main

        out = io.StringIO()
        with redirect_stdout(out):
            code = main(list(argv))
        return code, (json.loads(out.getvalue()) if code == 0 else None)

    def generation(self):
        from contextmesh_mcp.session import Session

        return Session.load(self.session).generation

    def test_refresh_saves_the_next_generation_only_when_something_changed(self):
        start = self.generation()
        code, report = self.run_main("--session", str(self.session), "--refresh", str(self.notes))
        self.assertEqual((code, report["changed"], report["saved"]), (0, False, None))
        self.assertEqual(self.generation(), start)

        write(self.notes, "a.md", EditTest.EDITED, mtime=JUN)
        code, report = self.run_main("--session", str(self.session), "--refresh", str(self.notes))
        self.assertEqual(code, 0)
        self.assertTrue(report["changed"])
        self.assertEqual(report["generation"], start + 1)
        self.assertEqual(self.generation(), start + 1)

        # Reloaded, the session knows it is up to date.
        code, report = self.run_main("--session", str(self.session), "--refresh", str(self.notes))
        self.assertFalse(report["changed"])

    def test_a_dry_run_saves_nothing(self):
        start = self.generation()
        write(self.notes, "a.md", EditTest.EDITED, mtime=JUN)
        code, report = self.run_main(
            "--session", str(self.session), "--refresh", str(self.notes), "--dry-run"
        )
        self.assertEqual((code, report["changed"], report["saved"]), (0, True, None))
        self.assertEqual(self.generation(), start)

    def test_misuse_is_refused(self):
        from unittest import mock

        from contextmesh_mcp.session import main

        bad = [
            ["--folder", str(self.notes), "--refresh", str(self.notes)],
            ["--folder", str(self.notes), "--dry-run"],
            ["--session", str(self.session), "--refresh", str(self.notes), "--as-of", "soon"],
            ["--session", str(self.session), "--refresh", str(self.notes),
             "--save", str(self.base / "elsewhere")],
        ]
        start = self.generation()
        for argv in bad:
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", err):
                self.assertEqual(main(argv), 2, argv)
            self.assertIn("contextmesh session:", err.getvalue())
        self.assertEqual(self.generation(), start)

if __name__ == "__main__":
    unittest.main()
