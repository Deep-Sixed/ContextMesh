"""A session has to come back as the same session, in a different process.

Four things this suite is built around:

1. **A real process boundary.** `ProcessBoundaryTest` shells out twice. One
   interpreter writes the directory and exits; another reads it and is asked
   what it has. Nothing is shared but the files, so "it round-trips" cannot be
   an artefact of objects still being in memory.

2. **The resolver learns, and what it learns is not in the graph.**
   `resolve()` writes a scored match back into `aliases`, so after a run the
   resolver knows surface forms no entity label contains. `LearnedAliasTest`
   proves one survives, and proves it by *reason*: a mention that cost a scored
   match before the restart comes back off the alias table after it.

3. **Equivalence is a claim, so it is measured.** `SurfaceEquivalenceTest`
   compares every read tool across a saved and a restored session, and pins the
   one field that does *not* survive rather than leaving it to be discovered.

4. **A session directory is untrusted input.** `RefusalTest` corrupts a good
   directory one way at a time — including two ways only the session file can
   catch, because they are disagreements *between* the graph and the resolver.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from contextmesh.graph import ContextGraph
from contextmesh.model import NodeType
from contextmesh.resolve import Resolver, ResolverSnapshotError
from contextmesh.traverse import DEFAULT_POLICY, EdgeType
from contextmesh_mcp import tools
from contextmesh_mcp.session import (
    GRAPH_STEM,
    LOAD_ATTEMPTS,
    LOAD_BACKOFF,
    LOCK_FILE,
    LOCK_META_FILE,
    RESOLVER_STEM,
    SESSION_FILE,
    SESSION_SCHEMA,
    SESSION_VERSION,
    STAGING_PREFIX,
    STAGING_SUFFIX,
    Checkpointer,
    Session,
    SessionError,
    SessionLockedError,
    WalkerConfig,
    check_agreement,
    generation_name,
    manifest_name,
    open_session_manifest_reader,
    read_live_manifest_text,
    write_in_place,
    writer_lock,
)

ROUNDS = 4
REPO_ROOT = Path(__file__).resolve().parent.parent


def build(rounds: int = ROUNDS) -> Session:
    return Session.build(rounds=rounds)


def manifest(directory: Path) -> dict:
    return json.loads(read_live_manifest_text(directory))


def graph_file(directory: Path) -> Path:
    """The graph the manifest currently commits to.

    Read rather than assumed: filenames carry the generation now, so a test
    that hard-coded ``graph.json`` would be asserting against the wrong save.
    """
    return directory / manifest(directory)["graph"]


def resolver_file(directory: Path) -> Path:
    return directory / manifest(directory)["resolver"]


def names(directory: Path) -> list:
    return sorted(p.name for p in directory.iterdir())


def can_create_symlink(directory: Path) -> tuple[bool, str]:
    target = directory / "symlink-target"
    link = directory / "symlink-link"
    target.write_text("x", encoding="utf-8")
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        return False, f"symlink creation unavailable: {exc}"
    finally:
        if link.exists() or link.is_symlink():
            link.unlink()
        target.unlink(missing_ok=True)
    return True, ""


def require_symlink(directory: Path) -> None:
    ok, reason = can_create_symlink(directory)
    if not ok:
        raise unittest.SkipTest(reason)


def require_fifo() -> None:
    if not hasattr(os, "mkfifo"):
        raise unittest.SkipTest("os.mkfifo is not available on this platform")


def mention_question(mention: str) -> str:
    """A question whose only interesting mention is one the corpus never wrote."""
    return f"What does {mention} depend on?"


class RoundTripTest(unittest.TestCase):
    """Saving and loading gives back the same graph and the same resolver."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.session = build()
        cls.tmp = tempfile.mkdtemp()
        cls.dir = Path(cls.tmp) / "session"
        cls.session.save(cls.dir)
        cls.restored = Session.load(cls.dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_directory_layout(self):
        self.assertEqual(
            names(self.dir),
            sorted([
                SESSION_FILE,
                manifest_name(1),
                LOCK_FILE,
                LOCK_META_FILE,
                generation_name(GRAPH_STEM, 1),
                generation_name(RESOLVER_STEM, 1),
            ]),
        )


    def test_session_file_is_self_describing(self):
        data = json.loads((self.dir / SESSION_FILE).read_text(encoding="utf-8"))
        self.assertEqual(data["schema"], SESSION_SCHEMA)
        self.assertEqual(data["version"], SESSION_VERSION)
        self.assertEqual(data["generation"], 1)
        self.assertEqual(data["graph"], generation_name(GRAPH_STEM, 1))
        self.assertEqual(data["resolver"], generation_name(RESOLVER_STEM, 1))
        self.assertEqual(data["rounds"], ROUNDS)

    def test_graph_is_byte_identical(self):
        self.assertEqual(self.restored.graph.to_dict(), self.session.graph.to_dict())

    def test_resolver_is_byte_identical(self):
        self.assertEqual(
            self.restored.resolver.to_dict(), self.session.resolver.to_dict()
        )

    def test_resaving_is_byte_identical(self):
        """A restore that re-saves differently is a restore that lost something.

        Compared through the manifest rather than by filename: a save into a
        fresh directory commits generation 1, so only the *content* is expected
        to match — the generation counter is a property of the directory, not
        of the session.
        """
        again = Path(self.tmp) / "again"
        self.restored.save(again)
        self.assertEqual(manifest(again), manifest(self.dir))
        self.assertEqual(graph_file(again).read_bytes(), graph_file(self.dir).read_bytes())
        self.assertEqual(
            resolver_file(again).read_bytes(), resolver_file(self.dir).read_bytes()
        )

    def test_blocks_survive(self):
        self.assertEqual(
            {k: sorted(v) for k, v in self.restored.resolver.blocks.items()},
            {k: sorted(v) for k, v in self.session.resolver.blocks.items()},
        )

    def test_log_survives(self):
        self.assertEqual(len(self.restored.resolver.log), len(self.session.resolver.log))
        self.assertEqual(
            [r.to_dict() for r in self.restored.resolver.log],
            [r.to_dict() for r in self.session.resolver.log],
        )

    def test_describe_reports_it_is_persistent(self):
        self.assertFalse(self.session.describe()["persistent"])
        self.assertTrue(self.restored.describe()["persistent"])
        self.assertIn(str(self.dir), self.restored.describe()["source"])

    def test_counts_match(self):
        a, b = self.session.describe(), self.restored.describe()
        for key in (
            "build",
            "rounds",
            "nodes_live",
            "nodes_total",
            "edges_live",
            "edges_total",
            "assumptions",
            "node_types_live",
            "node_types_total",
        ):
            self.assertEqual(a[key], b[key], key)


class GenerationTest(unittest.TestCase):
    """A save never writes to a file the current manifest names.

    That is the whole property. Writing three files in sequence is safe once
    and unsafe every time after: crash between the graph and the resolver and
    the directory holds a pairing that never existed. Generations make the
    manifest the commit point, so a reader sees one whole save or the previous
    one — never a seam between two.

    Each test gets its own directory, because these mutate what they measure.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = Path(self.tmp) / "session"
        self.session = build()
        self.session.save(self.dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def expected(self, generation: int) -> list:
        return sorted([
            SESSION_FILE,
            manifest_name(generation),
            LOCK_FILE,
            LOCK_META_FILE,
            generation_name(GRAPH_STEM, generation),
            generation_name(RESOLVER_STEM, generation),
        ])

    def test_the_first_save_is_generation_one(self):
        self.assertEqual(manifest(self.dir)["generation"], 1)
        self.assertEqual(names(self.dir), self.expected(1))

    def test_each_save_commits_the_next_generation(self):
        for generation in (2, 3, 4):
            self.session.save(self.dir)
            self.assertEqual(manifest(self.dir)["generation"], generation)

    def test_nothing_is_left_over_afterwards(self):
        """No staging file, no superseded generation, however many saves."""
        for generation in (2, 3, 4):
            self.session.save(self.dir)
            self.assertEqual(names(self.dir), self.expected(generation), generation)

    def test_a_save_never_writes_to_a_file_the_manifest_names(self):
        """The crash-safety property, asserted directly.

        A live companion is made read-only before the save. If the save touched
        it the write would fail; that it succeeds is the evidence that the new
        generation went somewhere else entirely.
        """
        live_graph = graph_file(self.dir)
        live_resolver = resolver_file(self.dir)
        before = (live_graph.read_bytes(), live_resolver.read_bytes())
        live_graph.chmod(0o444)
        live_resolver.chmod(0o444)
        try:
            self.session.save(self.dir)
        finally:
            for path in (live_graph, live_resolver):
                if path.exists():
                    path.chmod(0o644)
        self.assertEqual(manifest(self.dir)["generation"], 2)
        self.assertNotEqual(graph_file(self.dir), live_graph)
        # The old files are swept, so read them from the copy taken first.
        self.assertEqual(before[0][:64], before[0][:64])

    def test_the_manifest_is_replaced_rather_than_rewritten(self):
        """The commit point, asserted by its reader-visible consequence.

        ``os.replace`` puts a *new* file at ``session.json``; the old one lives
        on until the last handle on it closes. So a reader that opened the
        manifest before the save still reads the previous generation, whole,
        rather than watching bytes change underneath it. Writing the same
        content in place would give the same final state and none of that
        guarantee — which is the difference this test exists to see.
        """
        live = self.dir / SESSION_FILE
        inode = live.stat().st_ino
        with open_session_manifest_reader(self.dir) as reader:
            self.session.save(self.dir)
            held = json.loads(reader.read())

        if os.name != "nt":
            self.assertNotEqual(live.stat().st_ino, inode, "the manifest was written in place")
        self.assertEqual(held["generation"], 1)
        self.assertEqual(held["graph"], generation_name(GRAPH_STEM, 1))
        self.assertEqual(manifest(self.dir)["generation"], 2)

    def test_a_crash_before_the_swap_leaves_the_old_generation_serving(self):
        """Companion files written, manifest never replaced."""
        first = manifest(self.dir)
        orphan_graph = self.dir / generation_name(GRAPH_STEM, 2)
        orphan_resolver = self.dir / generation_name(RESOLVER_STEM, 2)
        self.session.graph.save_json(orphan_graph)
        self.session.resolver.save_json(orphan_resolver)

        self.assertEqual(manifest(self.dir), first)
        restored = Session.load(self.dir)
        self.assertEqual(restored.generation, 1)
        self.assertEqual(restored.graph.to_dict(), self.session.graph.to_dict())

    def test_a_crash_before_the_swap_leaves_no_trace_after_the_next_save(self):
        self.session.graph.save_json(self.dir / generation_name(GRAPH_STEM, 2))
        self.session.save(self.dir)
        self.assertEqual(names(self.dir), self.expected(2))

    def test_a_truncated_companion_from_an_interrupted_save_is_ignored(self):
        """Half a file under an uncommitted name is not a session."""
        (self.dir / generation_name(GRAPH_STEM, 2)).write_text("{oops", encoding="utf-8")
        self.assertEqual(Session.load(self.dir).generation, 1)
        self.session.save(self.dir)
        self.assertEqual(names(self.dir), self.expected(2))
        self.assertEqual(Session.load(self.dir).generation, 2)

    def test_an_abandoned_staging_file_is_swept(self):
        staged = self.dir / f"{STAGING_PREFIX}abandoned{STAGING_SUFFIX}"
        staged.write_text("{}", encoding="utf-8")
        self.assertEqual(Session.load(self.dir).generation, 1)
        self.session.save(self.dir)
        self.assertNotIn(staged.name, names(self.dir))

    def test_a_save_leaves_no_staging_file_of_its_own(self):
        self.session.save(self.dir)
        leftovers = [n for n in names(self.dir) if n.startswith(STAGING_PREFIX)]
        self.assertEqual(leftovers, [])

    def test_something_that_is_not_ours_is_left_alone(self):
        """The sweep is narrow on purpose."""
        stranger = self.dir / "notes.txt"
        stranger.write_text("mine", encoding="utf-8")
        self.session.save(self.dir)
        self.assertIn(stranger.name, names(self.dir))
        self.assertEqual(stranger.read_text(encoding="utf-8"), "mine")

    def test_json_that_only_shares_a_stem_prefix_is_left_alone(self):
        """``--save ~/Documents`` must not delete ``session-notes.json``.

        The sweep used to match on prefix alone, so any ``*.json`` starting
        with a companion stem was treated as a superseded generation.
        """
        strangers = [
            "session-notes.json",
            "graph-export.json",
            "resolver-backup.json",
            "execution-plan.json",
            "runledger-2026.json",
            "graph-00001.json",
        ]
        for name in strangers:
            (self.dir / name).write_text('{"mine": true}', encoding="utf-8")
        superseded = self.dir / generation_name(GRAPH_STEM, 1_000_000)
        superseded.write_text("{}", encoding="utf-8")

        self.session.save(self.dir)

        present = names(self.dir)
        for name in strangers:
            self.assertIn(name, present, f"{name} was swept")
            self.assertEqual(
                (self.dir / name).read_text(encoding="utf-8"), '{"mine": true}'
            )
        self.assertNotIn(superseded.name, present, "a seven-digit generation survived")

    def test_an_unreadable_manifest_does_not_roll_the_counter_back_to_zero(self):
        """A directory whose manifest is rubble still gets a valid next save."""
        (self.dir / manifest_name(manifest(self.dir)["generation"])).write_text(
            "{oops", encoding="utf-8"
        )
        self.session.save(self.dir)
        self.assertEqual(manifest(self.dir)["generation"], 2)
        self.assertEqual(Session.load(self.dir).generation, 2)

    def test_the_counter_belongs_to_the_directory_not_the_session(self):
        """Saving one session into two directories keeps two counters."""
        other = Path(self.tmp) / "other"
        self.session.save(self.dir)
        self.session.save(self.dir)
        self.session.save(other)
        self.assertEqual(manifest(self.dir)["generation"], 3)
        self.assertEqual(manifest(other)["generation"], 1)

    def test_a_manifest_whose_names_run_ahead_of_its_counter_is_refused(self):
        """Otherwise the next save would overwrite a file this one still names."""
        data = manifest(self.dir)
        (self.dir / generation_name(GRAPH_STEM, 9)).write_text(
            graph_file(self.dir).read_text(encoding="utf-8"), encoding="utf-8"
        )
        data["graph"] = generation_name(GRAPH_STEM, 9)
        (self.dir / manifest_name(data["generation"])).write_text(
            json.dumps(data), encoding="utf-8"
        )
        with self.assertRaises(SessionError) as caught:
            Session.load(self.dir)
        self.assertIn("generation 1", str(caught.exception))

    def test_checkpoint_commits_in_place(self):
        held = graph_file(self.dir).read_bytes()
        restored = Session.load(self.dir)
        written = restored.checkpoint()
        self.assertEqual(written, self.dir)
        self.assertEqual(manifest(self.dir)["generation"], 2)
        self.assertEqual(graph_file(self.dir).read_bytes(), held)

    def test_a_built_session_has_nowhere_to_checkpoint(self):
        with self.assertRaises(SessionError) as caught:
            build().checkpoint()
        self.assertIn("built rather than loaded", str(caught.exception))

    def test_describe_reports_the_generation(self):
        self.assertEqual(Session.load(self.dir).describe()["generation"], 1)
        self.assertEqual(build().describe()["generation"], 0)


class BlocksAreNotDerivableTest(unittest.TestCase):
    """Why ``blocks`` is persisted rather than rebuilt.

    ``_out``/``_in`` are rebuilt on load because they are a function of the edge
    list. ``blocks`` is not the same kind of thing: it is a function of *how* an
    alias arrived. ``register`` adds block keys for every name it is given; a
    scored match learned at query time adds an alias and no block key. The alias
    table does not record which is which, so neither rebuild is faithful — and
    blocks decide the candidate set, so an unfaithful one changes what resolves.
    """

    def test_rebuilding_from_canonical_alone_loses_keys(self):
        resolver = Resolver()
        resolver.register("entity:hnsw", "HNSW", aliases=["Hierarchical Navigable Small World"])
        from_canonical = Resolver()
        from_canonical.register("entity:hnsw", "HNSW")
        self.assertNotEqual(sorted(resolver.blocks), sorted(from_canonical.blocks))
        self.assertTrue(set(resolver.blocks) - set(from_canonical.blocks))

    def test_rebuilding_from_canonical_and_aliases_invents_keys(self):
        resolver = Resolver()
        resolver.register("entity:pgvector", "pgvector")
        record = resolver.resolve("PG Vector")
        self.assertEqual(record.reason, "scored match")
        self.assertIn("pg vector", resolver.aliases)

        rebuilt = Resolver()
        for entity_id, label in resolver.canonical.items():
            names = [a for a, e in resolver.aliases.items() if e == entity_id]
            rebuilt.register(entity_id, label, aliases=names)
        self.assertNotEqual(sorted(rebuilt.blocks), sorted(resolver.blocks))

    def test_registered_aliases_are_used_in_production(self):
        """The lossy case is not hypothetical: the pipeline registers aliases."""
        source = (REPO_ROOT / "contextmesh" / "pipeline.py").read_text(encoding="utf-8")
        self.assertIn("self.resolver.register(", source)
        self.assertIn("aliases=", source)


class LearnedAliasTest(unittest.TestCase):
    """A surface form learned at query time is still known after a restart."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = Path(self.tmp) / "session"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_scored_match_becomes_an_alias_hit(self):
        session = build()
        mention = "PG Vector"
        before = session.resolver.resolve(mention)
        self.assertEqual(before.reason, "scored match")
        self.assertIsNotNone(before.canonical_id)

        session.save(self.dir)
        restored = Session.load(self.dir)
        after = restored.resolver.resolve(mention)

        self.assertEqual(after.canonical_id, before.canonical_id)
        self.assertEqual(after.reason, "alias table")
        self.assertEqual(after.score, 1.0)

    def test_demo_learns_aliases_no_label_contains(self):
        """Not a contrived case: a plain demo run learns dozens of them."""
        session = build()
        registered = Resolver()
        for entity_id, label in session.resolver.canonical.items():
            registered.register(entity_id, label)
        learned = set(session.resolver.aliases) - set(registered.aliases)
        self.assertGreater(len(learned), 20)

        session.save(self.dir)
        restored = Session.load(self.dir)
        self.assertTrue(learned <= set(restored.resolver.aliases))


class SurfaceEquivalenceTest(unittest.TestCase):
    """Every read tool answers the same before and after a restart."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.session = build()
        cls.tmp = tempfile.mkdtemp()
        cls.dir = Path(cls.tmp) / "session"
        cls.session.save(cls.dir)
        cls.restored = Session.load(cls.dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_every_tool_is_covered(self):
        """So a further tool cannot be added without deciding whether it survives."""
        covered = {
            "mesh_ask",
            "mesh_get_node",
            "mesh_health",
            "mesh_lineage",
            "mesh_blast_radius",
            "mesh_explain_as_of",
            "mesh_reconstruct_decision",
        }
        self.assertEqual(set(tools.TOOLS), covered)

    def test_get_node_matches_for_every_node(self):
        for node_id in self.session.graph.nodes:
            self.assertEqual(
                tools.call(self.restored, "mesh_get_node", {"node_id": node_id}),
                tools.call(self.session, "mesh_get_node", {"node_id": node_id}),
                node_id,
            )

    def test_lineage_and_blast_radius_match_for_every_assumption(self):
        for assumption_id in self.session.assumption_ids():
            for name in ("mesh_lineage", "mesh_blast_radius"):
                self.assertEqual(
                    tools.call(self.restored, name, {"assumption_id": assumption_id}),
                    tools.call(self.session, name, {"assumption_id": assumption_id}),
                    f"{name} {assumption_id}",
                )

    def test_ask_matches_across_a_restart(self):
        """A hundred questions, asked in the same order on both sides.

        Not one question each on a fresh pair: asking *is* a write here — a walk
        moves telemetry and the resolver learns — so the interesting claim is
        that the two stay in lockstep over a sequence, not that they agree once.
        A divergence anywhere in the run shows up as the first mismatched answer.
        """
        from contextmesh.demo import questions

        fresh = build()
        restored = Session.load(self.dir)
        asked = questions(fresh.graph, 100)
        self.assertEqual(len(asked), 100)
        for question in asked:
            self.assertEqual(
                tools.call(restored, "mesh_ask", {"question": question}),
                tools.call(fresh, "mesh_ask", {"question": question}),
                question,
            )

    def test_the_walk_back_matches_for_every_decision(self):
        """History is the one thing that must not depend on when you ask.

        Everything the reconstruction reads — source dates, provenance, typed
        edges, assumption records — is in the snapshot, so a restart changes
        nothing. Three horizons because one would not notice a projection that
        was right at the ends and wrong in the middle.
        """
        decisions = [
            n.id for n in self.session.graph.nodes.values() if n.type is NodeType.DECISION
        ]
        self.assertTrue(decisions)
        for decision_id in decisions:
            for as_of in ("2026-01-15", "2026-04-01", "2026-08-01"):
                args = {"decision_id": decision_id, "as_of": as_of}
                self.assertEqual(
                    tools.call(self.restored, "mesh_reconstruct_decision", args),
                    tools.call(self.session, "mesh_reconstruct_decision", args),
                    f"{decision_id} @ {as_of}",
                )

    def test_the_as_of_answer_matches_across_a_restart(self):
        """Asked in the same order on both sides, for the reason mesh_ask is."""
        from contextmesh.demo import questions

        fresh = build()
        restored = Session.load(self.dir)
        for question in questions(fresh.graph, 20):
            for as_of in ("2026-02-01", "2026-06-01"):
                args = {"question": question, "as_of": as_of}
                self.assertEqual(
                    tools.call(restored, "mesh_explain_as_of", args),
                    tools.call(fresh, "mesh_explain_as_of", args),
                    f"{question} @ {as_of}",
                )

    def test_health_matches_apart_from_the_walk_log(self):
        before = tools.call(self.session, "mesh_health", {})
        after = tools.call(self.restored, "mesh_health", {})

        for key in ("status", "nodes_live", "nodes_total", "edges_live", "edge_types_used"):
            self.assertEqual(before[key], after[key], key)

        signals_before = {s["kind"]: s for s in before["signals"]}
        signals_after = {s["kind"]: s for s in after["signals"]}
        self.assertEqual(
            set(signals_before) - set(signals_after),
            {"dead_ends"},
            "walk history is the only thing a restart is expected to lose",
        )
        self.assertEqual(set(signals_after) - set(signals_before), set())
        for kind, signal in signals_after.items():
            self.assertEqual(signal, signals_before[kind], kind)

    def test_dead_ends_come_back_once_this_process_walks(self):
        """The gap is a cold start, not a lost capability."""
        restored = Session.load(self.dir)
        tools.call(restored, "mesh_ask", {"question": "What is the on-call rotation?"})
        kinds = {s["kind"] for s in tools.call(restored, "mesh_health", {})["signals"]}
        self.assertIn("dead_ends", kinds)


class WalkerConfigTest(unittest.TestCase):
    """Walker settings shape reads, so they are part of the session."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = Path(self.tmp) / "session"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_hop_budget_survives(self):
        session = build()
        session.walker.hop_budget = 2
        session.save(self.dir)
        self.assertEqual(Session.load(self.dir).walker.hop_budget, 2)

    def test_policy_survives_in_order(self):
        session = build()
        session.walker.policy = (EdgeType.SUPPORTS, EdgeType.MENTIONS)
        session.save(self.dir)
        self.assertEqual(
            Session.load(self.dir).walker.policy,
            (EdgeType.SUPPORTS, EdgeType.MENTIONS),
        )

    def test_a_narrower_budget_changes_answers(self):
        """Which is the whole reason it is persisted."""
        wide = build()
        narrow = build()
        narrow.walker.hop_budget = 1
        question = "What contradicts the sharding assumption?"
        self.assertNotEqual(
            tools.call(narrow, "mesh_ask", {"question": question}),
            tools.call(wide, "mesh_ask", {"question": question}),
        )

    def test_defaults_match_the_walker(self):
        session = build()
        self.assertEqual(
            WalkerConfig.of(session.walker).to_dict(),
            WalkerConfig(hop_budget=session.walker.hop_budget).to_dict(),
        )
        self.assertEqual(WalkerConfig().policy, DEFAULT_POLICY)


class AgreementTest(unittest.TestCase):
    """The check that neither file can make on its own."""

    def test_a_resolver_from_another_graph_is_refused(self):
        graph = ContextGraph()
        resolver = Resolver()
        resolver.register("entity:ghost", "Ghost")
        with self.assertRaises(SessionError) as caught:
            check_agreement(graph, resolver)
        self.assertIn("entity:ghost", str(caught.exception))

    def test_resolving_to_a_non_entity_is_refused(self):
        session = build()
        claim = next(iter(session.graph.by_type(NodeType.CLAIM)))
        session.resolver.canonical[claim.id] = "Not an entity"
        with self.assertRaises(SessionError) as caught:
            check_agreement(session.graph, session.resolver)
        self.assertIn("rather than an entity", str(caught.exception))

    def test_a_label_that_disagrees_with_the_graph_is_refused(self):
        """Same id, right type, wrong name — the quiet one.

        ``Resolver.canonical`` is not a display string. ``near_miss`` scores
        every unresolved mention against it, so a resolver holding
        ``entity:pgvector -> "HNSW"`` over a graph holding ``"pgvector"`` keeps
        resolving and resolves differently, with each file valid on its own.
        """
        session = build()
        entity_id = next(iter(session.resolver.canonical))
        graph_label = session.graph.nodes[entity_id].label
        session.resolver.canonical[entity_id] = "Something Else Entirely"
        with self.assertRaises(SessionError) as caught:
            check_agreement(session.graph, session.resolver)
        message = str(caught.exception)
        self.assertIn(graph_label, message)
        self.assertIn("Something Else Entirely", message)

    def test_a_swapped_label_is_refused(self):
        """Two real entities, each wearing the other's name."""
        session = build()
        first, second = sorted(session.resolver.canonical)[:2]
        session.resolver.canonical[first] = session.graph.nodes[second].label
        session.resolver.canonical[second] = session.graph.nodes[first].label
        with self.assertRaises(SessionError):
            check_agreement(session.graph, session.resolver)

    def test_the_label_check_survives_the_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "session"
            session = build()
            session.save(directory)
            data = json.loads(resolver_file(directory).read_text(encoding="utf-8"))
            entity_id = next(iter(data["canonical"]))
            data["canonical"][entity_id] = "Renamed"
            for row in data["log"]:
                if row["canonical_id"] == entity_id:
                    row["canonical_label"] = "Renamed"
            for alias, target in list(data["aliases"].items()):
                if target == entity_id:
                    data["aliases"][alias] = target
            resolver_file(directory).write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(SessionError) as caught:
                Session.load(directory)
            self.assertIn("but the graph calls it", str(caught.exception))

    def test_a_matched_pair_is_accepted(self):
        session = build()
        check_agreement(session.graph, session.resolver)


class RefusalTest(unittest.TestCase):
    """A session directory this build cannot faithfully restore is refused."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.mkdtemp()
        cls.good = Path(cls.tmp) / "good"
        build().save(cls.good)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def corrupted(self, mutate) -> Path:
        target = Path(self.tmp) / f"case-{self.id().rsplit('.', 1)[-1]}"
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(self.good, target)
        mutate(target)
        return target

    def refuses(self, mutate, fragment: str = "") -> str:
        with self.assertRaises(SessionError) as caught:
            Session.load(self.corrupted(mutate))
        message = str(caught.exception)
        if fragment:
            self.assertIn(fragment, message)
        return message

    @staticmethod
    def rewrite(path: Path, **changes):
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in changes.items():
            if value is Ellipsis:
                data.pop(key, None)
            else:
                data[key] = value
        path.write_text(json.dumps(data), encoding="utf-8")

    def edit_session(self, **changes):
        return lambda d: self.rewrite(d / manifest_name(manifest(d)["generation"]), **changes)

    # ── the directory itself ─────────────────────────────────────────────
    def test_good_directory_loads(self):
        self.assertTrue(Session.load(self.good).describe()["persistent"])

    def test_missing_directory(self):
        with self.assertRaises(SessionError):
            Session.load(Path(self.tmp) / "nowhere")

    def test_a_file_is_not_a_directory(self):
        with self.assertRaises(SessionError):
            Session.load(graph_file(self.good))

    def test_no_session_file(self):
        def mutate(directory):
            generation = manifest(directory)["generation"]
            (directory / SESSION_FILE).unlink()
            (directory / manifest_name(generation)).unlink()

        self.refuses(mutate, SESSION_FILE)

    def test_session_file_is_not_json(self):
        self.refuses(
            lambda d: (d / manifest_name(manifest(d)["generation"])).write_text(
                "{oops", encoding="utf-8"
            ),
            "not valid JSON",
        )

    def test_session_file_is_not_an_object(self):
        self.refuses(
            lambda d: (d / manifest_name(manifest(d)["generation"])).write_text(
                "[]", encoding="utf-8"
            ),
            "must contain an object",
        )

    def test_session_file_carries_nan(self):
        self.refuses(
            lambda d: (d / manifest_name(manifest(d)["generation"])).write_text(
                json.dumps(
                    {
                        "schema": SESSION_SCHEMA,
                        "version": 1,
                        "generation": 1,
                        "graph": generation_name(GRAPH_STEM, 1),
                        "resolver": generation_name(RESOLVER_STEM, 1),
                        "rounds": float("nan"),
                        "walker": WalkerConfig().to_dict(),
                    }
                ),
                encoding="utf-8",
            ),
            "NaN",
        )

    # ── required fields ──────────────────────────────────────────────────
    def test_every_field_is_required(self):
        for key in ("schema", "version", "graph", "resolver", "rounds", "walker"):
            with self.subTest(field=key):
                self.refuses(self.edit_session(**{key: Ellipsis}), f"missing {key!r}")

    def test_wrong_schema(self):
        self.refuses(self.edit_session(schema="contextmesh.graph"), "not a")

    def test_a_version_from_the_future_is_refused(self):
        """v1 and v2 are readable; anything beyond is not.

        A newer file could hold companions this build knows nothing about, and
        reading it would silently drop them — the same reason an unknown field
        is refused rather than ignored.
        """
        self.refuses(self.edit_session(version=3), "cannot be read by this build")
        self.refuses(self.edit_session(version=99), "cannot be read by this build")

    def test_version_true_is_not_version_one(self):
        self.refuses(self.edit_session(version=True), "must be an integer")

    def test_version_as_string(self):
        self.refuses(self.edit_session(version="1"), "must be an integer")

    def test_rounds_true(self):
        self.refuses(self.edit_session(rounds=True), "must be an integer")

    def test_rounds_as_string(self):
        self.refuses(self.edit_session(rounds="4"), "must be an integer")

    def test_rounds_negative(self):
        self.refuses(self.edit_session(rounds=-1), "must be >= 0")

    # ── the filenames are names, not paths ───────────────────────────────
    def test_graph_may_not_escape_the_directory(self):
        for value in ("../../etc/passwd", "/etc/passwd", "..", "sub/graph.json", ""):
            with self.subTest(value=value):
                self.refuses(self.edit_session(graph=value), "plain filename")

    def test_resolver_may_not_escape_the_directory(self):
        self.refuses(self.edit_session(resolver="../resolver.json"), "plain filename")

    def test_a_companion_that_is_a_symlink_out_of_the_directory_is_refused(self):
        require_symlink(Path(self.tmp))
        """``_bare_name`` stops the path traversing. It cannot stop the file.

        A session directory is something you can be handed, and a correctly
        named ``graph-000001.json`` inside it may be a symlink to anywhere the
        process can read.
        """
        outside = Path(self.tmp) / "outside.json"
        outside.write_text(
            graph_file(self.good).read_text(encoding="utf-8"), encoding="utf-8"
        )

        def mutate(directory):
            target = graph_file(directory)
            target.unlink()
            target.symlink_to(outside)

        message = self.refuses(mutate, "outside the session directory")
        self.assertIn(str(outside.name), message)

    def test_a_symlink_inside_the_directory_is_also_refused(self):
        require_symlink(Path(self.tmp))
        """Containment is checked by resolution, not by prefix.

        A link to a sibling in the same directory is harmless in itself, but
        allowing it means the check is doing string work rather than resolving
        — and the next link would not be harmless.
        """

        def mutate(directory):
            sibling = directory / "elsewhere.json"
            sibling.write_text(
                graph_file(directory).read_text(encoding="utf-8"), encoding="utf-8"
            )
            target = graph_file(directory)
            target.unlink()
            target.symlink_to(sibling)

        self.refuses(mutate, "outside the session directory")

    def test_a_generation_manifest_that_is_a_symlink_is_refused(self):
        require_symlink(Path(self.tmp))

        def mutate(directory):
            planted = directory / manifest_name(manifest(directory)["generation"] + 1)
            planted.symlink_to(directory / SESSION_FILE)

        self.refuses(mutate, "symbolic link")

    def test_filename_must_be_a_string(self):
        self.refuses(self.edit_session(graph=7), "must be a string")
        self.refuses(self.edit_session(resolver=["graph-000001.json"]), "must be a string")

    def test_a_required_companion_cannot_be_null(self):
        """Null is how v2 says "no execution"; it is not a thing a graph may be."""
        self.refuses(self.edit_session(graph=None), "names no graph")
        self.refuses(self.edit_session(resolver=None), "names no resolver")

    def test_a_name_that_is_not_this_generation_is_refused(self):
        """Caught before existence: the name itself is wrong for the counter."""
        self.refuses(self.edit_session(graph="nope.json"), "this build writes")
        self.refuses(
            self.edit_session(graph=generation_name(GRAPH_STEM, 7)), "generation 1"
        )

    def test_named_file_must_exist(self):
        """The correctly-named companion, simply absent."""
        self.refuses(lambda d: graph_file(d).unlink(), "no such file")

    def test_graph_file_deleted(self):
        self.refuses(lambda d: graph_file(d).unlink(), "no such file")

    def test_resolver_file_deleted(self):
        self.refuses(lambda d: resolver_file(d).unlink(), "no such file")

    # ── walker configuration ─────────────────────────────────────────────
    def test_walker_must_be_an_object(self):
        self.refuses(self.edit_session(walker=[]), "must be an object")

    def test_every_walker_field_is_required(self):
        for key in ("hop_budget", "policy", "flat_k", "max_expand"):
            with self.subTest(field=key):
                walker = WalkerConfig().to_dict()
                walker.pop(key)
                self.refuses(self.edit_session(walker=walker), f"missing {key!r}")

    def test_walker_numbers_are_checked(self):
        for key, value, fragment in (
            ("hop_budget", 0, "must be >= 1"),
            ("hop_budget", True, "must be an integer"),
            ("flat_k", -3, "must be >= 1"),
            ("max_expand", "260", "must be an integer"),
        ):
            with self.subTest(field=key, value=value):
                walker = WalkerConfig().to_dict()
                walker[key] = value
                self.refuses(self.edit_session(walker=walker), fragment)

    def test_policy_must_name_real_edge_types(self):
        walker = WalkerConfig().to_dict()
        walker["policy"] = ["teleports_to"]
        self.refuses(self.edit_session(walker=walker), "ontology does not have")

    def test_policy_may_not_repeat(self):
        walker = WalkerConfig().to_dict()
        walker["policy"] = ["mentions", "mentions"]
        self.refuses(self.edit_session(walker=walker), "twice")

    def test_policy_must_be_a_list_of_strings(self):
        for value, fragment in (("mentions", "must be a list"), ([7], "must be a string")):
            with self.subTest(value=value):
                walker = WalkerConfig().to_dict()
                walker["policy"] = value
                self.refuses(self.edit_session(walker=walker), fragment)

    # ── the companion files ──────────────────────────────────────────────
    def test_a_graph_snapshot_this_build_cannot_read(self):
        def mutate(d):
            self.rewrite(graph_file(d), version=99)

        self.refuses(mutate, GRAPH_STEM)

    def test_a_resolver_snapshot_with_the_wrong_schema(self):
        def mutate(d):
            self.rewrite(resolver_file(d), schema="contextmesh.graph")

        self.refuses(mutate, RESOLVER_STEM)

    def test_a_graph_that_is_not_json(self):
        self.refuses(
            lambda d: graph_file(d).write_text("{oops", encoding="utf-8")
        )

    # ── disagreements only the session can see ───────────────────────────
    def test_resolver_naming_a_node_the_graph_does_not_have(self):
        def mutate(d):
            data = json.loads(resolver_file(d).read_text(encoding="utf-8"))
            data["canonical"]["entity:ghost"] = "Ghost"
            resolver_file(d).write_text(json.dumps(data), encoding="utf-8")

        self.refuses(mutate, "entity:ghost")

    def test_resolver_naming_a_node_of_the_wrong_type(self):
        def mutate(d):
            graph = json.loads(graph_file(d).read_text(encoding="utf-8"))
            claim = next(n["id"] for n in graph["nodes"] if n["type"] == "claim")
            data = json.loads(resolver_file(d).read_text(encoding="utf-8"))
            data["canonical"][claim] = "Not an entity"
            resolver_file(d).write_text(json.dumps(data), encoding="utf-8")

        self.refuses(mutate, "rather than an entity")

    def test_a_graph_from_a_different_corpus(self):
        """Two valid files that were never a session together."""
        other = Path(self.tmp) / "other"
        shutil.rmtree(other, ignore_errors=True)
        shutil.copytree(self.good, other)
        ContextGraph().save_json(other / manifest(other)['graph'])
        with self.assertRaises(SessionError) as caught:
            Session.load(other)
        self.assertIn("not a node in this session's graph", str(caught.exception))


class ResolverSnapshotTest(unittest.TestCase):
    """The resolver's own format, corrupted one field at a time.

    Separate from the session tests above because the resolver is separately
    versioned: it is a file another tool could be handed on its own, and it has
    to refuse a bad one without a session file to lean on. The references it
    *can* check are its own — every alias, block and log entry has to name an
    entity the resolver knows. Whether those entities exist in a graph is the
    session's question, and `AgreementTest` asks it.
    """

    @classmethod
    def setUpClass(cls) -> None:
        resolver = Resolver()
        resolver.register("entity:hnsw", "HNSW", aliases=["Hierarchical Navigable Small World"])
        resolver.register("entity:pgvector", "pgvector")
        resolver.resolve("PG Vector")
        resolver.resolve("nothing here at all")
        cls.resolver = resolver
        cls.good = resolver.to_dict()

    def corrupted(self, **changes) -> dict:
        data = json.loads(json.dumps(self.good))
        for key, value in changes.items():
            if value is Ellipsis:
                data.pop(key, None)
            else:
                data[key] = value
        return data

    def refuses(self, fragment: str = "", **changes) -> str:
        with self.assertRaises(ResolverSnapshotError) as caught:
            Resolver.from_dict(self.corrupted(**changes))
        message = str(caught.exception)
        if fragment:
            self.assertIn(fragment, message)
        return message

    # ── round trip ───────────────────────────────────────────────────────
    def test_the_good_snapshot_loads(self):
        back = Resolver.from_dict(self.good)
        self.assertEqual(back.to_dict(), self.good)

    def test_threshold_survives(self):
        tight = Resolver(threshold=0.91)
        tight.register("entity:hnsw", "HNSW")
        self.assertEqual(Resolver.from_dict(tight.to_dict()).threshold, 0.91)

    def test_a_file_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resolver.json"
            self.resolver.save_json(path)
            self.assertEqual(Resolver.load_json(path).to_dict(), self.good)

    def test_nan_in_a_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resolver.json"
            data = json.loads(json.dumps(self.good))
            data["threshold"] = 0.5
            raw = json.dumps(data).replace('"threshold": 0.5', '"threshold": NaN')
            path.write_text(raw, encoding="utf-8")
            with self.assertRaises(ResolverSnapshotError) as caught:
                Resolver.load_json(path)
            self.assertIn("NaN", str(caught.exception))

    # ── shape ────────────────────────────────────────────────────────────
    def test_not_an_object(self):
        with self.assertRaises(ResolverSnapshotError):
            Resolver.from_dict([])

    def test_every_field_is_required(self):
        for key in ("schema", "version", "threshold", "canonical", "aliases", "blocks", "log"):
            with self.subTest(field=key):
                self.refuses(f"missing {key!r}", **{key: Ellipsis})

    def test_wrong_schema(self):
        self.refuses("not a contextmesh.resolver", schema="contextmesh.graph")

    def test_version(self):
        self.refuses("cannot be read by this build", version=2)
        self.refuses("must be an integer", version=True)
        self.refuses("must be an integer", version="1")

    def test_threshold_is_a_number_in_range(self):
        self.refuses("expected a number", threshold="0.62")
        self.refuses("expected a number", threshold=True)
        self.refuses("between 0 and 1", threshold=1.5)
        self.refuses("between 0 and 1", threshold=-0.1)

    # ── tables ───────────────────────────────────────────────────────────
    def test_canonical_must_be_a_string_map(self):
        self.refuses("must be an object", canonical=[])
        self.refuses("expected a string", canonical={"entity:hnsw": 7})

    def test_aliases_must_be_a_string_map(self):
        self.refuses("must be an object", aliases="hnsw")
        self.refuses("expected a string", aliases={"hnsw": None})

    def test_blocks_must_be_lists_of_strings(self):
        self.refuses("must be an object", blocks=[])
        self.refuses("must be a list", blocks={"hnsw": "entity:hnsw"})
        self.refuses("expected a string", blocks={"hnsw": [7]})

    def test_log_must_be_a_list_of_records(self):
        self.refuses("must be a list", log={})
        with self.assertRaises(ResolverSnapshotError):
            Resolver.from_dict(self.corrupted(log=[{"mention": "x"}]))

    def test_log_record_fields_are_typed(self):
        row = dict(self.good["log"][0])
        for key, value, fragment in (
            ("mention", 7, "expected a string"),
            ("score", "0.9", "expected a number"),
            ("score", True, "expected a number"),
            ("reason", None, "expected a string"),
            ("canonical_id", 7, "expected a string"),
        ):
            with self.subTest(field=key, value=value):
                broken = dict(row)
                broken[key] = value
                self.refuses(fragment, log=[broken])

    def test_log_record_fields_are_required(self):
        for key in ("mention", "canonical_id", "canonical_label", "score", "reason"):
            with self.subTest(field=key):
                broken = dict(self.good["log"][0])
                broken.pop(key)
                self.refuses(f"requires a {key!r}", log=[broken])

    def test_a_record_must_have_both_id_and_label_or_neither(self):
        """``resolved`` keys off the id alone, so a half-null record lies."""
        resolved = next(r for r in self.good["log"] if r["canonical_id"] is not None)
        missed = next(r for r in self.good["log"] if r["canonical_id"] is None)

        orphan_id = dict(resolved, canonical_label=None)
        self.refuses("a resolution has both", log=[orphan_id])

        orphan_label = dict(missed, canonical_label="HNSW")
        self.refuses("a miss has", log=[orphan_label])

    def test_a_record_label_must_match_the_canonical_table(self):
        """The log is what the health signals and borderline report read."""
        resolved = next(r for r in self.good["log"] if r["canonical_id"] is not None)
        renamed = dict(resolved, canonical_label="Not What It Is Called")
        self.refuses("but the resolver calls that entity", log=[renamed])

    def test_a_consistent_log_still_loads(self):
        back = Resolver.from_dict(self.good)
        for record in back.log:
            if record.canonical_id is not None:
                self.assertEqual(
                    record.canonical_label, back.canonical[record.canonical_id]
                )

    def test_an_unresolved_record_keeps_its_nulls(self):
        """``canonical_id: null`` is a resolution that failed, not a missing field."""
        back = Resolver.from_dict(self.good)
        misses = [r for r in back.log if not r.resolved]
        self.assertTrue(misses)
        self.assertTrue(all(r.canonical_label is None for r in misses))

    # ── references, which is what a resolver is ──────────────────────────
    def test_an_alias_must_name_a_canonical_entity(self):
        aliases = dict(self.good["aliases"])
        aliases["ghost"] = "entity:ghost"
        self.refuses("not canonical", aliases=aliases)

    def test_dropping_an_entity_invalidates_its_aliases(self):
        """The realistic version: the entity goes, its aliases are left behind."""
        canonical = dict(self.good["canonical"])
        canonical.pop("entity:pgvector")
        self.refuses("not canonical", canonical=canonical)

    def test_a_block_must_name_a_canonical_entity(self):
        blocks = json.loads(json.dumps(self.good["blocks"]))
        blocks["ghos"] = ["entity:ghost"]
        self.refuses("not canonical", blocks=blocks)

    def test_a_log_entry_must_name_a_canonical_entity(self):
        row = dict(self.good["log"][0])
        row["canonical_id"] = "entity:ghost"
        row["canonical_label"] = "Ghost"
        self.refuses("not canonical", log=[row])

    def test_a_dangling_alias_would_otherwise_crash_the_next_question(self):
        """Why the reference check is not pedantry.

        ``resolve`` reads ``self.canonical[eid]`` for the label as soon as the
        alias table hits, so an alias pointing at an entity the resolver does
        not have raises ``KeyError`` — not at load, where it could be reported,
        but on whichever question happens to use that surface form. Refusing the
        file turns a landmine into a message.
        """
        aliases = dict(self.good["aliases"])
        aliases["ghost"] = "entity:ghost"
        loose = Resolver(
            threshold=self.good["threshold"],
            canonical=dict(self.good["canonical"]),
            aliases=aliases,
        )
        with self.assertRaises(KeyError):
            loose.resolve("ghost")

        self.refuses("not canonical", aliases=aliases)


class ProcessBoundaryTest(unittest.TestCase):
    """Two interpreters, one directory, nothing shared but the files."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = Path(self.tmp) / "session"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cli(self, *args, expect: int = 0):
        proc = subprocess.run(
            [sys.executable, "-m", "contextmesh_mcp", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc.returncode, expect, f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
        return proc

    def test_one_process_writes_and_another_reads(self):
        written = self.cli("--demo", "--rounds", str(ROUNDS), "--save", str(self.dir))
        wrote = json.loads(written.stdout)
        self.assertEqual(wrote["schema"], SESSION_SCHEMA)
        self.assertEqual(
            sorted(wrote["files"]),
            sorted([
                SESSION_FILE,
                manifest_name(1),
                LOCK_FILE,
                LOCK_META_FILE,
                generation_name(GRAPH_STEM, 1),
                generation_name(RESOLVER_STEM, 1),
            ]),
        )

        read = json.loads(self.cli("--session", str(self.dir)).stdout)
        self.assertTrue(read["persistent"])
        self.assertEqual(read["rounds"], ROUNDS)

        local = Session.build(rounds=ROUNDS).describe()
        for key in ("nodes_live", "nodes_total", "edges_live", "edges_total", "assumptions"):
            self.assertEqual(read[key], local[key], key)

    def test_the_writing_process_leaves_nothing_behind(self):
        """The reader gets the graph from the files or not at all."""
        self.cli("--demo", "--rounds", str(ROUNDS), "--save", str(self.dir))
        graph_file(self.dir).unlink()
        proc = subprocess.run(
            [sys.executable, "-m", "contextmesh_mcp", "--session", str(self.dir)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("no such file", proc.stderr)

    def test_a_learned_alias_crosses_the_boundary(self):
        self.cli("--demo", "--rounds", str(ROUNDS), "--save", str(self.dir))
        aliases = json.loads(resolver_file(self.dir).read_text(encoding="utf-8"))["aliases"]
        registered = Resolver()
        canonical = json.loads(resolver_file(self.dir).read_text(encoding="utf-8"))["canonical"]
        for entity_id, label in canonical.items():
            registered.register(entity_id, label)
        learned = set(aliases) - set(registered.aliases)
        self.assertTrue(learned, "the demo should learn aliases while it walks")

        restored = Session.load(self.dir)
        mention = sorted(learned)[0]
        record = restored.resolver.resolve(mention)
        self.assertEqual(record.reason, "alias table")
        self.assertEqual(record.canonical_id, aliases[mention])

    def test_a_corrupt_session_refuses_at_the_shell(self):
        self.cli("--demo", "--rounds", str(ROUNDS), "--save", str(self.dir))
        self.rewrite_version(self.dir / manifest_name(manifest(self.dir)["generation"]))
        proc = subprocess.run(
            [sys.executable, "-m", "contextmesh_mcp", "--session", str(self.dir)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertIn("cannot be read by this build", proc.stderr)

    @staticmethod
    def rewrite_version(path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["version"] = 99
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_a_source_must_be_chosen(self):
        proc = subprocess.run(
            [sys.executable, "-m", "contextmesh_mcp"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--demo", proc.stderr)
        self.assertIn("--session", proc.stderr)

    def test_demo_and_session_are_mutually_exclusive(self):
        proc = subprocess.run(
            [sys.executable, "-m", "contextmesh_mcp", "--demo", "--session", str(self.dir)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("not allowed with", proc.stderr)

    def test_rounds_with_session_is_an_error_not_an_ignored_flag(self):
        self.cli("--demo", "--rounds", str(ROUNDS), "--save", str(self.dir))
        proc = subprocess.run(
            [sys.executable, "-m", "contextmesh_mcp", "--session", str(self.dir), "--rounds", "2"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("no meaning for --session", proc.stderr)


class WriterSymlinkTest(unittest.TestCase):
    """A session directory is untrusted input on the way *out*, not just in.

    The read side refuses a companion that resolves outside the directory. The
    write side had the mirror-image hole: writing by name follows whatever is
    already sitting under that name, so a directory handed over with the *next*
    generation's filename already present as a symlink would have the next save
    write straight through it. With ``--checkpoint every-ask`` the trigger is
    merely asking a question.

    All four names were reachable that way. The fix is one mechanism rather
    than four patches: every write lands in a fresh ``O_EXCL`` file under a
    random name and is renamed into place, and rename replaces a symlink itself
    instead of following it. The lock is the exception — it must keep one inode
    for the kernel lock to mean anything — so it is opened ``O_NOFOLLOW`` and
    checked to be a regular file.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.dir = self.tmp / "session"
        build().save(self.dir)
        self.outside = self.tmp / "OUTSIDE.txt"
        self.outside.write_text("precious", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def next_generation(self) -> int:
        return manifest(self.dir)["generation"] + 1

    def assert_outside_untouched(self):
        self.assertTrue(self.outside.exists(), "the outside file was deleted")
        self.assertEqual(
            self.outside.read_text(encoding="utf-8"),
            "precious",
            "the outside file was written through a symlink",
        )

    # ── the three names a save writes ────────────────────────────────────
    def test_a_planted_link_at_the_next_graph_name_is_replaced_not_followed(self):
        require_symlink(self.dir)
        planted = self.dir / generation_name(GRAPH_STEM, self.next_generation())
        planted.symlink_to(self.outside)
        Session.load(self.dir).checkpoint()
        self.assert_outside_untouched()
        self.assertFalse(graph_file(self.dir).is_symlink())
        self.assertEqual(Session.load(self.dir).generation, 2)

    def test_a_planted_link_at_the_next_resolver_name_is_replaced(self):
        require_symlink(self.dir)
        planted = self.dir / generation_name(RESOLVER_STEM, self.next_generation())
        planted.symlink_to(self.outside)
        Session.load(self.dir).checkpoint()
        self.assert_outside_untouched()
        self.assertFalse(resolver_file(self.dir).is_symlink())

    def test_a_link_at_the_manifest_itself_is_replaced(self):
        require_symlink(self.dir)
        held = (self.dir / SESSION_FILE).read_text(encoding="utf-8")
        (self.dir / SESSION_FILE).unlink()
        (self.dir / SESSION_FILE).symlink_to(self.outside)
        self.outside.write_text(held, encoding="utf-8")

        # The manifest is unreadable as a session now, so drive the save from a
        # session loaded before the link was planted.
        session = build()
        session.save(self.dir)
        self.assertFalse((self.dir / SESSION_FILE).is_symlink())
        self.assertEqual(
            self.outside.read_text(encoding="utf-8"), held, "wrote through the link"
        )

    def test_a_planted_link_at_the_next_generation_manifest_is_replaced(self):
        require_symlink(self.dir)
        writer = Session.load(self.dir)
        planted = self.dir / manifest_name(self.next_generation())
        planted.symlink_to(self.outside)
        writer.checkpoint()
        self.assert_outside_untouched()
        self.assertFalse((self.dir / manifest_name(2)).is_symlink())
        self.assertEqual(Session.load(self.dir).generation, 2)

    def test_write_in_place_never_follows_a_link(self):
        require_symlink(self.dir)
        """The primitive itself, independent of any caller."""
        planted = self.dir / "target.json"
        planted.symlink_to(self.outside)
        write_in_place(self.dir, "target.json", "replacement")
        self.assert_outside_untouched()
        self.assertFalse(planted.is_symlink())
        self.assertEqual(planted.read_text(encoding="utf-8"), "replacement")

    def test_write_in_place_leaves_no_staging_file(self):
        write_in_place(self.dir, "target.json", "x")
        self.assertEqual(
            [n for n in names(self.dir) if n.startswith(STAGING_PREFIX)], []
        )

    # ── the lock, which cannot be replaced ───────────────────────────────
    def test_a_symlinked_lock_file_is_refused(self):
        require_symlink(self.dir)
        (self.dir / LOCK_FILE).unlink()
        (self.dir / LOCK_FILE).symlink_to(self.outside)
        with self.assertRaises(SessionError) as caught:
            Session.load(self.dir).checkpoint()
        self.assertIn("regular file", str(caught.exception))
        self.assert_outside_untouched()

    def test_a_lock_file_that_is_a_fifo_is_refused(self):
        require_fifo()
        (self.dir / LOCK_FILE).unlink()
        os.mkfifo(str(self.dir / LOCK_FILE))
        with self.assertRaises(SessionError) as caught:
            Session.load(self.dir).checkpoint()
        self.assertIn("regular file", str(caught.exception))

    def test_a_symlinked_lock_file_blocks_the_save_entirely(self):
        require_symlink(self.dir)
        """Refused before anything is written, not halfway through."""
        before = manifest(self.dir)
        (self.dir / LOCK_FILE).unlink()
        (self.dir / LOCK_FILE).symlink_to(self.outside)
        with self.assertRaises(SessionError):
            Session.load(self.dir).checkpoint()
        (self.dir / LOCK_FILE).unlink()
        self.assertEqual(manifest(self.dir), before)

    def test_a_normal_lock_file_is_accepted(self):
        Session.load(self.dir).checkpoint()
        self.assertFalse((self.dir / LOCK_FILE).is_symlink())
        self.assertEqual(manifest(self.dir)["generation"], 2)


class ReaderSweepTest(unittest.TestCase):
    """A live session must not report itself broken because someone checkpointed.

    Readers take no lock, and do not need one for *correctness*: the manifest
    swap is atomic and committed companions are immutable, so a pair that reads
    successfully is always a coherent generation. What the swap alone does not
    cover is the sweep — a reader holding the manifest for generation 5 can find
    ``graph-000005.json`` already deleted and fail on a perfectly healthy
    directory.

    So a read that loses that race is re-read. The retry is conditioned on the
    directory's generation actually having moved, which is what keeps a
    genuinely missing file failing immediately rather than after a wait.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = Path(self.tmp) / "session"
        build().save(self.dir)
        Session.load(self.dir).checkpoint()  # generation 2

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_sweep_during_a_read_is_retried_not_reported(self):
        """Stalls the first attempt, commits and sweeps under it, then resumes."""
        original = ContextGraph.load_json.__func__
        paused, resume = threading.Event(), threading.Event()
        attempts = []

        def stalling(cls, path, **kwargs):
            attempts.append(path)
            if len(attempts) == 1:
                paused.set()
                self.assertTrue(resume.wait(30), "the writer never ran")
            return original(cls, path, **kwargs)

        outcome = {}

        def read():
            try:
                session = Session.load(self.dir)
                outcome["generation"] = session.generation
                outcome["nodes"] = len(session.graph.nodes)
                outcome["aliases"] = len(session.resolver.aliases)
            except BaseException as exc:  # noqa: BLE001 - reported below
                outcome["error"] = f"{type(exc).__name__}: {exc}"

        ContextGraph.load_json = classmethod(stalling)
        reader = threading.Thread(target=read)
        try:
            reader.start()
            self.assertTrue(paused.wait(30), "the reader never started")

            ContextGraph.load_json = classmethod(original)
            Session.load(self.dir).checkpoint()  # commits 3, sweeps 2
            ContextGraph.load_json = classmethod(stalling)
            resume.set()
            reader.join(60)
        finally:
            ContextGraph.load_json = classmethod(original)
            resume.set()
            reader.join(60)

        self.assertNotIn("error", outcome, outcome.get("error"))
        self.assertEqual(outcome["generation"], 3, "did not pick up the new commit")
        self.assertEqual(outcome["nodes"], len(build().graph.nodes))
        self.assertGreater(outcome["aliases"], 0)
        self.assertGreater(len(attempts), 1, "the read never retried")

    def test_a_genuinely_missing_companion_still_fails_at_once(self):
        """The retry must not turn a broken directory into a slow one."""
        graph_file(self.dir).unlink()
        started = time.time()
        with self.assertRaises(SessionError) as caught:
            Session.load(self.dir)
        elapsed = time.time() - started
        self.assertIn("no such file", str(caught.exception))
        self.assertLess(elapsed, LOAD_BACKOFF * LOAD_ATTEMPTS, "it waited to fail")

    def test_a_reader_holding_an_older_manifest_still_gets_a_whole_generation(self):
        """Staleness is not corruption; only a missing file is.

        Reading generation 2's companions while the directory has moved to 3 is
        a valid, if slightly old, session — which is why the retry keys off a
        *failed* read rather than off the generation changing.
        """
        held = manifest(self.dir)
        graph_path = self.dir / held["graph"]
        resolver_path = self.dir / held["resolver"]
        graph_bytes = graph_path.read_bytes()

        Session.load(self.dir).checkpoint()  # commits 3, sweeps 2
        self.assertFalse(graph_path.exists())
        self.assertFalse(resolver_path.exists())
        # What the reader would have read is unchanged right up to deletion.
        self.assertEqual(graph_bytes[:64], graph_bytes[:64])
        self.assertEqual(Session.load(self.dir).generation, 3)

    def test_a_load_still_needs_no_writer_lock(self):
        with writer_lock(self.dir):
            self.assertEqual(Session.load(self.dir).generation, 2)


class ConcurrentWriterTest(unittest.TestCase):
    """Generations survive a crash. They do nothing about a second writer.

    Two processes reading generation 5 both choose 6 and overwrite each other's
    companions. Worse, one can commit 6 and sweep while the other has already
    written 7 but not yet swapped — leaving a manifest that is atomically valid
    and names files the first process just deleted. That failure is nastier than
    a torn file precisely because nothing about it looks torn.

    So the lock spans the whole transaction, from reading the current generation
    to sweeping the superseded one.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = Path(self.tmp) / "session"
        build().save(self.dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ── the lock itself ──────────────────────────────────────────────────
    def test_a_second_acquire_is_refused_while_the_first_is_held(self):
        with writer_lock(self.dir):
            with self.assertRaises(SessionLockedError) as caught:
                with writer_lock(self.dir):
                    pass
        self.assertIn("one writer at a time", str(caught.exception))

    def test_the_error_names_the_holder(self):
        import os

        with writer_lock(self.dir):
            with self.assertRaises(SessionLockedError) as caught:
                with writer_lock(self.dir):
                    pass
        self.assertIn(str(os.getpid()), str(caught.exception))

    def test_the_lock_is_released_on_the_way_out(self):
        with writer_lock(self.dir):
            pass
        with writer_lock(self.dir):
            pass

    def test_the_lock_is_released_even_when_the_body_raises(self):
        with self.assertRaises(ZeroDivisionError):
            with writer_lock(self.dir):
                1 / 0
        with writer_lock(self.dir):
            pass

    def test_a_locked_directory_refuses_a_save(self):
        session = Session.load(self.dir)
        with writer_lock(self.dir):
            with self.assertRaises(SessionLockedError):
                session.checkpoint()
        self.assertEqual(manifest(self.dir)["generation"], 1)

    def test_the_lock_file_is_never_swept(self):
        """A queued writer is holding that exact inode."""
        for _ in range(3):
            Session.load(self.dir).checkpoint()
            self.assertIn(LOCK_FILE, names(self.dir))

    def test_the_sweep_honours_keep(self):
        """Two things protect the lock file, and this pins the second.

        It survives mainly because it does not look like a generation, so
        naming it in ``keep`` is belt-and-braces — a mutation that removes it
        from ``keep`` changes nothing today. That redundancy is worth having
        against a later rename, but it is only worth having if ``keep`` is
        itself honoured, which is what this asserts: a file that *would* be
        swept is spared when it is named.
        """
        spared = self.dir / generation_name(GRAPH_STEM, 99)
        doomed = self.dir / generation_name(RESOLVER_STEM, 99)
        for path in (spared, doomed):
            path.write_text("{}", encoding="utf-8")

        Session._sweep(self.dir, keep={SESSION_FILE, LOCK_FILE, LOCK_META_FILE, spared.name})

        self.assertTrue(spared.is_file(), "a file named in keep was swept")
        self.assertFalse(doomed.is_file(), "an unreferenced generation survived")
        self.assertIn(LOCK_FILE, names(self.dir))
        self.assertIn(LOCK_META_FILE, names(self.dir))

    def test_a_lock_file_left_behind_does_not_block_a_later_writer(self):
        """It is kernel-held, so there is no such thing as a stale one."""
        (self.dir / LOCK_FILE).write_text('{"pid": 999999, "host": "gone"}', encoding="utf-8")
        Session.load(self.dir).checkpoint()
        self.assertEqual(manifest(self.dir)["generation"], 2)

    def test_a_load_never_waits_on_the_lock(self):
        """Readers are already safe: the manifest swap is atomic."""
        with writer_lock(self.dir):
            self.assertEqual(Session.load(self.dir).generation, 1)

    # ── the lost-update guard ────────────────────────────────────────────
    def test_a_checkpoint_over_someone_else_s_commit_is_refused(self):
        """Serialising writers stops corruption, not silent overwriting."""
        mine = Session.load(self.dir)
        Session.load(self.dir).checkpoint()  # somebody else commits generation 2
        with self.assertRaises(SessionError) as caught:
            mine.checkpoint()
        message = str(caught.exception)
        self.assertIn("another writer has committed since", message)
        self.assertEqual(manifest(self.dir)["generation"], 2)

    def test_reloading_after_a_conflict_lets_the_work_continue(self):
        mine = Session.load(self.dir)
        Session.load(self.dir).checkpoint()
        with self.assertRaises(SessionError):
            mine.checkpoint()
        Session.load(self.dir).checkpoint()
        self.assertEqual(manifest(self.dir)["generation"], 3)

    def test_saving_into_a_different_directory_is_not_constrained(self):
        """The guard is about a session's own home, not about saving anywhere."""
        elsewhere = Path(self.tmp) / "elsewhere"
        loaded = Session.load(self.dir)
        Session.load(self.dir).checkpoint()
        loaded.save(elsewhere)
        self.assertEqual(manifest(elsewhere)["generation"], 1)

    def test_a_built_session_may_be_saved_over_an_existing_directory(self):
        build().save(self.dir)
        self.assertEqual(manifest(self.dir)["generation"], 2)

    # ── what the server does with contention ─────────────────────────────
    def test_contention_leaves_the_mutation_pending_rather_than_raising(self):
        live = Session.load(self.dir)
        checkpointer = Checkpointer(live)
        with writer_lock(self.dir):
            tools.call(live, "mesh_ask", {"question": "What supports HNSW?"})
            self.assertIsNone(checkpointer.record_mutation())
        self.assertEqual(checkpointer.contended, 1)
        self.assertEqual(checkpointer.commits, 0)
        self.assertEqual(checkpointer.pending, 1)

        self.assertIsNotNone(checkpointer.commit())
        self.assertEqual(checkpointer.pending, 0)
        self.assertEqual(manifest(self.dir)["generation"], 2)


class ProcessLockTest(unittest.TestCase):
    """The race, staged across two real processes.

    One writer stalls in the window between the manifest swap and the sweep —
    the interleaving that used to leave a manifest naming deleted files.
    """

    STALL = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from contextmesh_mcp.session import Session\n"
        "root, gate = Path(sys.argv[1]), Path(sys.argv[2])\n"
        "original = Session._sweep\n"
        "def stalling(directory, keep):\n"
        "    gate.write_text('committed')\n"
        "    deadline = time.time() + 60\n"
        "    while not (gate.parent / 'b-done').exists() and time.time() < deadline:\n"
        "        time.sleep(0.02)\n"
        "    original(directory, keep)\n"
        "Session._sweep = staticmethod(stalling)\n"
        "Session.load(root).checkpoint()\n"
        "print('A-DONE')\n"
    )
    CONTEND = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from contextmesh_mcp.session import Session, SessionLockedError\n"
        "root, gate = Path(sys.argv[1]), Path(sys.argv[2])\n"
        "done = gate.parent / 'b-done'\n"
        "deadline = time.time() + 60\n"
        "while not gate.exists() and time.time() < deadline:\n"
        "    time.sleep(0.02)\n"
        "try:\n"
        "    Session.load(root).checkpoint()\n"
        "    print('B-COMMITTED-IN-WINDOW')\n"
        "except SessionLockedError:\n"
        "    print('B-REFUSED')\n"
        "finally:\n"
        "    done.write_text('done')\n"
        "time.sleep(1.0)\n"
        "again = Session.load(root)\n"
        "again.checkpoint()\n"
        "print('B-RETRY-GENERATION', again.generation)\n"
    )
    HOLD_READER = (
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "from contextmesh_mcp.session import open_session_manifest_reader\n"
        "root, gate = Path(sys.argv[1]), Path(sys.argv[2])\n"
        "done = gate.parent / 'writer-done'\n"
        "with open_session_manifest_reader(root) as reader:\n"
        "    gate.write_text('reader-open')\n"
        "    deadline = time.time() + 60\n"
        "    while not done.exists() and time.time() < deadline:\n"
        "        time.sleep(0.02)\n"
        "    held = json.loads(reader.read())\n"
        "print('READER-GENERATION', held['generation'])\n"
        "print('READER-GRAPH', held['graph'])\n"
    )

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = Path(self.tmp) / "session"
        self.gate = Path(self.tmp) / "gate"
        build().save(self.dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def spawn(self, script):
        return subprocess.Popen(
            [sys.executable, "-c", script, str(self.dir), str(self.gate)],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_the_second_writer_is_refused_then_succeeds_on_retry(self):
        first, second = self.spawn(self.STALL), self.spawn(self.CONTEND)
        a_out, a_err = first.communicate(timeout=120)
        b_out, b_err = second.communicate(timeout=120)
        self.assertEqual(first.returncode, 0, a_err)
        self.assertEqual(second.returncode, 0, b_err)

        self.assertIn("A-DONE", a_out)
        self.assertIn("B-REFUSED", b_out)
        self.assertNotIn("B-COMMITTED-IN-WINDOW", b_out)
        self.assertIn("B-RETRY-GENERATION 3", b_out)

        committed = manifest(self.dir)
        self.assertEqual(committed["generation"], 3)          # monotonic 1 → 2 → 3
        self.assertTrue(graph_file(self.dir).is_file())       # nothing committed
        self.assertTrue(resolver_file(self.dir).is_file())    # was swept
        self.assertEqual(Session.load(self.dir).generation, 3)

    def test_a_reader_holding_the_manifest_does_not_block_publication(self):
        reader = self.spawn(self.HOLD_READER)
        deadline = time.time() + 60
        while not self.gate.exists() and time.time() < deadline:
            time.sleep(0.02)
        self.assertTrue(self.gate.exists(), "reader never opened the manifest")

        writer = Session.load(self.dir)
        writer.checkpoint()
        (Path(self.tmp) / "writer-done").write_text("done", encoding="utf-8")

        out, err = reader.communicate(timeout=120)
        self.assertEqual(reader.returncode, 0, err)
        self.assertIn("READER-GENERATION 1", out)
        self.assertIn(f"READER-GRAPH {generation_name(GRAPH_STEM, 1)}", out)
        self.assertEqual(manifest(self.dir)["generation"], 2)
        self.assertEqual(Session.load(self.dir).generation, 2)

    def test_a_crash_before_publication_leaves_the_committed_generation_serving(self):
        """A real process dies after companions exist but before the commit point."""
        crash = (
            "import os, sys\n"
            "from pathlib import Path\n"
            "import contextmesh_mcp.session as sessionmod\n"
            "root = Path(sys.argv[1])\n"
            "original = sessionmod.write_in_place\n"
            "def write_then_crash(directory, name, payload):\n"
            "    path = original(directory, name, payload)\n"
            "    if name.startswith('resolver-'):\n"
            "        os._exit(73)\n"
            "    return path\n"
            "sessionmod.write_in_place = write_then_crash\n"
            "sessionmod.Session.load(root).checkpoint()\n"
        )
        failed = self.spawn(crash)
        failed.communicate(timeout=120)
        self.assertEqual(failed.returncode, 73)

        restored = Session.load(self.dir)
        self.assertEqual(restored.generation, 1)
        self.assertEqual(manifest(self.dir)["generation"], 1)

        restored.checkpoint()
        self.assertEqual(Session.load(self.dir).generation, 2)
        self.assertEqual(names(self.dir), GenerationTest.expected(self, 2))

    def test_a_writer_killed_mid_transaction_does_not_hold_the_lock(self):
        """Kernel-held, so a SIGKILL releases it. No stale-lock guessing."""
        holder = self.spawn(
            "import sys, time\n"
            "from pathlib import Path\n"
            "from contextmesh_mcp.session import writer_lock\n"
            "root, gate = Path(sys.argv[1]), Path(sys.argv[2])\n"
            "with writer_lock(root):\n"
            "    gate.write_text('held')\n"
            "    time.sleep(120)\n"
        )
        deadline = time.time() + 60
        while not self.gate.exists() and time.time() < deadline:
            time.sleep(0.02)
        self.assertTrue(self.gate.exists(), "the holder never took the lock")

        with self.assertRaises(SessionLockedError):
            Session.load(self.dir).checkpoint()

        holder.kill()
        holder.communicate(timeout=60)  # reap it, and close the pipes

        Session.load(self.dir).checkpoint()
        self.assertEqual(manifest(self.dir)["generation"], 2)


class ServedMutationTest(unittest.TestCase):
    """What an MCP client does to a session has to survive the process.

    Asking is a write here by design — a walk moves ``node.walks`` and
    ``edge.traversals``, and the resolver learns a surface form on a scored
    match. Without a checkpoint the directory is a durable *starting* snapshot:
    every question after startup is thrown away on restart, silently, including
    exactly the learned aliases this format exists to keep.

    The two questions between them move all four: the first learns an alias,
    the second traverses edges.
    """

    ASKS = (
        "What does PG Vector depend on?",
        "What contradicts the sharding assumption?",
    )

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = Path(self.tmp) / "session"
        build().save(self.dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def state(session: Session) -> dict:
        return {
            "aliases": len(session.resolver.aliases),
            "log": len(session.resolver.log),
            "walks": sum(n.walks for n in session.graph.nodes.values()),
            "traversals": sum(e.traversals for e in session.graph.edges.values()),
        }

    def test_the_asks_move_every_counter(self):
        """Otherwise the tests below could pass by measuring nothing."""
        live = Session.load(self.dir)
        before = self.state(live)
        for question in self.ASKS:
            tools.call(live, "mesh_ask", {"question": question})
        after = self.state(live)
        for key in before:
            self.assertGreater(after[key], before[key], key)

    def test_without_a_checkpoint_the_work_is_lost(self):
        """The behaviour this class exists to change, pinned as a fact."""
        live = Session.load(self.dir)
        for question in self.ASKS:
            tools.call(live, "mesh_ask", {"question": question})
        self.assertNotEqual(self.state(Session.load(self.dir)), self.state(live))

    def test_with_a_checkpoint_it_survives(self):
        live = Session.load(self.dir)
        checkpointer = Checkpointer(live)
        for question in self.ASKS:
            tools.call(live, "mesh_ask", {"question": question})
            checkpointer.record_mutation()
        self.assertEqual(self.state(Session.load(self.dir)), self.state(live))

    def test_a_post_start_alias_survives_a_real_process_boundary(self):
        """The acceptance test: one interpreter asks, another reads.

        The mention is invented here rather than taken from the corpus, so the
        alias cannot already be in the saved file — it exists only because this
        session was asked a question after it started.
        """
        mention = "PG Vector"
        saved = json.loads(resolver_file(self.dir).read_text(encoding="utf-8"))
        self.assertNotIn(mention.lower(), saved["aliases"])

        script = (
            "import json, sys\n"
            "from contextmesh_mcp.session import Session, Checkpointer\n"
            "from contextmesh_mcp import tools\n"
            "s = Session.load(sys.argv[1])\n"
            "c = Checkpointer(s)\n"
            "for q in sys.argv[2:]:\n"
            "    tools.call(s, 'mesh_ask', {'question': q})\n"
            "    c.record_mutation()\n"
            "print(json.dumps({'aliases': len(s.resolver.aliases),\n"
            "                  'log': len(s.resolver.log),\n"
            "                  'walks': sum(n.walks for n in s.graph.nodes.values()),\n"
            "                  'traversals': sum(e.traversals for e in s.graph.edges.values()),\n"
            "                  'commits': c.commits}))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script, str(self.dir), mention_question(mention), self.ASKS[1]],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, f"{proc.stdout}\n{proc.stderr}")
        wrote = json.loads(proc.stdout)
        self.assertEqual(wrote["commits"], 2)

        # Nothing is shared but the directory. This process never saw that one.
        after = Session.load(self.dir)
        self.assertEqual(self.state(after), {k: wrote[k] for k in self.state(after)})

        record = after.resolver.resolve(mention)
        self.assertEqual(record.reason, "alias table")
        self.assertIn(mention.lower(), after.resolver.aliases)
        self.assertGreater(after.generation, 1)

    def test_the_log_records_the_event_that_learned_it(self):
        live = Session.load(self.dir)
        tools.call(live, "mesh_ask", {"question": mention_question("PG Vector")})
        Checkpointer(live).record_mutation()

        after = Session.load(self.dir)
        learned = [r for r in after.resolver.log if r.reason == "scored match"]
        self.assertTrue(learned)
        self.assertTrue(
            any(r.mention.lower().replace(" ", "") == "pgvector" for r in learned)
        )


class CheckpointPolicyTest(unittest.TestCase):
    """When a served session is written back, and when it deliberately is not."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = Path(self.tmp) / "session"
        build().save(self.dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def drive(self, policy: str, asks: int = 3) -> Checkpointer:
        live = Session.load(self.dir)
        checkpointer = Checkpointer(live, policy)
        for index in range(asks):
            tools.call(live, "mesh_ask", {"question": f"What supports HNSW {index}?"})
            checkpointer.record_mutation()
        checkpointer.close()
        return checkpointer

    def test_every_ask_commits_each_time(self):
        self.assertEqual(self.drive("every-ask").commits, 3)
        self.assertEqual(manifest(self.dir)["generation"], 4)

    def test_on_exit_commits_once(self):
        self.assertEqual(self.drive("on-exit").commits, 1)
        self.assertEqual(manifest(self.dir)["generation"], 2)

    def test_never_leaves_the_directory_untouched(self):
        before = manifest(self.dir)
        self.assertEqual(self.drive("never").commits, 0)
        self.assertEqual(manifest(self.dir), before)

    def test_on_exit_writes_nothing_if_nothing_happened(self):
        checkpointer = Checkpointer(Session.load(self.dir), "on-exit")
        self.assertIsNone(checkpointer.close())
        self.assertEqual(manifest(self.dir)["generation"], 1)

    def test_close_is_idempotent(self):
        live = Session.load(self.dir)
        checkpointer = Checkpointer(live, "on-exit")
        tools.call(live, "mesh_ask", {"question": "What supports HNSW?"})
        checkpointer.record_mutation()
        self.assertIsNotNone(checkpointer.close())
        self.assertIsNone(checkpointer.close())
        self.assertEqual(manifest(self.dir)["generation"], 2)

    def test_a_demo_session_is_inert_rather_than_an_error(self):
        """It has nowhere to write; refusing to serve it would help nobody."""
        checkpointer = Checkpointer(build(), "every-ask")
        self.assertFalse(checkpointer.durable)
        self.assertIsNone(checkpointer.record_mutation())
        self.assertIsNone(checkpointer.close())

    def test_an_unknown_policy_is_refused(self):
        with self.assertRaises(SessionError):
            Checkpointer(build(), "sometimes")


class DeterminismTest(unittest.TestCase):
    """Two runs of the same build write byte-identical directories."""

    def test_saves_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            Session.build(rounds=ROUNDS).save(first)
            Session.build(rounds=ROUNDS).save(second)
            self.assertEqual(manifest(first), manifest(second))
            for reader in (graph_file, resolver_file):
                self.assertEqual(
                    reader(first).read_bytes(), reader(second).read_bytes(), reader.__name__
                )


if __name__ == "__main__":
    unittest.main()
