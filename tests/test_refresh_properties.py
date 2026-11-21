"""Phase 2, slice 4: properties of refresh over random edit sequences.

Each sequence is generated from a fixed seed, so a failure names the seed and
step that reproduce it. After every refresh the graph is checked against the
invariants the design promises (docs/PHASE2_INCREMENTAL_REBUILDS.md,
"Verification"), and against a fresh build of the same folder.
"""

import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from contextmesh.decisions import DecisionLog
from contextmesh.graph import ContextGraph
from contextmesh.ingest import CLAIM_ASSUMPTION_AUTHOR, build_folder
from contextmesh.model import AssumptionStatus, EdgeType, NodeType
from contextmesh.refresh import refresh_folder
from contextmesh.resolve import normalise

REPO = Path(__file__).resolve().parent.parent
BASE_TIME = 1767225600  # 2026-01-01T00:00:00Z
DAY = 86400

#: Sentences a file can say. Capitalised names recur across sentences so that
#: entities appear, qualify and are shared as files change.
SENTENCES = [
    "The Index Builder writes shards every night.",
    "The Index Builder retries a failed shard three times.",
    "The Retention Window keeps shards for thirty days.",
    "Audit shortened the Retention Window this quarter.",
    "The cache is keyed per tenant.",
    "The cache holds entries for an hour.",
    "Entries are evicted on every deploy.",
    "The Shard Store keeps every shard it is given.",
    "The Shard Store is read by the Index Builder.",
    "Tenants are billed by the Metering Service each month.",
    "The Metering Service reads usage from the Shard Store.",
    "Deploys are frozen during the audit window.",
]
FILES = ["a.md", "b.md", "c.md", "d.md", "notes/e.md"]


def render(lines):
    return "# Notes\n\n" + "\n".join(lines) + "\n"


class Folder:
    """A folder whose files, contents and modification times a test controls."""

    def __init__(self, root: Path, rng: random.Random):
        self.root = root
        self.rng = rng
        self.day = 0
        self.history = {}  # path -> earlier contents, for restores

    def write(self, rel, lines):
        self.day += 1
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(render(lines).encode("utf-8"))
        stamp = BASE_TIME + self.day * DAY
        os.utime(path, (stamp, stamp))
        self.history.setdefault(rel, []).append(list(lines))

    def present(self):
        return [f for f in FILES if (self.root / f).exists()]

    def lines_of(self, rel):
        text = (self.root / rel).read_text(encoding="utf-8")
        return [line for line in text.splitlines()[2:] if line]

    def random_step(self):
        """Apply one random change; return a description for failure messages."""
        rng, present = self.rng, self.present()
        absent = [f for f in FILES if f not in present]
        choices = ["edit", "edit", "touch"]
        if absent:
            choices.append("add")
        if len(present) > 1:
            choices.append("delete")
        if any(len(self.history.get(f, [])) > 1 for f in present):
            choices.append("restore")
        action = rng.choice(choices)
        if action == "add":
            rel = rng.choice(absent)
            self.write(rel, rng.sample(SENTENCES, rng.randint(1, 4)))
        elif action == "delete":
            rel = rng.choice(present)
            (self.root / rel).unlink()
        elif action == "touch":
            rel = rng.choice(present)
            self.day += 1
            stamp = BASE_TIME + self.day * DAY
            os.utime(self.root / rel, (stamp, stamp))
        elif action == "restore":
            rel = rng.choice([f for f in present if len(self.history.get(f, [])) > 1])
            self.write(rel, rng.choice(self.history[rel][:-1]))
        else:
            rel = rng.choice(present)
            lines = self.lines_of(rel)
            if lines and rng.random() < 0.5:
                lines.pop(rng.randrange(len(lines)))
            if rng.random() < 0.7:
                lines.insert(rng.randint(0, len(lines)), rng.choice(SENTENCES))
            if rng.random() < 0.3:
                rng.shuffle(lines)
            self.write(rel, lines or [rng.choice(SENTENCES)])
        return f"{action} {rel}"

    def as_of(self):
        return f"2026-{1 + (self.day // 28) % 12:02d}-{1 + self.day % 28:02d}"


def live_claims(graph):
    return sorted(
        (n.attrs["path"], n.label) for n in graph.nodes.values()
        if n.type is NodeType.CLAIM and n.live
    )


def claim_mentions(graph):
    """(path, sentence) -> ids of the entities that live claim mentions."""
    out = {}
    for node in graph.nodes.values():
        if node.type is NodeType.CLAIM and node.live:
            key = (node.attrs["path"], node.label)
            for edge in graph.out_edges(node.id, [EdgeType.MENTIONS]):
                out.setdefault(key, set()).add(edge.dst)
    return out


def entity_forms(resolver):
    """entity id -> every normalised name the resolver files under it."""
    forms = {eid: {normalise(label)} for eid, label in resolver.canonical.items()}
    for key, eid in resolver.aliases.items():
        forms.setdefault(eid, set()).add(key)
    return forms


class RandomEditSequenceTest(unittest.TestCase):
    SEEDS = range(20)
    STEPS = 8

    def check_invariants(self, graph, resolver, folder, decisions, where):
        # 1. The live view is what a fresh build of the folder would extract.
        fresh_build = build_folder(folder.root) if folder.present() else None
        fresh = fresh_build.graph if fresh_build is not None else None
        expected = live_claims(fresh) if fresh is not None else []
        self.assertEqual(live_claims(graph), expected, where)

        # 2. Every mention a fresh build makes is in the refreshed graph too,
        #    compared by the names an entity answers to rather than its id or
        #    label: when several names fold into one entity, whichever was
        #    registered first names it, and a refresh registers in history
        #    order where a fresh build registers in file order. (The refreshed
        #    graph may hold more: an entity that once qualified is never
        #    removed, so its old links stay.)
        if fresh is not None:
            ours, theirs = entity_forms(resolver), entity_forms(fresh_build.resolver)
            have = claim_mentions(graph)
            for key, entity_ids in claim_mentions(fresh).items():
                for fresh_id in entity_ids:
                    self.assertTrue(
                        any(ours[mine] & theirs[fresh_id] for mine in have.get(key, ())),
                        (where, key, fresh_build.resolver.canonical[fresh_id]),
                    )

        # 3. Every live claim rests on exactly one active "file still says"
        #    assumption, naming its own file.
        for node in graph.nodes.values():
            if node.type is not NodeType.CLAIM or not node.live:
                continue
            active = [
                graph.assumptions[e.dst]
                for e in graph.out_edges(node.id, [EdgeType.DEPENDS_ON])
                if graph.assumptions[e.dst].created_by == CLAIM_ASSUMPTION_AUTHOR
            ]
            self.assertEqual(len(active), 1, (where, node.label))
            self.assertIs(active[0].status, AssumptionStatus.ACTIVE, where)
            self.assertTrue(active[0].statement.startswith(node.attrs["path"] + "@"), where)

        # 4. Every retired claim was retired by evidence, through its assumption.
        for node in graph.nodes.values():
            if node.type is NodeType.CLAIM and node.invalidated:
                deps = graph.out_edges(node.id, [EdgeType.DEPENDS_ON], live_only=False)
                rejected = [
                    graph.assumptions[e.dst] for e in deps
                    if graph.assumptions[e.dst].status is AssumptionStatus.REJECTED
                ]
                self.assertTrue(rejected, (where, node.label))
                witnesses = graph.in_edges(
                    rejected[0].id, [EdgeType.CONTRADICTS], live_only=False
                )
                self.assertTrue(
                    all(graph.nodes[w.src].type is NodeType.EVIDENCE for w in witnesses)
                    and witnesses,
                    where,
                )

        # 5. Exactly one current source version per file present, none for
        #    files that are gone; no source is ever invalidated.
        current = {}
        for node in graph.nodes.values():
            if node.type is NodeType.SOURCE:
                self.assertFalse(node.invalidated, (where, node.id))
                if node.attrs.get("current"):
                    self.assertNotIn(node.attrs["path"], current, where)
                    current[node.attrs["path"]] = node
        self.assertEqual(sorted(current), sorted(folder.present()), where)

        # 6. Selectivity: a decision stands exactly when every claim it was
        #    derived from stands.
        for decision_id, claim_ids in decisions.items():
            expected_live = all(graph.nodes[c].live for c in claim_ids)
            self.assertEqual(graph.nodes[decision_id].live, expected_live, (where, decision_id))

        # 7. Structural: no untyped edges, and the snapshot round-trips.
        self.assertEqual(graph.untyped_edges, 0, where)
        self.assertEqual(ContextGraph.from_dict(graph.to_dict()).to_json(), graph.to_json(), where)

    def run_seed(self, seed):
        rng = random.Random(seed)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Folder(Path(tmp) / "notes", rng)
            for rel in rng.sample(FILES, 3):
                folder.write(rel, rng.sample(SENTENCES, rng.randint(2, 4)))
            built = build_folder(folder.root)
            graph, log = built.graph, DecisionLog(built.graph)
            decisions = {}
            for step in range(self.STEPS):
                live = [n for n in graph.nodes.values() if n.type is NodeType.CLAIM and n.live]
                if live and rng.random() < 0.6:
                    picked = rng.sample(live, min(len(live), rng.randint(1, 2)))
                    node = log.decide(
                        f"decision {seed}.{step}", "test",
                        source_id=picked[0].provenance.source_id,
                        supported_by=[c.id for c in picked],
                    )
                    decisions[node.id] = [c.id for c in picked]
                change = folder.random_step()
                where = f"seed {seed}, step {step}: {change}"
                report = refresh_folder(
                    graph, built.resolver, built.ledger, folder.root, as_of=folder.as_of()
                )
                # What fell beyond the claims is exactly the decisions that lost
                # a claim in this refresh.
                fell = {d for d, cs in decisions.items() if not graph.nodes[d].live}
                self.assertTrue(set(report.downstream) <= fell, where)
                self.check_invariants(graph, built.resolver, folder, decisions, where)

    def test_random_edit_sequences_keep_every_invariant(self):
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                self.run_seed(seed)


SEQUENCE_SCRIPT = r"""
import hashlib, os, random, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from tests.test_refresh_properties import Folder
from contextmesh.ingest import build_folder
from contextmesh.refresh import refresh_folder

rng = random.Random(7)
folder = Folder(Path(sys.argv[1]) / "notes", rng)
for rel in ("a.md", "b.md", "c.md"):
    folder.write(rel, rng.sample(__import__("tests.test_refresh_properties",
        fromlist=["SENTENCES"]).SENTENCES, 3))
built = build_folder(folder.root)
for _ in range(10):
    folder.random_step()
    refresh_folder(built.graph, built.resolver, built.ledger, folder.root,
                   as_of=folder.as_of())
blob = built.graph.to_json() + built.resolver.to_json()
print(hashlib.sha256(blob.encode()).hexdigest())
"""


class HashSeedTest(unittest.TestCase):
    def test_a_refresh_sequence_is_identical_under_any_hash_seed(self):
        digests = set()
        for seed in ("0", "4242"):
            with tempfile.TemporaryDirectory() as tmp:
                result = subprocess.run(
                    [sys.executable, "-c", SEQUENCE_SCRIPT, tmp, str(REPO)],
                    cwd=REPO, env=dict(os.environ, PYTHONHASHSEED=seed),
                    capture_output=True, text=True, check=True,
                )
                digests.add(result.stdout.strip())
        self.assertEqual(len(digests), 1, digests)


class PersistenceRoundTripTest(unittest.TestCase):
    """Refresh, save, load, refresh again == both refreshes in memory."""

    def test_refreshing_through_saved_generations_matches_staying_in_memory(self):
        from contextmesh_mcp.session import Session

        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            folders = []
            for tmp in (one, two):
                folder = Folder(Path(tmp) / "notes", random.Random(11))
                for rel in ("a.md", "b.md", "c.md"):
                    folder.write(rel, folder.rng.sample(SENTENCES, 3))
                folders.append(folder)

            memory = build_folder(folders[0].root)
            saved = Session.from_folder(folders[1].root)
            home = Path(two) / "session"
            saved.save(home)

            for step in range(6):
                for folder in folders:
                    folder.random_step()
                refresh_folder(memory.graph, memory.resolver, memory.ledger,
                               folders[0].root, as_of=folders[0].as_of())
                loaded = Session.load(home)
                report = refresh_folder(loaded.graph, loaded.resolver, loaded.ledger,
                                        folders[1].root, as_of=folders[1].as_of())
                if report.changed:
                    loaded.save(home)

            final = Session.load(home)
            self.assertEqual(final.graph.to_json(), memory.graph.to_json())
            self.assertEqual(
                json.loads(final.resolver.to_json())["canonical"],
                json.loads(memory.resolver.to_json())["canonical"],
            )


class SequenceDigestTest(unittest.TestCase):
    def test_two_runs_of_the_same_sequence_give_the_same_graph(self):
        def run(tmp):
            folder = Folder(Path(tmp) / "notes", random.Random(3))
            for rel in ("a.md", "b.md"):
                folder.write(rel, folder.rng.sample(SENTENCES, 3))
            built = build_folder(folder.root)
            for _ in range(10):
                folder.random_step()
                refresh_folder(built.graph, built.resolver, built.ledger, folder.root,
                               as_of=folder.as_of())
            return hashlib.sha256(built.graph.to_json().encode()).hexdigest()

        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            self.assertEqual(run(one), run(two))


if __name__ == "__main__":
    unittest.main()
