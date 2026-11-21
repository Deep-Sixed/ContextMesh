"""The benchmark scores what it says it scores."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from contextmesh import cli
from contextmesh.bench import BM25, Item, is_correct, load_suite, render, run

REPO = Path(__file__).resolve().parent.parent

CACHE = """# Cache

The Cache Layer is keyed per tenant.
The Cache Layer holds entries for an hour.
"""
INDEX = """# Index

The Index Builder writes shards every night.
The Index Builder retries a failed shard three times.
"""
SECRET = "# Secret\n\nThe launch code is swordfish today.\n"


def suite(**overrides):
    base = {
        "name": "tiny",
        "corpus": {"root": "corpus", "exclude": ["secret.md"]},
        "answerable": [
            {"q": "How long does the Cache Layer hold entries?", "paths": ["cache.md"],
             "any": ["an hour"]},
            {"q": "How often does the Index Builder retry a failed shard?",
             "paths": ["index.md"], "any": ["three times"]},
            {"q": "What is the launch code?", "paths": ["secret.md"], "any": ["swordfish"]},
        ],
        "out_of_scope": ["What is the capital of Australia?"],
    }
    base.update(overrides)
    return base


class ScoringTest(unittest.TestCase):
    def test_an_answer_must_come_from_an_accepted_file_and_say_the_thing(self):
        expected = {"paths": ["a.md"], "any": ["an hour", "60 minutes"]}
        self.assertTrue(is_correct(Item("Entries last AN HOUR.", "a.md"), expected))
        self.assertFalse(is_correct(Item("Entries last an hour.", "b.md"), expected))
        self.assertFalse(is_correct(Item("Entries last a day.", "a.md"), expected))
        self.assertFalse(is_correct(None, expected))

    def test_bm25_ranks_the_matching_sentence_first_and_is_deterministic(self):
        docs = [
            "The cache is keyed per tenant.",
            "The Index Builder retries a failed shard three times.",
            "Shards are written every night.",
        ]
        index = BM25(docs)
        self.assertEqual(index.top("How often does the builder retry a shard?", 1), [1])
        self.assertEqual(index.top("retry shard", 3), index.top("retry shard", 3))

    def test_bm25_returns_nothing_when_no_word_matches(self):
        self.assertEqual(BM25(["The cache is keyed per tenant."]).top("capital Australia", 5), [])


class RunTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        corpus = self.base / "corpus"
        corpus.mkdir()
        for name, text in (("cache.md", CACHE), ("index.md", INDEX), ("secret.md", SECRET)):
            (corpus / name).write_bytes(text.encode("utf-8"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_excluded_files_are_not_in_the_corpus(self):
        result = run(suite(), base=self.base)
        self.assertEqual(result["corpus"]["files"], 2)
        launch = next(r for r in result["rows"] if "launch" in r["q"])
        self.assertFalse(launch["covered"])
        self.assertIn("What is the launch code?", result["questions"]["not_covered"])

    def test_read_everything_scores_the_coverage_ceiling_not_every_question(self):
        result = run(suite(), base=self.base)
        self.assertEqual(result["questions"]["covered"], 2)
        everything = result["methods"]["read_everything"]
        self.assertEqual(everything["correct"], 2)
        self.assertEqual(everything["accuracy_of_covered"], 1.0)
        for method, scores in result["methods"].items():
            self.assertLessEqual(scores["correct"], 2, method)

    def test_bm25_never_refuses_and_the_walk_can(self):
        result = run(suite(out_of_scope=["What does the Cache Layer cost?"]), base=self.base)
        self.assertEqual(result["methods"]["bm25@1"]["refused_out_of_scope"], 0)
        self.assertIn("walk_refused", result["refusals"][0])

    def test_the_same_inputs_give_the_same_result(self):
        first, second = run(suite(), base=self.base), run(suite(), base=self.base)
        for result in (first, second):
            result["corpus"].pop("build_seconds")
        self.assertEqual(first, second)

    def test_render_reports_coverage_and_every_method(self):
        text = render(run(suite(), base=self.base))
        for label in ("Graph walk", "BM25 top-1", "BM25 top-5", "Read everything"):
            self.assertIn(label, text)
        self.assertIn("2 have an answer in the extracted claims", text)


class RepositorySuiteTest(unittest.TestCase):
    """The shipped suite stays valid as the docs change."""

    def test_the_repo_suite_names_files_that_exist_and_no_duplicate_questions(self):
        path = REPO / "bench" / "contextmesh-docs.json"
        data = load_suite(path)
        root = (path.parent / data["corpus"]["root"]).resolve()
        questions = [q["q"] for q in data["answerable"]] + data["out_of_scope"]
        self.assertEqual(len(questions), len(set(questions)))
        for expected in data["answerable"]:
            self.assertTrue(expected["paths"] and expected["any"], expected["q"])
            for rel in expected["paths"]:
                self.assertTrue((root / rel).is_file(), rel)
                self.assertNotIn(rel, data["corpus"]["exclude"], rel)

    def test_the_cli_refuses_a_missing_suite(self):
        err = io.StringIO()
        from unittest import mock

        with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", err):
            self.assertEqual(cli.main(["bench", "--suite", "/no/such/suite.json"]), 2)
        self.assertIn("no benchmark suite", err.getvalue())

    def test_the_cli_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus"
            corpus.mkdir()
            (corpus / "cache.md").write_bytes(CACHE.encode("utf-8"))
            (corpus / "index.md").write_bytes(INDEX.encode("utf-8"))
            suite_path = Path(tmp) / "suite.json"
            suite_path.write_text(json.dumps(suite(corpus={"root": "corpus", "exclude": []})))
            out = Path(tmp) / "result.json"
            with redirect_stdout(io.StringIO()):
                code = cli.main(["bench", "--suite", str(suite_path), "--json", str(out)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.read_text())["suite"], "tiny")


if __name__ == "__main__":
    unittest.main()
