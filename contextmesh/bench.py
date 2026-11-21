"""Benchmark question answering over a folder: correctness and token cost.

The figures this project has quoted ("60 tokens against 7,300") compare a walk's
path with a fixed budget: ``Walker.tokens_flat`` is 40 claims sampled by a hash
of the question, plus a 160-token window each, whatever the question asks. It
never checks whether that sample holds the answer, and nothing checked whether
the walk's answer was right. This module measures both, on labelled questions:

* **graph walk** — ``Walker.ask``; correct when the answer claim comes from an
  accepted file and contains an expected phrase; on an out-of-scope question,
  correct when it refuses (a dead end) instead of answering.
* **BM25 top-k** — a plain lexical baseline over the *same* claim sentences,
  for k in 1, 3, 5; correct when any of the k contains the answer. It never
  refuses, so it is wrong on every out-of-scope question by construction.
* **read everything** — every claim in the corpus, at the full token cost.
  Correct exactly when *some* extracted claim answers the question, which is
  also the ceiling for every method here: a sentence the extractor dropped (no
  verb, under four words, inside a table) cannot be returned by any of them.
  That ceiling is reported as ``covered``, and each method's accuracy is also
  given against it, so extraction gaps and retrieval misses stay separate.

Every method is costed with the same counter, the walker's own
(``words × 1.3 + 8`` per text). The walk is charged for every node on its
path, as the walker charges itself. The README's "flat" figure is reported
too, labelled for what it is.

Stdlib only, deterministic: the same corpus and questions give the same
result. The corpus is fingerprinted so a result can be tied to what it ran on.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .embed import tokens
from .ingest import build_folder, iter_files
from .model import NodeType

K_VALUES = (1, 3, 5)


@dataclass
class Item:
    """A claim the methods can return: its text and the file it came from."""

    label: str
    path: str


def load_suite(path: Any) -> Dict[str, Any]:
    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("name", "corpus", "answerable", "out_of_scope"):
        if key not in suite:
            raise ValueError(f"benchmark suite {path} has no {key!r}")
    return suite


def is_correct(item: Optional[Item], expected: Dict[str, Any]) -> bool:
    """An answer is right if it comes from an accepted file and says the thing."""
    if item is None or item.path not in expected["paths"]:
        return False
    text = item.label.lower()
    return any(phrase.lower() in text for phrase in expected["any"])


class BM25:
    """Okapi BM25 over short texts; ties broken by position, so deterministic."""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = [tokens(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = (sum(len(d) for d in self.docs) / len(self.docs)) if self.docs else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def top(self, query: str, k: int) -> List[int]:
        q = tokens(query)
        scored = []
        for i, (tf, doc) in enumerate(zip(self.tf, self.docs)):
            score = 0.0
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                norm = f + self.k1 * (1 - self.b + self.b * len(doc) / (self.avgdl or 1))
                score += self.idf[term] * f * (self.k1 + 1) / norm
            if score > 0:
                scored.append((-score, i))
        scored.sort()
        return [i for _, i in scored[:k]]


def _copy_corpus(root: Path, exclude: Sequence[str], dest: Path) -> List[str]:
    """Copy the corpus's readable files into ``dest``, minus ``exclude``."""
    paths, _ = iter_files(root)
    kept = []
    for path in paths:
        rel = path.relative_to(root).as_posix()
        if rel in exclude:
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        kept.append(rel)
    return kept


def _fingerprint(root: Path, rels: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(rels):
        digest.update(rel.encode())
        digest.update(hashlib.sha256((root / rel).read_bytes()).digest())
    return digest.hexdigest()[:16]


def run(suite: Dict[str, Any], base: Any = ".") -> Dict[str, Any]:
    """Run ``suite`` with its corpus root resolved against ``base``.

    ``python -m contextmesh bench`` passes the suite file's own directory, so a
    suite names its corpus relative to itself and runs the same from anywhere.
    """
    root = (Path(base) / suite["corpus"].get("root", ".")).resolve()
    exclude = list(suite["corpus"].get("exclude", []))
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus"
        rels = _copy_corpus(root, exclude, corpus)
        started = time.perf_counter()
        built = build_folder(corpus)
        build_seconds = time.perf_counter() - started
        fingerprint = _fingerprint(corpus, rels)

    graph, walker = built.graph, built.walker
    cost = walker._text_tokens
    items = [
        Item(n.label, n.attrs["path"])
        for n in graph.nodes.values()
        if n.type is NodeType.CLAIM and n.live
    ]
    index = BM25([it.label for it in items])

    def walk_answer(question: str) -> Tuple[Optional[Item], Any]:
        walk = walker.ask(question)
        if not walk.resolved:
            return None, walk
        node = graph.node(walk.answer_id)
        if node.type is not NodeType.CLAIM:
            # A decision or entity answer has no file of its own.
            return Item(node.label, node.attrs.get("path", "")), walk
        return Item(node.label, node.attrs["path"]), walk

    rows: List[Dict[str, Any]] = []
    for expected in suite["answerable"]:
        item, walk = walk_answer(expected["q"])
        covered = any(is_correct(it, expected) for it in items)
        row = {
            "q": expected["q"],
            "covered": covered,
            "walk": {
                "correct": is_correct(item, expected),
                "answered": walk.resolved,
                "answer": item.label if item else None,
                "answer_path": item.path if item else None,
                "dead_end": walk.dead_end.value if walk.dead_end else None,
                "hops": walk.hops,
                "tokens": walk.tokens_walked,
                "readme_flat_tokens": walk.tokens_flat,
            },
        }
        for k in K_VALUES:
            hits = [items[i] for i in index.top(expected["q"], k)]
            row[f"bm25@{k}"] = {
                "correct": any(is_correct(h, expected) for h in hits),
                "tokens": sum(cost(h.label) for h in hits),
            }
        rows.append(row)

    refusals = []
    for question in suite["out_of_scope"]:
        _, walk = walk_answer(question)
        refusals.append({
            "q": question,
            "walk_refused": not walk.resolved,
            "dead_end": walk.dead_end.value if walk.dead_end else None,
            "bm25_returned": bool(index.top(question, 1)),
        })

    everything = sum(cost(it.label) for it in items)
    n = len(rows)
    covered = sum(r["covered"] for r in rows)

    def summarise(method: str) -> Dict[str, Any]:
        correct = sum(r[method]["correct"] for r in rows)
        toks = [r[method]["tokens"] for r in rows]
        return {
            "correct": correct,
            "accuracy": round(correct / n, 3) if n else 0.0,
            "accuracy_of_covered": round(correct / covered, 3) if covered else 0.0,
            "mean_tokens": round(statistics.mean(toks), 1) if toks else 0.0,
            "median_tokens": statistics.median(toks) if toks else 0,
        }

    methods: Dict[str, Any] = {"walk": summarise("walk")}
    walked = [r for r in rows if r["walk"]["answered"]]
    methods["walk"]["answered"] = len(walked)
    methods["walk"]["median_hops"] = (
        statistics.median([r["walk"]["hops"] for r in walked]) if walked else 0
    )
    methods["walk"]["refused_out_of_scope"] = sum(r["walk_refused"] for r in refusals)
    for k in K_VALUES:
        methods[f"bm25@{k}"] = summarise(f"bm25@{k}")
        methods[f"bm25@{k}"]["refused_out_of_scope"] = sum(
            not r["bm25_returned"] for r in refusals
        )
    methods["read_everything"] = {
        "correct": covered,
        "accuracy": round(covered / n, 3) if n else 0.0,
        "accuracy_of_covered": 1.0 if covered else 0.0,
        "mean_tokens": everything,
        "median_tokens": everything,
        "refused_out_of_scope": 0,
    }
    flat = [r["walk"]["readme_flat_tokens"] for r in rows]

    return {
        "suite": suite["name"],
        "corpus": {
            "files": len(rels),
            "fingerprint": fingerprint,
            "claims": len(items),
            "tokens": everything,
            "build_seconds": round(build_seconds, 2),
        },
        "questions": {
            "answerable": n,
            "covered": covered,
            "out_of_scope": len(refusals),
            "not_covered": [r["q"] for r in rows if not r["covered"]],
        },
        "methods": methods,
        "readme_flat_metric": {
            "about": "Walker.tokens_flat: 40 claims sampled by question hash + 160 tokens "
                     "of window each; not a retriever, never checked for the answer",
            "mean_tokens": round(statistics.mean(flat), 1) if flat else 0.0,
        },
        "rows": rows,
        "refusals": refusals,
    }


def render(result: Dict[str, Any]) -> str:
    """A Markdown summary of a result."""
    c, q = result["corpus"], result["questions"]
    lines = [
        f"Suite `{result['suite']}`: {c['files']} files, {c['claims']} claims, "
        f"{c['tokens']:,} tokens (corpus `{c['fingerprint']}`, built in {c['build_seconds']} s). "
        f"{q['answerable']} answerable questions, of which {q['covered']} have an answer "
        f"in the extracted claims (the ceiling for every method); "
        f"{q['out_of_scope']} out of scope.",
        "",
        "| Method | Correct | Of covered | Mean tokens / question | Out-of-scope refused |",
        "|---|---|---|---|---|",
    ]
    names = {
        "walk": "Graph walk",
        "bm25@1": "BM25 top-1",
        "bm25@3": "BM25 top-3",
        "bm25@5": "BM25 top-5",
        "read_everything": "Read everything",
    }
    for key, label in names.items():
        m = result["methods"][key]
        lines.append(
            f"| {label} | {m['correct']}/{q['answerable']} ({m['accuracy']:.0%}) | "
            f"{m['correct']}/{q['covered']} ({m['accuracy_of_covered']:.0%}) | "
            f"{m['mean_tokens']:,} | {m['refused_out_of_scope']}/{q['out_of_scope']} |"
        )
    w = result["methods"]["walk"]
    if q["not_covered"]:
        lines += ["", "Not answerable from the extracted claims: "
                  + "; ".join(f"\"{x}\"" for x in q["not_covered"]) + "."]
    lines += [
        "",
        f"The walk answered {w['answered']}/{q['answerable']} answerable questions "
        f"(median {w['median_hops']} hops). The README's \"flat\" figure averaged "
        f"{result['readme_flat_metric']['mean_tokens']:,} tokens: "
        f"{result['readme_flat_metric']['about']}.",
    ]
    return "\n".join(lines)
