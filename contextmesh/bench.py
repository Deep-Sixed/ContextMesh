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
import shutil
import statistics
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .hybrid import HybridRetriever
from .ingest import PROSE_SUFFIXES, build_folder, iter_files
from .lexical import BM25 as _BM25
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


# BM25 lives in ``lexical`` so the baseline and the hybrid share one
# implementation; re-exported here for callers of the benchmark.
BM25 = _BM25


def _copy_corpus(
    root: Path, exclude: Sequence[str], dest: Path, suffixes: Sequence[str]
) -> List[str]:
    """Copy the corpus's readable files into ``dest``, minus ``exclude``."""
    paths, _ = iter_files(root, suffixes=suffixes)
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


def build_corpus(suite: Dict[str, Any], base: Any = ".") -> Dict[str, Any]:
    """Copy and build a suite's corpus once; ``evaluate`` can then run on it many times."""
    root = (Path(base) / suite["corpus"].get("root", ".")).resolve()
    exclude = list(suite["corpus"].get("exclude", []))
    # A suite names the file types it covers, so a new default (code files)
    # cannot quietly change what an existing suite measures. Prose only when
    # unsaid, which is what suites meant before code was read.
    suffixes = list(suite["corpus"].get("suffixes", sorted(PROSE_SUFFIXES)))
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus"
        rels = _copy_corpus(root, exclude, corpus, suffixes)
        started = time.perf_counter()
        built = build_folder(corpus)
        build_seconds = time.perf_counter() - started
        fingerprint = _fingerprint(corpus, rels)
    return {
        "built": built,
        "files": len(rels),
        "fingerprint": fingerprint,
        "build_seconds": build_seconds,
    }


def run(
    suite: Dict[str, Any], base: Any = ".", hybrid: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run ``suite`` with its corpus root resolved against ``base``.

    ``python -m contextmesh bench`` passes the suite file's own directory, so a
    suite names its corpus relative to itself and runs the same from anywhere.
    ``hybrid`` overrides the hybrid retriever's parameters (k, weights,
    min_coverage); calibration uses it, nothing else should.
    """
    return evaluate(suite, build_corpus(suite, base), hybrid=hybrid)


def evaluate(
    suite: Dict[str, Any], corpus: Dict[str, Any], hybrid: Optional[Dict[str, Any]] = None,
    methods_wanted: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    built = corpus["built"]
    graph, walker = built.graph, built.walker
    cost = walker._text_tokens
    claims = [n for n in graph.nodes.values() if n.type is NodeType.CLAIM and n.live]
    items = [Item(n.label, n.attrs["path"]) for n in claims]
    by_id = {n.id: it for n, it in zip(claims, items)}
    index = BM25([it.label for it in items])

    tracemalloc.start()
    retriever = HybridRetriever(graph, built.resolver, **(hybrid or {}))
    retriever._index()
    hybrid_index_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Each method answers with the items it hands over, whether it answered at
    # all, what it cost, and anything worth keeping about how.
    def walk(question: str) -> Dict[str, Any]:
        w = walker.ask(question)
        node = graph.node(w.answer_id) if w.resolved else None
        item = Item(node.label, node.attrs.get("path", "")) if node else None
        return {
            "items": [item] if item else [], "answered": w.resolved, "tokens": w.tokens_walked,
            "answer": item.label if item else None, "answer_path": item.path if item else None,
            "dead_end": w.dead_end.value if w.dead_end else None, "hops": w.hops,
            "readme_flat_tokens": w.tokens_flat,
        }

    def bm25_at(k: int):
        def answer(question: str) -> Dict[str, Any]:
            hits = [items[i] for i in index.top(question, k)]
            return {"items": hits, "answered": bool(hits),
                    "tokens": sum(cost(h.label) for h in hits)}
        return answer

    def hybrid_answer(question: str) -> Dict[str, Any]:
        a = retriever.ask(question)
        item = by_id.get(a.answer_id) if a.resolved else None
        given = [by_id[i] for i in a.answer_ids if i in by_id]
        return {
            "items": given, "answered": a.resolved, "tokens": a.tokens,
            "claims_returned": len(given),
            "answer": item.label if item else None, "answer_path": item.path if item else None,
            "refused": a.refused, "path_nodes": [n for _, n in a.path],
        }

    answerers = {"walk": walk, **{f"bm25@{k}": bm25_at(k) for k in K_VALUES},
                 "hybrid": hybrid_answer}
    if methods_wanted is not None:
        answerers = {m: f for m, f in answerers.items() if m in methods_wanted}
    latency: Dict[str, List[float]] = {m: [] for m in answerers}

    def ask(method: str, question: str) -> Dict[str, Any]:
        started = time.perf_counter()
        out = answerers[method](question)
        latency[method].append(time.perf_counter() - started)
        return out

    rows: List[Dict[str, Any]] = []
    for expected in suite["answerable"]:
        row: Dict[str, Any] = {
            "q": expected["q"],
            "covered": any(is_correct(it, expected) for it in items),
        }
        for method in answerers:
            out = ask(method, expected["q"])
            got = out.pop("items")
            out["correct"] = any(is_correct(it, expected) for it in got)
            row[method] = out
        rows.append(row)

    refusals = []
    for question in suite["out_of_scope"]:
        row = {"q": question}
        for method in answerers:
            out = ask(method, question)
            row[f"{method}_refused"] = not out["answered"]
            if method == "walk":
                row["dead_end"] = out["dead_end"]
        # Kept under the names earlier results used.
        if "bm25@1" in answerers:
            row["bm25_returned"] = not row["bm25@1_refused"]
        refusals.append(row)

    everything = sum(cost(it.label) for it in items)
    n = len(rows)
    covered = sum(r["covered"] for r in rows)

    methods: Dict[str, Any] = {}
    for method in answerers:
        correct = sum(r[method]["correct"] for r in rows)
        toks = [r[method]["tokens"] for r in rows]
        methods[method] = {
            "correct": correct,
            "accuracy": round(correct / n, 3) if n else 0.0,
            "accuracy_of_covered": round(correct / covered, 3) if covered else 0.0,
            "answered": sum(r[method]["answered"] for r in rows),
            # Answered, and wrong: the failure an answer path makes look right.
            "wrong": sum(r[method]["answered"] and not r[method]["correct"] for r in rows),
            "mean_tokens": round(statistics.mean(toks), 1) if toks else 0.0,
            "median_tokens": statistics.median(toks) if toks else 0,
            "refused_out_of_scope": sum(r[f"{method}_refused"] for r in refusals),
            "mean_latency_ms": round(1000 * statistics.mean(latency[method]), 2)
            if latency[method] else 0.0,
        }
    if "walk" in methods:
        walked = [r for r in rows if r["walk"]["answered"]]
        methods["walk"]["median_hops"] = (
            statistics.median([r["walk"]["hops"] for r in walked]) if walked else 0
        )
    if "hybrid" in methods:
        methods["hybrid"]["index_peak_bytes"] = hybrid_index_bytes
        methods["hybrid"]["params"] = {
            "k": retriever.k, "weights": retriever.weights,
            "min_coverage": retriever.min_coverage,
            "max_answers": retriever.max_answers, "margin": retriever.margin,
        }
    methods["read_everything"] = {
        "correct": covered,
        "accuracy": round(covered / n, 3) if n else 0.0,
        "accuracy_of_covered": 1.0 if covered else 0.0,
        "answered": n,
        "wrong": n - covered,
        "mean_tokens": everything,
        "median_tokens": everything,
        "refused_out_of_scope": 0,
        "mean_latency_ms": 0.0,
    }
    flat = [r["walk"]["readme_flat_tokens"] for r in rows if "walk" in r]

    return {
        "suite": suite["name"],
        "corpus": {
            "files": corpus["files"],
            "fingerprint": corpus["fingerprint"],
            "claims": len(items),
            "tokens": everything,
            "build_seconds": round(corpus["build_seconds"], 2),
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
        "| Method | Correct | Of covered | Wrong answers | Mean tokens / question "
        "| Out-of-scope refused | Mean latency |",
        "|---|---|---|---|---|---|---|",
    ]
    names = {
        "hybrid": "Hybrid (BM25 + graph)",
        "walk": "Graph walk",
        "bm25@1": "BM25 top-1",
        "bm25@3": "BM25 top-3",
        "bm25@5": "BM25 top-5",
        "read_everything": "Read everything",
    }
    for key, label in names.items():
        m = result["methods"].get(key)
        if m is None:
            continue
        lines.append(
            f"| {label} | {m['correct']}/{q['answerable']} ({m['accuracy']:.0%}) | "
            f"{m['correct']}/{q['covered']} ({m['accuracy_of_covered']:.0%}) | "
            f"{m['wrong']} | {m['mean_tokens']:,} | "
            f"{m['refused_out_of_scope']}/{q['out_of_scope']} | "
            f"{m['mean_latency_ms']:.1f} ms |"
        )
    w = result["methods"].get("walk", {"answered": 0, "median_hops": 0})
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
