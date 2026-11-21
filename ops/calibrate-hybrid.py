#!/usr/bin/env python3
"""Choose the hybrid retriever's parameters on the development suites only.

    python ops/calibrate-hybrid.py            # prints the grid's best rows and the choice

Runs over bench/contextmesh-docs.json and bench/contextmesh-code.json. The
held-out suites (bench/heldout-*.json) are never read here: they exist to
evaluate what this script chooses, so they cannot also be what it chooses on.

Each question's candidates are scored once (BM25 top 10 with every component),
then every grid point re-ranks and thresholds those same candidates, so the
whole grid costs one retrieval pass per question.

Constraint: on each suite, the mean tokens per answerable question must stay
below BM25 top-3's on that suite. Otherwise "return more claims" would win
every time, at a cost the acceptance criteria rule out.

Objective, among grid points meeting it, in order: most correct answers plus
correct out-of-scope refusals across both suites; then fewest wrong answers;
then fewest tokens; then the earliest grid point (a fixed, deterministic
order, smallest values first).
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from contextmesh.bench import Item, build_corpus, is_correct, load_suite  # noqa: E402
from contextmesh.hybrid import HybridRetriever  # noqa: E402
from contextmesh.model import NodeType  # noqa: E402

DEV_SUITES = ("contextmesh-docs.json", "contextmesh-code.json")
MAX_K = 10
GRID = {
    "k": (3, 5, 10),
    "bm25": (0.0, 0.5, 1.0),
    "coverage": (0.0, 0.5, 1.0, 2.0),
    "entity": (0.0, 0.5, 1.0),
    "vector": (0.0, 0.5, 1.0),
    "min_coverage": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    "max_answers": (1, 2, 3),
    "margin": (0.0, 0.1, 0.25, 0.5),
}


def prepare(name):
    path = REPO / "bench" / name
    suite = load_suite(path)
    corpus = build_corpus(suite, base=path.parent)
    built = corpus["built"]
    graph = built.graph
    items = {
        n.id: Item(n.label, n.attrs["path"])
        for n in graph.nodes.values()
        if n.type is NodeType.CLAIM and n.live
    }
    retriever = HybridRetriever(graph, built.resolver, k=MAX_K, min_coverage=0.0)
    tokens = retriever._text_tokens

    def source_cost(claim_id):
        node = graph.node(claim_id)
        sid = node.provenance.source_id if node.provenance else None
        return tokens(graph.node(sid).label) if sid in graph.nodes else 0

    def candidates(question):
        answer = retriever.ask(question)
        rows = []
        entities = set(retriever._question_entities(question))
        for c in answer.candidates:
            # As the primary answer: its path (entity, claim, source). As a
            # runner-up: the claim and its source.
            path = retriever._path(c.claim_id, entities)
            primary = sum(tokens(graph.node(n).label) for _, n in path)
            extra = tokens(graph.node(c.claim_id).label) + source_cost(c.claim_id)
            rows.append((c, primary, extra))
        return rows

    claims, bm25 = retriever._index()
    bm25_top3 = [
        sum(tokens(claims[i].label) for i in bm25.top(q["q"], 3)) for q in suite["answerable"]
    ]
    answerable = [(q, candidates(q["q"])) for q in suite["answerable"]]
    out_of_scope = [candidates(q) for q in suite["out_of_scope"]]
    return {"name": name, "items": items, "answerable": answerable,
            "out_of_scope": out_of_scope, "fingerprint": corpus["fingerprint"],
            "bm25_top3_mean_tokens": sum(bm25_top3) / len(bm25_top3)}


def choose(cands, params):
    """The claims the retriever would return, and their token cost; mirrors
    HybridRetriever.ask so a calibrated setting behaves the same live."""
    pool = cands[: params["k"]]
    if not pool:
        return None

    def value(c):
        return round(
            params["bm25"] * c.bm25 + params["coverage"] * c.coverage
            + params["entity"] * c.entity + params["vector"] * c.vector, 6,
        )

    best = max(pool, key=lambda row: value(row[0]))
    if best[0].coverage < params["min_coverage"]:
        return None
    picked, cost = [best[0].claim_id], best[1]
    for row in sorted(pool, key=lambda r: -value(r[0])):
        if len(picked) >= params["max_answers"]:
            break
        c = row[0]
        if (c.claim_id not in picked and value(c) >= value(best[0]) - params["margin"]
                and c.coverage >= params["min_coverage"]):
            picked.append(c.claim_id)
            cost += row[2]
    return picked, cost


def score(prepared, params):
    correct = wrong = refused = 0
    tokens_ok = True
    total_tokens = 0
    for suite in prepared:
        suite_tokens = 0
        for expected, cands in suite["answerable"]:
            pick = choose(cands, params)
            if pick is None:
                continue
            suite_tokens += pick[1]
            if any(is_correct(suite["items"][cid], expected) for cid in pick[0]):
                correct += 1
            else:
                wrong += 1
        refused += sum(choose(c, params) is None for c in suite["out_of_scope"])
        mean = suite_tokens / len(suite["answerable"])
        tokens_ok = tokens_ok and mean < suite["bm25_top3_mean_tokens"]
        total_tokens += suite_tokens
    return {"correct": correct, "refused": refused, "wrong": wrong,
            "tokens": total_tokens, "under_bm25_top3_tokens": tokens_ok}


def main():
    prepared = [prepare(name) for name in DEV_SUITES]
    keys = list(GRID)
    results = []
    for order, values in enumerate(itertools.product(*(GRID[k] for k in keys))):
        params = dict(zip(keys, values))
        if not any(params[w] for w in ("bm25", "coverage", "entity", "vector")):
            continue
        s = score(prepared, params)
        if not s["under_bm25_top3_tokens"]:
            continue
        results.append(((-(s["correct"] + s["refused"]), s["wrong"], s["tokens"], order),
                        params, s))
    results.sort(key=lambda r: r[0])
    print("dev suites:", ", ".join(
        f"{p['name']} ({p['fingerprint']}, BM25 top-3 mean {p['bm25_top3_mean_tokens']:.1f} "
        "tokens)" for p in prepared))
    print(f"grid points meeting the token constraint: {len(results)}")
    print("\ntop 10 (correct + refused, then fewest wrong, then fewest tokens):")
    for _, params, s in results[:10]:
        print(json.dumps({**params, **s}))
    _, chosen, s = results[0]
    print("\nchosen:", json.dumps(chosen))
    print("on dev:", json.dumps(s))


if __name__ == "__main__":
    main()
