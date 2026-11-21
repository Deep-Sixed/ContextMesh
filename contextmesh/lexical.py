"""Lexical retrieval over claim sentences: BM25, and how much of a question a text covers.

Shared by the benchmark (as a baseline) and the hybrid retriever (as its
candidate stage), so both measure and use exactly the same thing.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Sequence, Tuple

from .embed import tokens


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
        # A term no text contains is as rare as a term can be: the idf of a
        # term seen in zero documents.
        self.unseen_idf = math.log(1 + (n + 0.5) / 0.5) if n else 0.0
        self.tf = [Counter(d) for d in self.docs]

    def scored(self, query: str, k: int) -> List[Tuple[int, float]]:
        """The ``k`` best ``(index, score)`` pairs with a positive score."""
        q = tokens(query)
        results = []
        for i, (tf, doc) in enumerate(zip(self.tf, self.docs)):
            score = 0.0
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                norm = f + self.k1 * (1 - self.b + self.b * len(doc) / (self.avgdl or 1))
                score += self.idf[term] * f * (self.k1 + 1) / norm
            if score > 0:
                results.append((-score, i))
        results.sort()
        return [(i, -s) for s, i in results[:k]]

    def top(self, query: str, k: int) -> List[int]:
        return [i for i, _ in self.scored(query, k)]

    def coverage(self, query: str, index: int) -> float:
        """The idf-weighted share of the query's terms that text ``index`` contains.

        An absolute measure in [0, 1], unlike a BM25 score: 1 means every term
        of the question is in the text, weighted by how telling each term is.
        Terms the corpus has never seen count at the highest idf, so a question
        about something the corpus does not cover scores low even when its
        common words match.
        """
        q = set(tokens(query))
        if not q:
            return 0.0
        present = self.tf[index]
        total = sum(self.idf.get(t, self.unseen_idf) for t in q)
        hit = sum(self.idf[t] for t in q if t in present)
        return hit / total if total else 0.0
