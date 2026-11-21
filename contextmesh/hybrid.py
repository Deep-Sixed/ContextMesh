"""Hybrid retrieval: BM25 finds candidate claims, the graph decides among them.

The benchmark (``docs/BENCHMARKS.md``) showed lexical search and the graph walk
getting *different* questions right, and BM25 often holding the right claim in
its top few without ranking it first. So this does not replace either:

1. **Candidates.** BM25 over the graph's *live* claims returns the top ``k``.
   A retired claim is not in the index, so it can never come back as current
   evidence.
2. **Re-ranking.** Each candidate is scored by BM25's relative score, how much
   of the question it covers, whether it mentions the entities the question
   names (the graph's signal), and embedding similarity. The weights are fixed
   numbers, chosen on the development suites only; see ``ops/calibrate-hybrid.py``.
3. **Evidence, or a reason.** The best candidate is answered with its path in
   the graph — the question's entity it mentions, the claim, the source it was
   lifted from — or the question is refused when the best candidate covers too
   little of it. A path is never treated as proof: refusal looks at what the
   claim says, not at whether it is connected.
4. **Close calls carry their runners-up.** When other candidates score within
   ``margin`` of the best (and clear the same coverage bar), up to
   ``max_answers`` claims are returned, each with its source, and every one is
   paid for in tokens. An unambiguous question gets one claim; an ambiguous one
   gets the two or three it is ambiguous between, rather than a confident
   guess.

Read-only: asking resolves names with ``Resolver.match``, which neither logs nor
learns, so one question cannot change the answer to the next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .embed import cosine, embed
from .graph import ContextGraph
from .lexical import BM25
from .model import EdgeType, Node, NodeType
from .pipeline import _candidate_mentions
from .resolve import Resolver

#: Chosen by ops/calibrate-hybrid.py on the development suites only (its
#: output is committed as ops/calibrate-hybrid.log). The graph's own
#: re-ranking signals — entity overlap and embedding similarity — earned no
#: weight there; they stay available, at 0, so a corpus where they help can be
#: calibrated without code changes.
DEFAULT_K = 3
DEFAULT_WEIGHTS = {"bm25": 1.0, "coverage": 0.5, "entity": 0.0, "vector": 0.0}
DEFAULT_MIN_COVERAGE = 0.3
DEFAULT_MAX_ANSWERS = 2
DEFAULT_MARGIN = 0.5

REFUSED_NO_MATCH = "no_lexical_match"
REFUSED_LOW_COVERAGE = "insufficient_evidence"


@dataclass
class Candidate:
    claim_id: str
    bm25: float  # relative to the best candidate, in (0, 1]
    coverage: float
    entity: float
    vector: float
    score: float


@dataclass
class HybridAnswer:
    question: str
    answer_id: Optional[str] = None
    #: The answer first, then any close runners-up returned with it.
    answer_ids: List[str] = field(default_factory=list)
    refused: Optional[str] = None
    detail: str = ""
    #: (edge type into this node or None, node id): entity -> claim -> source
    path: List[Tuple[Optional[str], str]] = field(default_factory=list)
    candidates: List[Candidate] = field(default_factory=list)
    tokens: int = 0

    @property
    def resolved(self) -> bool:
        return self.answer_id is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer_id": self.answer_id,
            "answer_ids": list(self.answer_ids),
            "refused": self.refused,
            "detail": self.detail,
            "path": [{"edge": e, "node": n} for e, n in self.path],
            "candidates": [c.__dict__ for c in self.candidates],
            "tokens": self.tokens,
        }


class HybridRetriever:
    """Ask questions of a graph; see the module docstring."""

    TOKENS_PER_WORD = 1.3  # the walker's own accounting, so costs compare

    def __init__(
        self,
        graph: ContextGraph,
        resolver: Resolver,
        *,
        k: int = DEFAULT_K,
        weights: Optional[Dict[str, float]] = None,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
        max_answers: int = DEFAULT_MAX_ANSWERS,
        margin: float = DEFAULT_MARGIN,
    ) -> None:
        self.graph = graph
        self.resolver = resolver
        self.k = k
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.min_coverage = min_coverage
        self.max_answers = max_answers
        self.margin = margin
        self._index_key: Optional[Tuple[int, int, int]] = None
        self._claims: List[Node] = []
        self._bm25: Optional[BM25] = None

    # ── the index follows the graph ──────────────────────────────────────
    def _index(self) -> Tuple[List[Node], BM25]:
        """BM25 over the live claims, rebuilt whenever that set can have moved.

        Keyed on the build counter (a refresh is one build), the node count and
        the number of invalidated nodes (a rejection invalidates without a
        build), so a claim retired since the last question is never served.
        """
        graph = self.graph
        key = (
            graph.build,
            len(graph.nodes),
            sum(1 for n in graph.nodes.values() if n.invalidated or n.pruned),
        )
        if key != self._index_key or self._bm25 is None:
            self._claims = [
                n for n in graph.nodes.values() if n.type is NodeType.CLAIM and n.live
            ]
            self._bm25 = BM25([n.label for n in self._claims])
            self._index_key = key
        return self._claims, self._bm25

    def _text_tokens(self, label: str) -> int:
        return int(len(label.split()) * self.TOKENS_PER_WORD) + 8

    def _question_entities(self, question: str) -> List[str]:
        found: List[str] = []
        for mention in _candidate_mentions(question):
            hit = self.resolver.match(mention)
            if hit is None:
                continue
            node = self.graph.nodes.get(hit[0])
            if node is not None and node.live and node.id not in found:
                found.append(node.id)
        return found

    # ── asking ───────────────────────────────────────────────────────────
    def ask(self, question: str) -> HybridAnswer:
        claims, bm25 = self._index()
        answer = HybridAnswer(question)
        hits = bm25.scored(question, self.k)
        if not hits:
            answer.refused = REFUSED_NO_MATCH
            answer.detail = "no live claim shares a content word with the question"
            return answer

        q_entities = set(self._question_entities(question))
        q_vector = embed(question)
        top_score = hits[0][1]
        w = self.weights
        for index, score in hits:
            claim = claims[index]
            mentioned = {
                e.dst for e in self.graph.out_edges(claim.id, [EdgeType.MENTIONS])
            }
            entity = len(q_entities & mentioned) / len(q_entities) if q_entities else 0.0
            vector = max(0.0, cosine(q_vector, claim.embedding)) if claim.embedding else 0.0
            coverage = bm25.coverage(question, index)
            relative = score / top_score if top_score else 0.0
            answer.candidates.append(Candidate(
                claim_id=claim.id,
                bm25=round(relative, 6),
                coverage=round(coverage, 6),
                entity=round(entity, 6),
                vector=round(vector, 6),
                score=round(
                    w["bm25"] * relative + w["coverage"] * coverage
                    + w["entity"] * entity + w["vector"] * vector,
                    6,
                ),
            ))

        # Highest score wins; ties go to BM25's order, which is deterministic.
        best = max(answer.candidates, key=lambda c: c.score)
        if best.coverage < self.min_coverage:
            answer.refused = REFUSED_LOW_COVERAGE
            answer.detail = (
                f"the best candidate covers {best.coverage:.0%} of the question's terms, "
                f"below the {self.min_coverage:.0%} needed to answer"
            )
            return answer

        answer.answer_id = best.claim_id
        answer.answer_ids = [best.claim_id]
        ranked = sorted(answer.candidates, key=lambda c: -c.score)  # stable: BM25 order on ties
        for other in ranked:
            if len(answer.answer_ids) >= self.max_answers:
                break
            if (
                other.claim_id not in answer.answer_ids
                and other.score >= best.score - self.margin
                and other.coverage >= self.min_coverage
            ):
                answer.answer_ids.append(other.claim_id)
        answer.path = self._path(best.claim_id, q_entities)
        answer.tokens = sum(self._text_tokens(self.graph.node(n).label) for _, n in answer.path)
        for extra in answer.answer_ids[1:]:
            node = self.graph.node(extra)
            answer.tokens += self._text_tokens(node.label)
            if node.provenance and node.provenance.source_id in self.graph.nodes:
                answer.tokens += self._text_tokens(
                    self.graph.node(node.provenance.source_id).label
                )
        return answer

    def _path(self, claim_id: str, q_entities: Sequence[str]) -> List[Tuple[Optional[str], str]]:
        """entity -mentioned by- claim -derived_from-> source, over live edges.

        The source is the claim's provenance source — the file version it was
        lifted from — so the path ends where its location attrs point.
        """
        graph = self.graph
        path: List[Tuple[Optional[str], str]] = []
        for edge in graph.out_edges(claim_id, [EdgeType.MENTIONS]):
            if edge.dst in q_entities:
                path.append((None, edge.dst))
                path.append((EdgeType.MENTIONS.value, claim_id))
                break
        else:
            path.append((None, claim_id))
        claim = graph.node(claim_id)
        source_id = claim.provenance.source_id if claim.provenance else None
        for edge in graph.out_edges(claim_id, [EdgeType.DERIVED_FROM]):
            if edge.dst == source_id:
                path.append((EdgeType.DERIVED_FROM.value, source_id))
                break
        return path

    def render(self, answer: HybridAnswer) -> str:
        """The answer and its evidence path, readably."""
        if not answer.resolved:
            return f"refused — {answer.refused}: {answer.detail}"
        lines: List[str] = []
        for depth, (edge, node_id) in enumerate(answer.path):
            node = self.graph.node(node_id)
            where = node.attrs.get("location")
            arrow = f"-[{edge}]- " if edge else ""
            label = f"({node.type.value}) {node.label}"
            lines.append("  " * depth + arrow + label + (f"  [{where}]" if where else ""))
        for extra in answer.answer_ids[1:]:
            node = self.graph.node(extra)
            where = node.attrs.get("location")
            lines.append(f"also close: (claim) {node.label}" + (f"  [{where}]" if where else ""))
        return "\n".join(lines)
