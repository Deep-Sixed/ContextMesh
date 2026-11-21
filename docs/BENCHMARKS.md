# Benchmarks — does the graph walk answer correctly, and at what cost?

```bash
python -m contextmesh bench                       # this repo's docs suite
python -m contextmesh bench --suite my-suite.json --json result.json
```

## Why this exists

The number this project has quoted is "60 tokens against 7,300 flat". The
"flat" side is `Walker.tokens_flat`: **40 claims sampled by a hash of the
question, plus 160 tokens of window each**. That is about 6,400 tokens of
window before a single claim is counted, and the same budget whatever the
question asks. It is not a retriever, nothing checks whether those 40 claims
hold the answer, and nothing checked whether the walk's own answer was right.
So the ratio measured a budget, not a result.

This benchmark asks labelled questions and scores every method on the same
two things: **was the answer right**, and **what did it cost**.

## Method

**Corpus.** This repository's own Markdown: 12 files, 805 claims. It is real
documentation written to explain the project, not text written for the test.
`docs/BENCHMARKS.md` itself is excluded, because it quotes the questions.
Every result records a fingerprint of the exact files it ran on.

**Questions** (`bench/contextmesh-docs.json`):
- **32 answerable questions.** Each names the file(s) a correct answer comes
  from and phrases a correct answer contains.
- **8 out-of-scope questions** (refund policies, the 2018 World Cup, …). For
  these, the right result is to refuse.

**Methods,** all answering from the same extracted claim sentences:

| Method | Answers with | Correct when | Refuses when |
|---|---|---|---|
| Graph walk | the walk's answer claim | it comes from an accepted file and contains an expected phrase | it dead-ends |
| BM25 top-k (k = 1, 3, 5) | the k best lexical matches | any of the k is correct | no word matches at all |
| Read everything | every claim | some extracted claim is correct | never |

**Cost** uses the walker's own counter for every method: `words × 1.3 + 8` per
text. The walk is charged for every node on its path, exactly as the walker
charges itself.

**Coverage.** A question is *covered* when some extracted claim answers it.
That is the ceiling for every method: a sentence the extractor dropped can't be
returned by any of them. Accuracy is reported both overall and against
coverage, so extraction gaps and retrieval misses stay separate.

The walk's answers don't depend on question order. The same answers come back
with a fresh walker per question and with the questions reversed.

## Results

Corpus `92c57aeb78f8236b`: 12 files, 805 claims, 27,147 tokens. 32 answerable
questions, 29 covered, 8 out of scope.

| Method | Correct | Of covered | Mean tokens / question | Out-of-scope refused |
|---|---|---|---|---|
| Graph walk | 17/32 (53%) | 17/29 (59%) | 60 | 5/8 |
| BM25 top-1 | 17/32 (53%) | 17/29 (59%) | 26 | 3/8 |
| BM25 top-3 | 22/32 (69%) | 22/29 (76%) | 91 | 3/8 |
| BM25 top-5 | 22/32 (69%) | 22/29 (76%) | 151 | 3/8 |
| Read everything | 29/32 (91%) | 29/29 (100%) | 27,147 | 0/8 |

The README's "flat" figure averaged about 7,840 tokens on the same questions.

## What it says

- **Against a real retriever, the token advantage is gone.** The walk spends
  about 60 tokens per question; BM25 top-1 spends 26 for the same number of
  correct answers. BM25 top-3 gets 5 more questions right for 91 tokens.
  "Roughly 120× fewer tokens" was a comparison with padding, not with search.
- **Accuracy is the same as one keyword search.** The walk and BM25 top-1
  each get 17 right, but not the same 17. The walk alone answered "Which edges
  carry invalidation backwards?"; BM25 alone answered "What is an assumption
  rejected by?".
- **The walk refuses better, but not reliably.** It refused 5 of the 8
  out-of-scope questions, where BM25 refused 3. It answered three it should
  have refused, and it never refused an in-scope question, even the 12 covered
  ones it got wrong. A wrong answer comes back looking like a right one, with
  a path.
- **Reading everything is the only way to reach the ceiling,** at about 450
  times the walk's cost.
- **The extractor loses answers before retrieval starts.** 3 of 32 questions
  have no correct claim at all:
  - "`tokens ≈ words × 1.3`" has no verb;
  - "`supersedes` never deletes." is under four words;
  - "It never becomes an import, a callable…" uses a verb, *becomes*, that
    isn't on the extractor's list of assertive verbs.

What the walk does have that BM25 does not is the path: every answer comes with
the typed chain that justifies it, and the provenance, invalidation and
time-travel machinery behind that. This benchmark doesn't score any of that.
It scores only whether the right sentence comes back, and at what token cost.

## Code suite

`python -m contextmesh bench --suite bench/contextmesh-code.json` asks 18
questions answered by the docstrings and comments of `contextmesh/`, plus 6
out-of-scope ones. Corpus `843be1ee936a360c`: 25 files, 980 claims.

| Method | Correct | Mean tokens / question | Out-of-scope refused |
|---|---|---|---|
| Graph walk | 13/18 (72%) | 77 | 6/6 |
| BM25 top-1 | 12/18 (67%) | 31 | 2/6 |
| BM25 top-3 | 15/18 (83%) | 96 | 2/6 |
| BM25 top-5 | 15/18 (83%) | 158 | 2/6 |
| Read everything | 18/18 (100%) | 33,134 | 0/6 |

The walk beats one keyword result by one question, trails three, and refuses
every out-of-scope question. BM25 refuses two.

**An earlier run showed the walk at 15/18, and the difference is worth
keeping on record.** That run treated every function name and every import as
a marked-up name, so generic words became entities: `field` (from
`dataclasses.field`), `read`, `json`, `graph`. Two answers were right only
through those hubs. "Is a stored ledger digest regenerated on load?" was
reached through `field`, because the answer says "not a *field* to be
regenerated". Code names are now marked only when they are distinctive as
written (dotted, snake_case, capitalised, or imported from the project), so
those false links and those two lucky answers are gone.

Read this suite with two caveats beyond the docs suite's:

- **Its questions were written by sampling the extracted claims.** Every
  answer is in a claim by construction (18/18 coverage), and the questions
  lean toward the sentences and vocabulary the extractor keeps. The docs
  suite, written from the documents, is the fairer of the two.
- **Its corpus is this project's own source, so it moves with the code.**
  Adding a docstring that says "with a capital" was enough to make BM25 stop
  refusing "What is the capital of Australia?". Compare methods within one
  run, where they all see the same corpus, rather than across runs. Numbers
  are always reported with the fingerprint they ran on.

Suites name the file types they cover (`corpus.suffixes`), so the docs suite
still reads only Markdown and text now that `.py` files are read by default.
Code ingestion did not change its results. Its fingerprint moves only when the
repo's Markdown does.

## Hybrid retrieval

`contextmesh/hybrid.py` (`HybridRetriever`) combines the two:
1. **Candidates:** BM25 picks candidate claims, from live claims only.
2. **Re-ranking:** the candidates are re-ranked.
3. **Evidence or refusal:** the answer comes with its evidence path in the
   graph (question entity → claim → source, with the file location). If the
   best candidate covers too little of the question, it refuses instead.
4. **Close calls:** up to one runner-up is returned alongside, and paid for in
   tokens.

The defaults were chosen by `ops/calibrate-hybrid.py` on the two development
suites only, with the held-out suites committed beforehand (commit `eea2b08`)
and run once, after the implementation was committed (`cb49b46`). The grid
search covered 21,289 settings; the output is `ops/calibrate-hybrid.log`.
"Wrong" below means answered, and not correctly. Refusing an answerable
question counts as neither correct nor wrong.

### Development suites (calibrated on)

| Docs, corpus `92c57aeb78f8236b` | Correct | Wrong | Tokens | Out-of-scope refused | Latency |
|---|---|---|---|---|---|
| Hybrid | 22/32 | 8 | 73 | 8/8 | 0.8 ms |
| Graph walk | 17/32 | 15 | 60 | 5/8 | 13.9 ms |
| BM25 top-1 | 17/32 | 15 | 26 | 3/8 | 0.4 ms |
| BM25 top-3 | 22/32 | 10 | 91 | 3/8 | 0.3 ms |

| Code, corpus `bcc84c2d7e3bfc0f` | Correct | Wrong | Tokens | Out-of-scope refused | Latency |
|---|---|---|---|---|---|
| Hybrid | 13/18 | 4 | 80 | 5/6 | 0.9 ms |
| Graph walk | 12/18 | 6 | 77 | 6/6 | 13.2 ms |
| BM25 top-1 | 12/18 | 6 | 31 | 2/6 | 0.5 ms |
| BM25 top-3 | 15/18 | 3 | 96 | 2/6 | 0.4 ms |

### Held-out suites (run once, never calibrated on)

| Docs, corpus `92c57aeb78f8236b` | Correct | Wrong | Tokens | Out-of-scope refused |
|---|---|---|---|---|
| Hybrid | 3/10 | 3 | 48 | 4/4 |
| Graph walk | 1/10 | 9 | 58 | 4/4 |
| BM25 top-1 | 4/10 | 6 | 27 | 2/4 |
| BM25 top-3 | 4/10 | 6 | 95 | 2/4 |
| BM25 top-5 | 5/10 | 5 | 157 | 2/4 |

| Code, `contextmesh_mcp/`, corpus `b4d5e0662d61b4fb` | Correct | Wrong | Tokens | Out-of-scope refused |
|---|---|---|---|---|
| Hybrid | 7/10 | 3 | 84 | 4/4 |
| Graph walk | 5/10 | 5 | 71 | 4/4 |
| BM25 top-1 | 3/10 | 7 | 33 | 2/4 |
| BM25 top-3 | 8/10 | 2 | 94 | 2/4 |

### Against the acceptance criteria

| Criterion | Target | Result |
|---|---|---|
| Docs accuracy | exceed 22/32 | **Not met.** 22/32: ties BM25 top-3, does not exceed it. Held out: 3/10, below BM25 top-1's 4/10. |
| Code accuracy | at least 15/18 | **Not met.** 13/18, below BM25 top-3's 15/18 in the same run. (15/18 was the walk's figure before the generic-hub fix; graph-only is 12/18 in this run.) Held out: 7/10, against BM25 top-3's 8/10. |
| Out-of-scope refusals | maintain or improve | **Met.** 8/8 docs, 5/6 code, 4/4 and 4/4 held out, against BM25's 3/8, 2/6, 2/4 and 2/4. The walk is 6/6 on code, one better. |
| Tokens | below BM25 top-3 where possible | **Met everywhere it was measured.** 73 vs 91, 80 vs 96, 48 vs 95, 84 vs 94. |
| Provenance | complete evidence paths | **Met.** Every answer carries entity → claim → source over live edges, with the file location (tested). |
| Invalidation | never retrieve retired claims | **Met.** The index holds live claims only and is rebuilt when the graph moves. Tested on rejection and on refresh. |
| Reproducibility | same inputs, same results | **Met.** Asking is read-only (`Resolver.match`), answers do not depend on order, and the graph and resolver are byte-identical after asking (tested). |

### What it says

- **The hybrid is a precision gain, not an accuracy gain.** On held-out docs
  it answered 6 of 10 questions and got 3 right. BM25 top-1 answered all 10
  and got 4 right, so it was wrong 6 times to the hybrid's 3. Of the 4
  questions the hybrid refused, BM25 top-1 got 1 right. It gave up one
  correct answer to avoid three wrong ones. Whether that trade is right
  depends on how costly a confident wrong answer is.
- **The refusal threshold overfits.** Calibrated on the development suites,
  it refused no in-scope question on held-out code, but 4 of 10 on held-out
  docs, which ask in other words than the documents use. Coverage is lexical,
  so a paraphrase looks like thin evidence.
- **The graph's re-ranking signals earned no weight.** Calibration set entity
  overlap and embedding similarity to 0. On these suites the graph
  contributes the evidence path, live-only retrieval and invalidation, not the
  choice of answer. The walk alone stays weakest on held-out docs (1/10, 9
  wrong).
- **Cost:** about 1 ms per question against the walk's 12–16 ms. The index
  peaks at about 1.2 MB for 805 claims and 1.4 MB for 1,010.
- **Held-out questions are harder for every method.** BM25 top-5 gets 5 of
  10 on held-out docs, although every answer is in a claim. The development
  suites, written closer to the documents' wording, flatter everything.

### What would move it

- **A semantic measure of evidence, not a lexical one.** That would let
  refusal tell "says it differently" from "doesn't say it". This is the
  largest lever the held-out docs point to, and the one that needs a model or
  a better embedding than the hashed one the engine has.
- **Independently written questions.** The held-out suites were written by
  the same author as the code, before the hybrid existed, so they guard
  against tuning but not against blind spots.

## Limits

- **One small corpus.** 12 files of single-sentence claims. Long-document
  corpora would change the token picture, as `ARCHITECTURE.md` already notes.
- **The questions and labels were written by the same author as the code,**
  with the corpus open. That biases them, in an unknown direction. Independent
  labelling on a second corpus is the obvious next step.
- **Scoring is lexical:** an expected phrase in an accepted file. A correct
  answer worded differently is scored wrong, and a sentence that merely
  repeats the phrase is scored right.
- **No language model reads the result.** Token counts measure what would be
  handed to one, not what it would conclude.
- **Graphify's "70× fewer tokens" is not tested here.** This measures
  ContextMesh only.

## Directions the results point to

1. **Use lexical search to seed the walk.** BM25 top-1 matches the walk's
   accuracy at under half its cost, and they get different questions right.
   *Tried:* see [Hybrid retrieval](#hybrid-retrieval). It gains precision and
   refusals, not accuracy.
2. **Teach the walk to refuse in-scope questions it can't answer.** For
   example, refuse below a relevance score. Right now it never does.
   *Tried in the hybrid,* with a lexical coverage threshold. It cuts wrong
   answers, but it also refuses paraphrased questions it could have answered.
3. **Widen the extractor's verb test.** Three of 32 answers never became
   claims.
4. **Add independently labelled suites,** including a user's own notes folder.
