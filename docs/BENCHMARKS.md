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

**Corpus.** This repository's own Markdown: 12 files, 804 claims. It is real
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

Corpus `e7ca20d037881e85`: 12 files, 804 claims, 27,083 tokens. 32 answerable
questions, 29 covered, 8 out of scope.

| Method | Correct | Of covered | Mean tokens / question | Out-of-scope refused |
|---|---|---|---|---|
| Graph walk | 17/32 (53%) | 17/29 (59%) | 62 | 5/8 |
| BM25 top-1 | 17/32 (53%) | 17/29 (59%) | 26 | 3/8 |
| BM25 top-3 | 22/32 (69%) | 22/29 (76%) | 91 | 3/8 |
| BM25 top-5 | 22/32 (69%) | 22/29 (76%) | 152 | 3/8 |
| Read everything | 29/32 (91%) | 29/29 (100%) | 27,083 | 0/8 |

The README's "flat" figure averaged about 7,840 tokens on the same questions.

## What it says

- **Against a real retriever, the token advantage is gone.** The walk spends
  about 62 tokens per question; BM25 top-1 spends 26 for the same number of
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
- **Reading everything is the only way to reach the ceiling,** at about 430
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
out-of-scope ones. Corpus `7100f913821685b5`: 25 files, 977 claims.

| Method | Correct | Mean tokens / question | Out-of-scope refused |
|---|---|---|---|
| Graph walk | 15/18 (83%) | 68 | 6/6 |
| BM25 top-1 | 12/18 (67%) | 31 | 3/6 |
| BM25 top-3 | 15/18 (83%) | 96 | 3/6 |
| BM25 top-5 | 15/18 (83%) | 158 | 3/6 |
| Read everything | 18/18 (100%) | 32,973 | 0/6 |

Here the walk beats one keyword result, matches three at about 70% of the
cost, and refuses every out-of-scope question.

Read it with a heavier caveat than the docs suite. These questions were
written by sampling the extracted claims, so every answer is in a claim by
construction (18/18 coverage). They lean toward sentences the extractor keeps,
and toward the distinctive vocabulary of docstrings, which suits both methods.
The docs suite, written from the documents rather than from the claims, is the
fairer of the two.

Suites name the file types they cover (`corpus.suffixes`), so the docs suite
still reads only Markdown and text now that `.py` files are read by default.
Code ingestion did not change its results. Its fingerprint moves only when the
repo's Markdown does.

The code suite's corpus is this project's own source, so its fingerprint moves
with almost every code change. Numbers are always reported with the
fingerprint they ran on.

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
2. **Teach the walk to refuse in-scope questions it can't answer.** For
   example, refuse below a relevance score. Right now it never does.
3. **Widen the extractor's verb test.** Three of 32 answers never became
   claims.
4. **Add independently labelled suites,** including a user's own notes folder.
