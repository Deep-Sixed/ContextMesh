# Phase 2 — Incremental folder rebuilds

Status: **design, for review before implementation.** Nothing here is built yet.

Phase 1 (PR #48) builds a graph from a folder in one pass. Every change to the
folder means rebuilding from scratch, at roughly 15 seconds per MB of text, and
throws away anything built on top of the old graph: decisions, walk history and
assumptions recorded through MCP writes. This document proposes a refresh that
re-extracts only what changed, and retires what a file stopped saying through
the mechanism ContextMesh already has for "this is no longer true", so the
change is selective, explained and reversible in time.

## Non-negotiables

These are the properties the existing engine guarantees, restated as
constraints on the refresh:

1. **Provenance.** Every live claim is a verbatim span of a file version the
   graph can name, and says where (path, lines, version).
2. **Audit.** Nothing is deleted. A claim a file stops saying is *invalidated*
   by evidence and stays in the graph, with a readable chain explaining why
   (GRAPH.md rules 3 and 7).
3. **Selective invalidation.** Only what rested on the retired text falls. A
   decision built on an unchanged sentence survives an edit elsewhere in the
   same file (GRAPH.md rule 2).
4. **Determinism.** The same sequence of folder states produces a
   byte-identical graph, independent of `PYTHONHASHSEED` and wall-clock time.
   Wall-clock time stays out of the graph (`temporal.py`).
5. **Time travel.** An as-of query for a date before an edit sees the
   pre-edit claims as live, and one after it does not.
6. **The demo is untouched.** `dashboard/data/mesh.json` regenerates
   byte-identical.

## What a refresh must handle

| Folder change | Expected effect |
|---|---|
| File unchanged | Nothing re-extracted. Its claims, and everything built on them, stand. |
| File edited | Sentences still present stand. Sentences removed fall. New sentences become new claims. |
| File added | Extracted as in Phase 1. |
| File deleted | All its claims fall. Its source nodes stay, as history. |
| Sentence removed, later restored | The restored sentence is a **new** claim that `supersedes` the dead one. A dead node is never revived (rule 6). |
| Same sentence in two files | Two claims, one per file. Deleting it from one file leaves the other standing. |

## Model

### Today (Phase 1) and why it can't be refreshed

- A claim's id is `slug(sentence)`, so one sentence in two files is one claim,
  located at its first sighting.
- A source's id is `slug(path)`, one per file, with no notion of a version.
- Nothing records which file version a claim was read from, beyond the
  provenance span.

Retiring a sentence from one file would therefore also retire it from every
other file that says it, and there is no version to point evidence at.

### Proposed identities

| Node | Id | Notes |
|---|---|---|
| Source **version** | `slug("{path}@{sha256[:16]}", "source")` | One per distinct content of a file. Attrs: Phase 1's, plus `path`, `sha256`, `size_bytes`. Old versions stay as history. |
| Claim | `slug("{path}\|{sentence}\|{generation}", "claim")` | Scoped to the file. `generation` starts at 0 and rises only when a sentence returns after falling, so ids stay deterministic. |
| Claim assumption | `assume('{path}@{sha256[:12]} says: "{sentence}"')` | One per claim, created when the claim is. The statement names the version the claim was first read from, which keeps it unique across a fall and a return. |
| Re-read evidence | `kind: "file-reread"`, one per changed or deleted file per refresh | Provenance points at the new source version (for a deletion, at the last version with `attrs.deleted_at_build`). |

Each claim keeps Phase 1's `derived_from` edge to the source version it was
first read from, and gains:

```
claim -[depends_on]-> assumption("docs/a.md@3f9c… says: "…"")
```

### How an edit retires a sentence

On refresh, a changed file is re-read and diffed by sentence:

```
evidence("docs/a.md re-read at build 7: sha256 9a01…; 2 sentences no longer present")
   -[contradicts]-> assumption("docs/a.md@3f9c… says: "The cache is per tenant."")
                        <-[depends_on]- claim("The cache is per tenant.")
                                            <-[derived_from]- decision("Shard by tenant")
```

`AssumptionLedger.reject(assumption, evidence_id=…)` then does exactly what it
does for any other assumption: it walks the blast radius, invalidates the claim
and whatever was `derived_from` it, and returns an `InvalidationReport` with
the chain. No new invalidation path is introduced.

Unchanged sentences in the edited file are untouched. They depend on their own
assumptions, which nothing contradicts. Their `location` attrs are updated to
the new version's lines and gain `location_sha256`, while their provenance
stays the first sighting: a historical fact about where the text was lifted.

### Why one assumption per claim, not per file

A per-file-version assumption ("docs/a.md has content 3f9c…") would be cheaper,
but rejecting it would also take down every unchanged sentence in the file and
everything built on them. That breaks non-negotiable 3. Keeping unchanged
claims alive would then need their `depends_on` edges retired individually,
which is a second invalidation writer that GRAPH.md does not have. One
assumption per claim keeps a single rule doing all the work. The cost is
measured below.

### Entities

Entity discovery is corpus-wide: "appears in two or more files" can change
with any edit. A refresh re-runs discovery over the whole current corpus.
Parsing is cheap; it's resolution and embedding that cost.

- **Newly qualifying names** are registered and linked to every live claim that
  mentions them, including claims in unchanged files. `mentions` edges never
  propagate invalidation, so this is safe.
- **Names that stop qualifying are kept.** An entity is not asserted by any one
  file, so nothing contradicts it. Once only dead claims mention it, the walker
  already refuses dead edges, and graph health reports it.
- Entities are never invalidated by a refresh. They are `derived_from` source
  versions, which are never invalidated either: invalidating a source would
  take down every entity derived from it (rule 2), including ones other files
  still support.

### Time

- A new source version's `retrieved_at` is its file's modification date, as in
  Phase 1.
- The re-read evidence takes its source time from that version. So
  `temporal._fell_at` dates a retired claim's fall to the edit, and an as-of
  query before the edit sees the claim live (non-negotiable 5).
- For a deletion, the folder offers no modification time. The evidence is
  dated by an explicit `--as-of` given to the refresh, or is left `UNDATED`.
  `temporal.py` already reports undated items as uncertainty rather than
  guessing.
- Processing time is the build counter: each refresh is one build.

## Refresh algorithm

```
refresh(session, folder):
  1. Walk and read the folder exactly as Phase 1 does (same skip rules,
     limits, encodings). Hash each file's bytes with SHA-256.
  2. From the graph, recover the previous state: for each path, its latest
     source version and its live claims. Every fact needed is already an
     attr or id in the graph, so no side manifest exists to drift from it.
  3. Classify each path: unchanged (same sha256), changed, added, deleted.
     If nothing changed, stop: no build, no new generation.
  4. graph.build += 1.
  5. Re-run entity discovery over all current files; register new names.
  6. For each changed or added file: add the source version; split sentences
     as in Phase 1; diff its sentence keys against the live claims for that
     path:
       kept     -> update location attrs only
       new      -> create claim (next generation if a dead one exists, with
                   claim -[supersedes]-> dead claim) + assumption + depends_on
       removed  -> collect the claim's assumption
  7. For each changed or deleted file with removed sentences: create one
     evidence node, then ledger.reject(each collected assumption,
     evidence_id=…). Paths are processed in sorted order, so the order of
     rejections, and therefore the graph, is deterministic.
  8. Link mentions for newly registered entities across live claims.
  9. Embed nodes added this build; run the build-time half of PRUNE.
 10. Return a RefreshReport. The caller saves it as the session's next
     generation, through the existing atomic commit.
```

Steps 6–9 call into `Pipeline` with only the changed documents. `Pipeline`
gains a mode that adds a build onto an existing graph and resolver, rather
than assuming it owns the whole corpus. The `annotate` hook from Phase 1
carries the version attrs.

### Failure and atomicity

A refresh works on the loaded session in memory and changes nothing on disk
until the save, which is the existing generation commit. If any step raises,
nothing is saved, and the previous generation keeps serving. Refreshing a
session while the MCP server is serving it is **out of scope** here. It needs
a copy-and-swap so readers never see a half-applied refresh, and is listed
under open decisions.

## Surface

```bash
# first time: full build, saved as a session
python -m contextmesh_mcp --folder ~/notes --save ./notes-session

# later: re-read the folder and apply only what changed, as the next generation
python -m contextmesh_mcp --session ./notes-session --refresh ~/notes

# what would change, without saving anything
python -m contextmesh_mcp --session ./notes-session --refresh ~/notes --dry-run
```

`RefreshReport` (printed as JSON, and as text by the CLI):

- Files: unchanged / changed / added / deleted / skipped, each with a reason.
- Claims: kept / added / retired / superseded-on-return.
- **Downstream:** nodes invalidated beyond the retired claims themselves, such
  as decisions and produced entities, each with its `why` chain from the
  `InvalidationReport`. This is the number a user acts on.
- Entities registered this refresh.
- Build number, and `ingested_at` (reported only, as in Phase 1).

## Persistence and compatibility

- **No new session companion and no schema bump.** The refresh state is
  recovered from the graph (step 2), so `SESSION_VERSION` stays at 2. The
  generation commit protocol is reused as-is.
- **Phase 1 sessions** have `slug(path)` source ids and sentence-only claim
  ids. A refresh refuses them with a clear message ("built before incremental
  refresh; rebuild with `--folder … --save`"), rather than guessing at a
  migration. Phase 1 has not been released (PyPI is at 0.1.1, before it), so
  no published session format is affected.
- **Phase 1 full builds change ids.** With file-scoped claims, a sentence
  repeated in two files becomes two claims instead of one. Answers can now cite
  either file. Phase 1 tests that assume deduplication will be updated, and the
  change is called out in the PR.

## Cost

Per claim, the model adds one assumption node and one `depends_on` edge. The
assumption nodes are **not embedded**, because their text duplicates the
claim's, and embedding vectors are the largest per-node cost (Phase 1
measurement: about 140 MB of 350 MB per MB of text). The first implementation
slice measures the real overhead on the generated 4 MB corpus and records it
in the PR. If it exceeds 15% of peak memory, the decision comes back here
before going further.

A refresh after a small edit should cost roughly what building the changed
files from scratch costs, plus a re-read and hash of every file, which is
cheap. The benchmark slice (Phase 2, item 3) measures it.

## Verification

Property tests over sequences of folder states, alongside unit tests:

- **Equivalence of the live view.** After any sequence of edits, the live
  claims are exactly the sentences a fresh Phase 2 build of the final folder
  would extract, compared as `(path, sentence)` pairs.
- **Selectivity.** A decision `derived_from` an unchanged sentence survives an
  edit to the same file; one built on a removed sentence falls, and its `why`
  chain names the evidence and the file version.
- **No revival.** Remove, then restore, a sentence: two claims, the second
  `supersedes` the first, and the first stays invalidated.
- **Shared sentences.** The same sentence in two files; deleting one file
  leaves the other's claim live.
- **Determinism.** The same edit sequence under two `PYTHONHASHSEED` values
  gives byte-identical graph JSON. A refresh with no changes writes no
  generation.
- **Time.** As-of queries before and after an edit, including an undated
  deletion, which is reported as `UNDATED`.
- **Persistence.** Refresh, save, load, refresh again: identical to doing both
  refreshes in memory.
- **Unchanged guarantees.** The full existing suite, the demo export check and
  the leak check pass.

## Open decisions

1. **Per-claim assumption overhead.** The proposal above, with a measured
   threshold. The alternative is a new GRAPH.md mechanism for retracting claims
   directly. That is cheaper, but it is a schema change, and it means a second
   way for things to fall.
2. **Renames.** Treated as delete plus add, so a renamed file's claims fall
   and return under the new path, and anything built on them falls. Detecting
   a rename by identical content hash would avoid that, but it guesses intent.
   Proposed: not in this slice.
3. **Refreshing a served session.** Copy-and-swap inside the MCP server, plus
   an `mesh_refresh` tool. Proposed as a follow-up slice once the offline
   refresh is proven.
4. **Dating deletions.** `--as-of` or `UNDATED`, as above. The alternative,
   stamping the refresh's wall-clock date, would put a clock into the graph,
   which `temporal.py` rules out.

## Slices

1. Versioned sources, file-scoped claims, per-claim assumptions in full
   builds. Includes the memory measurement and updates to the Phase 1 tests.
2. `refresh()` for edits, additions and deletions, plus `RefreshReport`,
   `--refresh` and `--dry-run`.
3. Restored sentences (`supersedes`), entity re-discovery, and
   location-attr updates.
4. Property tests and the persistence round-trip.
