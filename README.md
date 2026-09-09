# ContextMesh

> One graph · six node types · every answer walks a path you can read

ContextMesh is a typed context graph for long-running agents. It keeps not only
*what* is connected, but *why* — where a claim came from, what a decision rested
on, and what stops being true when an assumption fails.

![The ContextMesh dashboard: build path, the live typed graph, hop budget, edge
ledger, walk-vs-flat token cost, the traversal grid and the dead-end ledger](docs/img/dashboard.png)

Every panel above is a query against the graph — the node counts, the hop
histogram, the edge traffic, the dead-end reasons. Open
[`dashboard/context-mesh.html`](dashboard/context-mesh.html) for the live page
in a single file, or rebuild it from your own data:

```bash
python -m contextmesh export --inline
```

Then open `dashboard/index.html` in a browser: `start dashboard/index.html` on
Windows, `open dashboard/index.html` on macOS, or
`xdg-open dashboard/index.html` on Linux.

```bash
git clone https://github.com/Deep-Sixed/ContextMesh
cd ContextMesh

python -m contextmesh demo            # build, walk, break an assumption, report
python -m contextmesh ask "Why did the Index Builder run out of memory?"
python -m contextmesh health          # what is quietly wrong with the graph
python -m contextmesh invalidate      # reject an assumption, see the fallout
python -m contextmesh execute         # break one, then re-run only what fell
python -m contextmesh export --inline # regenerate the dashboard's data
```

No dependencies. Python 3.9+. `python -m unittest discover -s tests` runs the
suite. CI runs it on 3.9 through 3.13, plus Windows, ruff and a set of
end-to-end smoke checks — including that a build is byte-identical across
`PYTHONHASHSEED`, and that `dashboard/data/mesh.json` still matches what the
engine produces.

## An answer is a path

```
$ python -m contextmesh ask "What made the sharding rule wrong?"

Q: What made the sharding rule wrong?
A: Sharding must key on tenant as well as corpus position.
   (2 hops, score 0.25, 60 tokens against 7,307 flat)

(claim) The Partitioned Rebuild is sound but the sizing rule was wrong.
  -[derived_from]-> (source) Postmortem 233 — the linear sharding assumption
    <-[derived_from]- (claim) Sharding must key on tenant as well as corpus position.
```

That path is the justification. There is no second pass that reconstructs a
rationale after the fact — the walk *is* the reasoning, and it is stored, so the
same question does not get re-guessed next time.

When the graph cannot answer, it says which of exactly four things went wrong
rather than returning its best guess:

```
Q: What is the refund policy for annual plans?
A: dead end — entity_unresolved
   no mention resolved to a canonical entity: 'refund', 'policy', 'annual', 'plans'
```

## The ontology is a file

[`GRAPH.md`](GRAPH.md) is not documentation of the schema. It **is** the schema:
`contextmesh/ontology.py` parses it at import time, and every write typechecks
against it.

```python
graph.add_edge(entity.id, EdgeType.MENTIONS, source.id)
# OntologyError: entity-[mentions]->source is not a legal pair;
#                GRAPH.md allows claim->entity, source->entity
```

Six node types — `entity`, `claim`, `source`, `decision`, `assumption`,
`evidence` — and ten edge types: `mentions`, `derived_from`, `cites`,
`contradicts`, `supports`, `depends_on`, `produces`, `supersedes`,
`justified_by`, `resolves_to`. There is no code path that stores an edge without
one, which is why `untyped_edges == 0` is an invariant rather than a metric.

## How a node earns its place

```
CHUNK → EXTRACT → RESOLVE → LINK → EMBED → PRUNE
```

| Stage | What happens | On the demo corpus |
|---|---|---|
| **CHUNK** | Spans in from each source | 70 spans, 14 sources |
| **EXTRACT** | A span becomes a claim only if it asserts something | 69 claims |
| **RESOLVE** | One id per real-world thing | 47 entities, 90 surface forms folded; **1,264 mentions dropped** |
| **LINK** | Typed edges only; an illegal pair is dropped, never stored untyped | 372 edges, 8 of the 10 types in use |
| **EMBED** | A vector on the node, for seeding walks | every node |
| **PRUNE** | Drop what nothing walked | orphans removed |

Most of the corpus dies at RESOLVE, and that is the point: admitting a wrong
merge corrupts every walk that later crosses the entity, while dropping a
mention costs one span — and the drop is recorded, not swallowed.

## Selective invalidation

Assumptions are first-class, versioned nodes. When one is disproven, exactly the
work that stood on it is invalidated, and the report proves both halves:

```
$ python -m contextmesh invalidate

rejected assumption: Shard count grows linearly with corpus size

INVALIDATED (2)
  ✗ Rebuild the index in partitions
      because: assumption(Shard count grows linearly…) <-depends_on- Rebuild the index in partitions
  ✗ Partitioned Rebuild
      because: assumption(…) <-depends_on- Rebuild the index in partitions -produces-> Partitioned Rebuild

PRESERVED (136) — untouched by the rejection
```

Failure travels along three edges and no others, and direction matters:
`depends_on` and `derived_from` are followed **backwards** (the thing that
depends falls with its ground), `produces` **forwards** (an artefact falls with
the decision that made it). `mentions`, `cites`, `supports` and `supersedes`
never propagate — that is what makes this selective rather than a purge.

`supersedes` never deletes, so decision history stays append-only and "why did
we change our mind" is answered by walking, not by reading a changelog.

## Selective re-execution

Knowing what fell is half the claim. `contextmesh/execute.py` is the half that
acts on it: a `Runner` binds units of work to the graph's own vocabulary — a
task *is* a `decision` node, its ground is a `depends_on` edge to an
`assumption`, its dependencies are `depends_on` edges to other decisions, its
outputs are `produces` edges to entities — so scheduling reads off the ontology
rather than a second copy of it.

```
$ python -m contextmesh execute

ROUND 1 · 5 executed, 0 cached
  layer 1: schema, hashing, tokens
  layer 2: routes
  layer 3: rate_limit

THE GROUND MOVED · discovered by an auditor, not announced
  hashing: argon2-cffi has an open advisory — CVE-2026-9999: memory-hard
           parameters silently downgraded
  rejected: argon2-cffi has no open advisory
  blast radius 8, preserved 15

ROUND 3 · 3 executed, 2 cached
  layer 1: hashing
  layer 2: routes
  layer 3: rate_limit
  cached  schema — untouched, previous result stands
  cached  tokens — untouched, previous result stands
```

Three properties are worth being precise about, because they are the ones a
demo can fake:

**The failure is discovered, not announced.** `recheck()` re-runs every standing
auditor against current facts and executes nothing. The CVE is published to a
feed *outside* the graph; the hashing auditor reads that feed and returns
`ctx.disproved(...)`, and that disproof is what rejects the assumption and
computes the blast radius. Nothing reaches in and marks a node false. An auditor
distinguishes "the output is wrong" (`ctx.fail`, which fails one task) from "the
ground is false" (`ctx.disproved`, which takes down everything standing on it),
because those have different blast radii and an engine should not have to guess.

**Re-running stays append-only.** A rerun never revives the old decision — it
appends a new one that `supersedes` it, so the superseded reasoning is still
walkable. Artefacts are the exception, and deliberately so: an `entity` fell
only because the decision that produced it fell, so rebuilding that decision
brings it back under the same id. An artefact the rerun *stops* producing — the
Argon2 parameters, once the hasher becomes bcrypt — stays invalidated, which is
the right answer for a thing that no longer exists.

**The ledger is append-only in a way you can check.** Each entry's digest is
SHA-256 over the canonical JSON of its own fields *and* the digest before it, so
a rewritten, reordered or dropped entry breaks the chain and `ledger.verify()`
says so. Canonical JSON rather than joined fields because `detail` is free text:
under a `|`-separated payload, `task="a", detail="b|c"` and `task="a|b",
detail="c"` produce identical bytes, and one entry can forge another's digest.
Payload values are refused unless they have a single JSON form — no sets, no
bytes, no `NaN` — since a digest over an unstable rendering is computed but not
meaningful. There are no timestamps either: this package promises builds that
are byte-identical across processes and hash seeds, and a clock would be the one
field that could not be reproduced.

**A disproof is one entry, not a pile of them.** The `DISPROVED` record carries
the whole receipt — which ground failed, what evidence disproved it, which
auditor found it, every invalidated node *with the chain that explains it*, and
the preserved set — so `ledger.receipts()` answers the full event without the
graph the run was performed against. The per-node `INVALIDATED` entries stay:
the receipt is the atomic event, those are each node's own history, and both are
worth having. Saying three nodes fell proves half of selective invalidation;
the preserved set is the other half, so it travels in the same record.

A rejected assumption is never edited back to life. It stays rejected, and
`repair()` grounds the task on a replacement that records what it supersedes —
so `lineage()` still answers "what did we used to believe, and why did we stop".
Until that repair happens the task is *blocked*, not run: nothing executes on
ground the graph knows to be false.

### Durable identity — a checkpoint holds a name, not a callable

A task's `run` is a Python function, and a function dies with the process that
made it. So a task can also carry a *key*: a string a later process looks up in
its own `TaskRegistry` to get a callable back.

```python
registry = TaskRegistry()
registry.register_worker("auth.hash.argon2.v1", argon2_hasher)
registry.register_auditor("auth.hash.audit.v1", advisory_auditor)

runner = Runner("auth", registry=registry)
runner.task(
    "hash_password",
    worker_key="auth.hash.argon2.v1",
    auditor_key="auth.hash.audit.v1",
    assumes="Argon2 has no open advisory",
)
```

**One Runner, one registry.** The registry belongs to the Runner, and there is
deliberately no per-task or per-repair `registry=`. A checkpoint records the key
and not the table it was resolved through, so two tables in one plan would let
the same string mean argon2 in the file and bcrypt at runtime — and a restore
would pick whichever one it was handed. `rebind(new_registry)` is the one way to
change it: every task is resolved first, and only if all of them succeed are the
callables and the registry adopted together. A task inside a Runner rebinds
through that Runner for the same reason — pointing one task at a foreign table
is the same hole reached one task at a time. To run different code, use a
different key.

**A key never becomes an import.** The route from string to callable is a table
this process filled in deliberately — no module path, no qualified name, no
pickle, no fallback to a similar key. A deployment that never registered
`auth.hash.bcrypt.v1` cannot resume a checkpoint naming it, and says so with the
key in the message. `tests/test_registry.py` parses `execute.py` rather than
grepping it and asserts the module imports no loader and calls nothing that
turns text into behaviour.

**Plain callables still work.** `runner.task("temp", run=fn, assumes=...)` runs,
audits and invalidates exactly as before — it simply cannot be checkpointed,
and `require_checkpointable()` says which task stopped it and why. Passing both
`run=` and `worker_key=` is refused rather than reconciled: they are two answers
to "which code is this", and a checkpoint would record the wrong one.

**A key names one implementation for the life of the process.** Registering
`auth.hash.v1` twice is refused even when the callable is identical, because the
alternative is a bcrypt worker silently replacing an argon2 one at startup.
Workers and auditors keep separate namespaces — they carry different authority,
since an auditor may disprove an assumption and a worker may not — and a key
found in the wrong one says so rather than reporting a plain absence.

**The load-bearing case is a repair that outlives the process:**

```
argon2 → CVE → auditor disproves → repair to bcrypt → checkpoint → process dies
                                                                        ↓
                                              restore must bind bcrypt, not argon2
```

which is why repairing a keyed task with a bare `run=` is refused. The task
would run bcrypt while its checkpoint still said argon2, and the next restore
would faithfully resurrect the worker the CVE was about. `repair(...,
worker_key="auth.hash.bcrypt.v1")` moves both halves together, and
`ArgonToBcryptTest` walks the whole sequence across two runners with nothing
shared but the two strings — checkpointing at the dangerous boundary, with the
repair done and bcrypt not yet executed, since that is the state a crash is
likeliest to catch and the only one that proves `repair()` moved the durable key
rather than some later run doing it.

A key is text with no whitespace and no control characters — not a naming
scheme, since dots and versions are conventions you pick, but enough that the
identifier survives a log line, a shell word and a diff unsplit.

`registry.describe()` returns the keys and only the keys, so a deployment can
answer "this checkpoint cannot resume here because I lack
`auth.hash.bcrypt.v1`" without exposing a route back to the code.

### A restored ledger has to be *the* ledger, not *a* ledger

`ledger.snapshot()` writes a versioned container — `contextmesh.runledger` v1 —
carrying the schema, the version, every entry, and the head:

```json
{ "schema": "contextmesh.runledger", "version": 1,
  "head": "b881a2c2…", "entries": [ … ] }
```

The head is stored even though it is the last entry's digest, so a truncated
array is a contradiction the loader names rather than a shorter history it would
accept. `RunLedger.from_snapshot()` takes each entry **exactly as written** and
then checks it. It does not replay `record()`, and that distinction is the whole
design: replaying recomputes each digest from whatever the file says, so an
edited entry comes back with a freshly consistent digest and a chain that
verifies — the loader meant to catch the tamper would launder it.

What the chain proves, and what it does not:

```
edit an entry     → its digest no longer recomputes      REFUSED
delete an entry   → the next one's previous is wrong     REFUSED
reorder entries   → seq and previous both disagree       REFUSED
rebuild the lot   → internally perfect                   ACCEPTED
```

That last row is not a hole; it is what a hash chain is. Anyone who can rewrite
every entry can recompute every digest, and no amount of self-checking
distinguishes that chain from the original — a fully reforged ledger passes
`verify()`. What distinguishes them is a head you trusted *before* the file
could be rewritten:

```
        trusted head H
              │
   A → B  → C  → D          the history that ran
   A → B' → C' → D'         every entry and digest rewritten
              │
        head H' ≠ H
```

Pass it as `from_snapshot(data, expect_head=H)` and the forgery is refused,
because the honest history and its digest already exist, so producing a
*different* history ending in that same digest is a SHA-256 **second** preimage. Omit it and you get tamper *evidence* — modification, deletion,
reordering — but not continuity. This is why the restored head being the *exact*
committed head is the load-bearing invariant, rather than the restored chain
merely hashing correctly.

Everything the digest covers is validated before an entry is constructed: the
schema and version exactly, `seq` contiguous from 1, `round` a non-negative
integer, the event one this build knows, the nullable ids a string or null, the
payload canonical JSON, and both digests 64 lowercase hex characters. An unknown
field is refused rather than dropped — the digest covers a fixed set, so
anything extra is content the chain does not sign. The container is exact for
the same reason: a v1 file carrying `approved_by` or an `external_anchor` holds
meaning a v1 reader would silently discard, and a later version giving those
names semantics would then disagree with every v1 reader about the same bytes.

Duplicate JSON keys are refused too, at every depth — container, entry and the
signed `data` payload. Python's parser keeps the last value, others keep the
first, and some reject the document; a digest is only meaningful if every reader
agrees which JSON value it was taken over, and this file is meant to be
re-checkable by an implementation that is not this one.

### A restored plan has to schedule the same work

`runner.snapshot()` writes `contextmesh.execution` v1 — the plan's semantic
state, and nothing that can be derived from it:

```json
{ "schema": "contextmesh.execution", "version": 1,
  "plan": "auth", "round": 2, "tasks": [ … ] }
```

Each task carries its name, title, rationale, state, attempt, ground, `needs`,
`produces`, both durable keys, its decision and assumption ids, its output and
its artefacts. Three things are deliberately absent:

```
run / audit    rebuilt from the keys, through the Runner's registry
ready/blocked  recomputed — facts about a round, not about a task
the registry   deployment configuration, never file state
```

`TaskState` has no `BLOCKED` member for the same reason: being blocked is
decided every round from state, ground and dependencies. A file that stored a
schedule could assert one its own contents contradict.

**Task order is not sorted.** `_ready()` walks declaration order, so for two
tasks that become ready in the same round it decides which runs first. Sorting
the array would reorder a restored plan's execution without changing a field.
`plan` is stored for a related reason — the source node's id and every decision
id are derived from it — and it is checked on the way back in. The graph must
already hold the source that name derives to, and every decision the snapshot
restores must cite it:

```
snapshot.plan  ──slug──►  source:execution-plan-<name>
                                    ▲
                                    │ CITES
                          every restored decision
```

Change nothing but that one string and the old decisions come back under a newly
created source; the next rerun then writes `decision:other|hashing|v2`
superseding `decision:auth|hashing|v1`, and one lineage crosses two plans, each
internally consistent. Both halves are native invariants rather than invented
ones: `Runner.__init__` derives the source from `plan`, and `DecisionLog.decide`
always draws that `CITES` edge.

**References are closed at load, not left to fail later.** The assumption must
exist in the graph, the decision must be a `DECISION`, the artefacts must be
`ENTITY` nodes, and every `needs` must name a task in the same file. Two facts written twice have to
agree: `assumes` is what the task reports and `assumption_id` is what the
scheduler and the auditor read, so a file holding them in disagreement would
restore a plan that shows one ground and runs on another. Ids are content
slugged (sha1 of the statement), so a real binding cannot drift there by
accident — and `assumption_id` is not nullable, because `Runner.task` binds one
at declaration and an unbound task could only ever block.

**DONE is a claim about provenance**, so it carries obligations the other states
do not:

```
DONE    → must name a decision, attempt ≥ 1, ground not rejected,
          decision not invalidated
STALE   → may point at an invalidated decision — that is exactly what
          selective invalidation leaves behind
PENDING → no decision, no attempts
FAILED  → no decision, no attempts
```

That asymmetry is measured against the live lifecycle rather than assumed. Only
DONE is constrained: `_commit` creates the decision and only then marks the task
done, so done-without-one is unreachable — and dangerous, since `_ready` skips
DONE tasks and a restored plan would cache work as complete that the graph has
no record of. Over-constraining the others would refuse plans the engine really
produces.

**A plan that cannot be scheduled is not a plan.** Dependency cycles are refused
at load rather than left for `run()` to discover, and the check happens before
the Runner is constructed, so a refused snapshot does not even leave its source
node behind in the graph.

The load-bearing case is the awkward one — repaired but not yet rerun:

```
argon2 green → CVE → recheck → hashing, routes STALE; schema, tokens DONE
                                    ↓
                          repair hashing → bcrypt
                                    ↓
                              CHECKPOINT          bcrypt has never run
                                    ↓
                    restore → same ready set, same attempts
                                    ↓
                    run() → only the stale closure moves
```

A restored plan and one carried forward in memory are compared task by task
after `run()` — same states, same attempt counts, same outputs.

### One session, four companions, one commit

Session v1 held a graph and the resolver that reads it. **v2 adds the execution
plan and its run ledger**, so a restart brings back not just what is known but
what was being done about it:

```
session/
  session-000003.json   the immutable manifest that commits generation 3
  session.json          compatibility pointer to the latest manifest
  graph-000003.json     what is known
  resolver-000003.json  how a question finds it
  execution-000003.json the plan, mid-repair
  runledger-000003.json the record of it getting there
```

All companions land in fresh files and are renamed into place; then the
generation manifest is published under a previously nonexistent name. That
immutable manifest is the single atomic commit, so a crash anywhere before it
appears leaves the previous generation serving exactly as before. `session.json`
is a latest pointer for humans and older tooling, not the correctness boundary.
`execution` and `ledger` are
`null` rather than absent when a session carries no plan — a reader can tell
"no execution" from "field missing" — and a plan and its ledger are committed
together or not at all.

**The interesting failures are between the files.** Each companion already
refuses its own corruption, and every one of those checks can pass while the
four describe different runs. So the session boundary adds only what needs the
pair:

```
graph ↔ resolver     already: every resolved id is a node of the right type
plan  ↔ graph        already: references close, plan owns its namespace
ledger ↔ plan        new: every entry names a task the plan holds, and no
                     entry sits in a round the plan never reached
ledger ↔ graph       new: every node_id and assumption_id an entry cites
                     exists — a verified chain says nothing about that
```

Kept to what the engine guarantees. The ledger holds no entry for a task that
has not run, and many for one that has, so neither count is an invariant. Nor is
there an event-shape matrix: which events carry which ids is not universal, so
only the ids that are *present* are required to resolve.

**The manifest commits to a history, not a filename.** 7B showed a whole chain
can be rewritten and still verify, so `"ledger": "runledger-000003.json"` on its
own commits to nothing — swap the file for another internally perfect ledger and
the session loads a different history under the same generation. The manifest
records `ledger_head` and restores with `expect_head=`:

```
session.json
     ├── graph-N          ┐
     ├── resolver-N       │ one generation
     ├── execution-N      │
     └── ledger-N ────────┘  and it must end at ledger_head
```

To be exact about what that buys: **this is not authentication.** A writer who
can rewrite the ledger can rewrite the manifest beside it, and a plausible
resealed history whose ids all resolve is accepted — there is a test asserting
that, so nobody reads the guard as stronger than it is. What the head stops is a
change to *one* companion quietly redefining a committed generation.

**A directory cannot carry a registry.** A key means something only because a
running process was configured to say so, so a session holding an execution is
restorable only by a deployment that brings one — `Session.load(path,
registry=...)`, and a clear refusal naming the missing key otherwise.

A **v1 directory still loads**, as a session with no execution, and the next
save upgrades it. A version from the *future* is still refused: this build
cannot know what it would be dropping.

### The standalone control layer

`examples/assumption_control_layer.py` is the original single-file sketch of
these ideas — a seven-node executor that carries an assumption on an edge,
rejects it mid-run, and re-executes only what stood on it:

```
INTENT → DECOMPOSE → WORKER → AUDIT → DRIFT → LEDGER → ROOT
             ↘
              BRANCH_C1 → BRANCH_C2   (unrelated, must survive)
```

```bash
python examples/assumption_control_layer.py
# OVERALL: ALL ACCEPTANCE CHECKS PASSED
```

It runs on its own with nothing imported from the package, and it is kept
because the whole idea fits on one screen there. It is a sketch, not the
implementation: its `DRIFT` node is a stub that reports alignment without
checking anything. `contextmesh/assumptions.py`, `contextmesh/decisions.py` and
`contextmesh/execute.py` are the version the rest of this repository uses, and
the one to read if you want the behaviour rather than the shape.

## Graph health

The states a long-running graph drifts into, none of which are errors:

```
$ python -m contextmesh health
     untyped_edges               0  edges stored without a type from GRAPH.md
   ! unresolved_entities       301  near-miss mentions dropped at RESOLVE
   ! missing_relationships      10  resolved entities no claim ever mentions
   ! open_contradictions         4  claims contradicted with no decision on top
   ! dead_ends                   9  entity_unresolved=3; wrong_node_type=4; pruned_too_early=2
     invalidated_nodes           3  work invalidated by a rejected assumption, kept for audit
```

`unresolved_entities` counts *near misses* — something in the graph nearly
matched. The extractor offers the resolver every phrase in every sentence, and
most of them are simply not entities; counting those would turn the signal into
a measure of traffic.

## The dashboard

```bash
python -m contextmesh export --inline
```

Open `dashboard/index.html` with `start` on Windows, `open` on macOS, or
`xdg-open` on Linux.

Header stats, the build path, the live graph, the hop budget, the edge ledger,
walk-vs-flat token cost, the traversal grid and the dead-end ledger — each panel
is a query against the graph. [`dashboard/README.md`](dashboard/README.md) maps
every panel to the call that produces it and is explicit about what is a live
stream and what is a snapshot. [`docs/DASHBOARD_SPEC.md`](docs/DASHBOARD_SPEC.md)
is the frame-by-frame record of the original capture.

## Persistence

```python
graph.save_json("graph.json")
graph = ContextGraph.load_json("graph.json")
```

A versioned snapshot (`contextmesh.graph` v1) that restores the same graph, not
a graph that looks the same. Nodes carry their actual embedding vector, not a
`embedded: true` flag; provenance spans come back as tuples; `walks`,
`traversals`, `pruned` and `invalidated` all survive.

**A snapshot is untrusted input, and it fails closed.** Every edge is restored
through `add_edge`, so `GRAPH.md` typechecks a file exactly as it would a live
write, and `_out`, `_in` and `_edge_key` are rebuilt rather than persisted. A
loader that writes straight into the internal dictionaries is faster and admits
graphs the live API would refuse — which makes the ontology a convention that
holds only until something is saved.

Every field the writer emits is **required** on load — absence is corruption,
not a default. Dropping `invalidated` would restore a node the graph had
deliberately killed; dropping `embedding` would restore one that answers
differently. Defaults belong to a schema version with an older shape to migrate
from, and v1 has none.

`null` is accepted only where the schema writes it — `provenance`, its `span`,
`embedding`, `assumption_id`, `supersedes`, `superseded_by`, `rejected_at_build`.
Everywhere else a null is corruption, and normalising it to an empty list would
discard what the field named: an assumption whose `evidence_ids` arrived as
`null` would restore with nothing recorded as having disproved it.

References between records are checked once every record exists: a lineage that
names a missing assumption, a supersession only one side agrees with, an edge
grounded on an assumption that is not there, evidence that is not a node. Each
of those loads cleanly and fails much later — `lineage()` raising `KeyError` on
a graph that has been serving for an hour.

Fields are checked rather than coerced, because coercion is not harmless here:
`bool("false")` is `True`, so a malformed flag would quietly turn a live node
into an invalidated one; `list("abc")` would turn a string into a
three-element "vector"; and `bool` being a subclass of `int` means a count
written as a flag would arrive as `1`. The suite corrupts a good snapshot
97 ways and requires each to be refused — a counted figure, not an
estimate: `tests/test_persistence.py` builds and rejects that many distinct
corrupted snapshots across 55 test methods.

**Records are emitted in insertion order, never sorted.** That is load-bearing.
The walker's frontier is a heap whose tie-breaker is an insertion counter, and
the lexical fallback iterates `graph.nodes`, so two equal-cost branches are
decided by which was added first:

```
edges added alpha-first: answer = alpha  (hops 2, score 0.2750)
edges added bravo-first: answer = bravo  (hops 2, score 0.2750)
```

Same facts, same score, different answer. Sorting the arrays would reorder those
lists on reload and change answers without changing anything true, so
`tests/test_persistence.py` builds that tie deliberately and asserts a round
trip preserves it. Sorting the *keys* of each JSON object is safe, and
`save_json` does that — along with `allow_nan=False`, because a snapshot other
parsers refuse is not durable.

One thing the snapshot **cannot** represent: disagreement. An assumption's
`status` and `version` are stored on the record and projected onto its node so a
walk can read them without a second lookup. The loader refuses a file where the
two disagree — and enforcing that surfaced a real bug, since two paths bumped
`version` on the record alone and left the mirror stale. `graph.sync_assumption`
is now the one place that writes it.

The execution state in `Runner` is not part of this format — `contextmesh.graph`
v1 stays closed to it — but it is not unpersisted either. It has its own
versioned format, `contextmesh.execution` v1, and a run ledger alongside it;
`Runner.snapshot()` / `Runner.load_json()` round-trip a plan against a live
graph, refusing to restore a task that says DONE while its decision has since
been invalidated. See **Sessions** below for how the two combine.

### Sessions — a graph *and* the resolver that reads it

```bash
python -m contextmesh_mcp --demo --rounds 8 --save ./session   # write one
python -m contextmesh_mcp --session ./session                  # inspect it
```

A restored graph answers questions only if something can turn *"pgvector"* into
`entity:pgvector-6db608`, and that is the resolver's job. So a session is a
directory of separately versioned files:

```
session/
  session-000003.json    contextmesh.session  v2  — the immutable commit manifest
  session.json           latest-manifest pointer for humans and older tooling
  session.lock           the writer lock, created once and never deleted
  graph-000003.json      contextmesh.graph    v1
  resolver-000003.json   contextmesh.resolver v1
```

Three formats rather than one because graph snapshot v1 is **closed**. Query
resolution is not graph state, and folding the resolver in would mean reopening
a settled format every time the resolver learns a new field. The session file is
the join.

**A save never writes to a file the current manifest names.** Writing three
files in sequence is safe the first time and unsafe every time after: crash
between the graph and the resolver and the directory holds a new graph beside an
old resolver — a pairing that never existed, and one that can still pass every
check made on it. So each save writes a whole new *generation* under new names
and commits it by publishing a new immutable `session-000NNN.json` manifest:

```
crash before manifest publication  ->  the previous generation is still named, intact
crash during publication           ->  the old manifest or the new one, never half
crash after publication            ->  the new generation is named, and complete
```

Superseded files are swept after the commit, so an interrupted save costs disk
rather than correctness. The guarantee is reader-visible and tested as such: a
ContextMesh reader holding the old committed manifest open across a save still
reads the previous generation whole while another process publishes the next one.

**Generations are atomic against a crash. They do nothing against a second
writer**, and that failure is nastier because nothing about it looks torn. Two
processes reading generation 5 both choose 6 and overwrite each other's
companions. Worse, one can commit 6 and sweep while the other has already
written 7 but not yet swapped — leaving a manifest that is atomically valid and
names files the first process just deleted:

```
end   : ['session.json']
manifest -> generation 3   graph-000003.json exists: False
```

That is a real reproduction, staged across two processes, and it is now a test.
So a save takes the directory's writer lock for the **whole** transaction —
from reading the current generation to sweeping the superseded one. Locking only
the swap would still allow both races above. The lock is `flock`/`msvcrt`, held
by the kernel rather than by convention, so it is released when the holder dies
however it dies; there is no stale-lock heuristic to guess wrong under exactly
the load that made a save slow. A second writer is refused, not queued:

```
SessionLockedError: … is already being written by pid 4127 on host; one writer at a time
```

Readers never take it. The manifest swap already gives them a consistent view,
and a read that had to wait behind a checkpoint would be a worse trade.

Serialising writers stops corruption; it does not stop the second writer from
silently discarding the first one's work. So a session that has a home also
checks that home before committing: if the directory has moved on since this
session last committed there, the save is refused rather than clobbering. The
`Checkpointer` treats lock contention as a skipped commit — the mutation stays
pending for the next one — and counts it, because a server that keeps losing
the race is a configuration problem worth being able to see.

Multi-writer *coordination* is out of scope: two servers on one directory get
one clear refusal each time they collide, not a merge.

**Untrusted on the way out, too.** The read side refuses a companion that
resolves outside the directory. Making the directory *writable* opened the
mirror-image hole: writing by name follows whatever is already under that name,
so a directory handed over with the next generation's filename already present
as a symlink would have the next save write straight through it — and with
`--checkpoint every-ask`, merely asking a question is the trigger. Four names
were reachable that way, the lock file among them.

One mechanism rather than four patches: every write lands in a fresh `O_EXCL`
file under a random name — which cannot already exist, so cannot already be a
link — and is moved into place with `os.replace`. Rename replaces a symlink
*itself* instead of following it, so a planted link is destroyed and the file it
pointed at is never touched. The same rename is what makes the write atomic, so
the manifest's commit and the symlink defence are the same line of code.

The lock is the exception, because it has to keep one inode for its whole life —
that inode is what the kernel lock attaches to, and replacing it each save would
hand two processes locks on two different inodes. So it is opened `O_NOFOLLOW`
and checked with `fstat` to be a regular file, which also rules out a fifo or a
device left under the name.

**A checkpoint must not make a live session look broken.** Readers take no lock
and do not need one for correctness — the manifest swap is atomic and committed
companions are immutable, so a pair that reads successfully is always a coherent
generation, at worst a slightly old one. What the swap alone does not cover is
the *sweep*: a reader holding the manifest for generation 5 can find
`graph-000005.json` already deleted and fail on a perfectly healthy directory.
So a read that loses that race is re-read rather than reported, and the retry
fires only when the directory's generation actually moved during the attempt —
which keeps a genuinely missing file failing immediately, with its own message,
instead of after a wait.

**The resolver cannot be rebuilt from the graph, and that is the point.**
`resolve()` writes a scored match back into its alias table, so a run learns
surface forms no entity label contains: in the bundled demo, 72 of the
resolver's 120 aliases are learned that way rather than registered. Restart
without them and the same mention costs a full block scan and a scored match
instead of a table hit.

`blocks` is persisted rather than rebuilt, which is where the analogy with
`_out`/`_in` breaks. Those are a function of the edge list. `blocks` is a
function of *how* an alias arrived: `register` adds block keys for every name it
is given, a match learned at query time adds none, and the alias table does not
record which is which. Rebuilding from `canonical` alone loses keys; rebuilding
from `canonical` plus `aliases` invents keys the resolver never had. Blocks
decide the candidate set, so either version silently changes what resolves —
`tests/test_session.py` measures both and shows them differing.

Walker settings ride along in `session.json` for the same reason: restore a
session saved with `hop_budget=3` into the default `6` and the same question
comes back with a different answer, with nothing in the output to say why.

**A session directory is untrusted input.** `Session.load` refuses one it cannot
faithfully restore — wrong schema, unreadable version, a missing or mistyped
field, a `rounds` written as `true`, a policy naming an edge type this ontology
does not have, a manifest whose filenames run ahead of its generation counter.
The two filenames must be plain names, so a directory you were handed cannot
point the loader at `../../etc/passwd`, and the files they name must resolve
*inside* the directory — a correctly named `graph-000001.json` that is a symlink
to somewhere else is refused too.

Three checks live here because neither file can make them alone. Every entity
the resolver resolves *to* must exist in the graph, must be an entity, and must
carry **the same label**. The first two fail loudly; the third is the quiet one.
`Resolver.canonical` is not a display string — it is scored against every
mention that reaches `near_miss` — so a resolver holding `entity:pgvector ->
"HNSW"` over a graph holding `"pgvector"` keeps resolving, resolves differently,
and both files are individually valid. The resolver's own log is held to the
same standard: a record naming an id must carry the label that id has, and a
resolution has both an id and a label while a miss has neither.

### Checkpoints — because asking is a write

A saved session that is never written back is a durable *starting* snapshot, not
a durable session. Asking a question moves `node.walks` and `edge.traversals`,
grows the resolver's log, and teaches it aliases it did not have — so without a
checkpoint every question asked after startup is discarded on restart, silently,
including exactly the learned aliases the format exists to keep.

```bash
contextmesh-mcp --session ./session                       # commit after each ask
contextmesh-mcp --session ./session --checkpoint on-exit  # commit once, on a clean stop
contextmesh-mcp --session ./session --checkpoint never    # serve it, never write to it
```

`every-ask` is the default and it is the expensive one on purpose: one full
serialisation per question is a real cost, and a silently lost question is a
worse one. `on-exit` is cheaper and worth nothing if the process is killed
rather than asked to stop. `never` is the honest choice for a directory you were
handed and do not own. `mesh_ask` is the only tool that triggers a commit —
which is asserted at the call site rather than inferred, so a sixth tool forces
a decision about whether it writes.

**What a restart still does not restore**, stated plainly because the suite pins
it: the walker's in-process walk list, and the assumption ledger's event log.
`mesh_lineage` and `mesh_blast_radius` come back identical — both read the
graph's own assumption records — and every *count* in `mesh_health` restores
exactly. The one field that does not is health's `dead_ends` signal, computed
from the walk list and so absent until the new process has walked. That is a
cold start, not a lost capability, and `SurfaceEquivalenceTest` asserts it is
the *only* one.

## MCP — controlled writes, narrow authority

```bash
pip install 'contextmesh-graph[mcp]'
contextmesh-mcp --demo               # a graph rebuilt for this process
contextmesh-mcp --session ./session  # a graph that outlives it
```

A session that contains a durable execution plan also contains worker and
auditor *keys*, never callables. The deployment that serves that session must
therefore bring the `TaskRegistry` that gives those keys meaning. The plain
console command deliberately has no way to invent or import implementations
from checkpoint text; embed the launcher and pass the deployment-owned registry:

```python
from contextmesh.execute import TaskRegistry
from contextmesh_mcp.server import main as mcp_main

registry = TaskRegistry()
registry.register_worker("auth.hash.bcrypt.v1", bcrypt_worker)
registry.register_auditor("auth.hash.audit.v1", advisory_auditor)
raise SystemExit(mcp_main(["--session", "./session"], registry=registry))
```

If the session has execution state and no registry is supplied, or a required
key is absent, startup fails closed before the MCP transport starts. Registry
selection is deployment configuration; no MCP request, checkpoint field, module
path, or import string can choose executable code.

An MCP server over the graph, so an agent can query it as memory instead of
being handed chunks. Seven read tools — `mesh_ask`, `mesh_get_node`,
`mesh_health`, `mesh_lineage`, `mesh_blast_radius`, `mesh_explain_as_of`,
`mesh_reconstruct_decision` — plus four controlled structural writes —
`mesh_submit_evidence`, `mesh_recheck`, `mesh_repair`, `mesh_resume` — and
`contextmesh://schema`, `health`, `session`, `assumptions`, and templates for
`node/{id}` and `assumption/{id}`.

The last two answer from the graph *as it stood* rather than as it is. They
are wrappers and nothing else: every temporal rule lives in
`contextmesh/temporal.py` and `contextmesh/reconstruct.py`, and
`tests/test_mcp.py` asserts the tool returns exactly what the engine returns,
so a rule cannot start living in two places.

```json
{
  "mcpServers": {
    "context-mesh": { "command": "contextmesh-mcp", "args": ["--demo"] }
  }
}
```

**Writes exist now, and they are deliberately narrow — one boundary stays
permanent regardless.** *No rejection by client fiat* was never a phase to
grow out of: under GRAPH.md rule 7 an assumption falls only to evidence
contradicting it, produced by an auditor that looked at the world. None of the
four write tools lets a caller name an assumption and have it rejected
directly — `mesh_submit_evidence` stores a raw observation and expresses no
verdict; `mesh_recheck` asks the registered auditors to interpret current
state and the client supplies none; `mesh_repair` can select only
deployment-owned `TaskRegistry` keys already registered — no callable, module
path or import string crosses MCP; `mesh_resume` delegates to the native
selective scheduler, rerunning only pending or stale work. Submitting evidence
and letting ContextMesh decide is the different thing a rejection tool would
not have been.

Each write commits through `writes.commit_mutation`: the mutation runs against
a lossless in-memory clone of the session first, and only that clone is made
live once its manifest has committed — so a failed write never leaves the
served session half-mutated, and a controlled write refuses an ephemeral
`--demo` session outright, since there is nothing durable to commit it into.

`mesh_blast_radius` still sits exactly on the rejection line as a pure dry
run. It answers *what would fall if this assumption were false* — the
closure, the reason chain for each node, and the preserved complement — and
changes nothing.

**The read boundary is not "the graph is unchanged".** Asking a question through
one of the seven read tools moves walk telemetry — `node.walks`,
`edge.traversals` — and PRUNE later drops what nothing walked, so a walk is a
write by design. The invariant tested instead is that no *read* changes graph
structure, ontology state, assumptions, supersession or invalidation, while
telemetry is free to move; that is a narrower guarantee than the four
structural write tools make, which commit exactly the mutation their name
describes and nothing else. `tests/test_mcp.py` asserts both halves
separately.

**It can now serve a saved session**, and it makes you say which:

```bash
contextmesh-mcp --demo --rounds 8 --save ./session   # write one
contextmesh-mcp --session ./session                  # serve it
```

`--demo` used to be what you got by saying nothing. It has to be said now,
because the alternative is no longer *nothing* but a real session on disk, and
silently serving a throwaway graph when someone meant to serve theirs is the one
failure the format exists to prevent. `contextmesh://session` reports which it
is and whether anything a client reads outlives the server.

```json
{
  "mcpServers": {
    "context-mesh": {
      "command": "contextmesh-mcp",
      "args": ["--session", "/path/to/session"]
    }
  }
}
```

What a client does to a served session now survives it — see **Checkpoints**
above, and that now includes execution state: a session's execution plan and
run ledger persist in their own versioned format, `contextmesh.execution` v1,
alongside the graph and resolver. What is still narrow is *how* a client can
change belief: `mesh_submit_evidence` is the one write that adds a node, and
it is deliberately confined to a raw evidence observation backed by a source —
nothing crosses MCP that names an edge, an assumption verdict, or a rejection
target directly. Structure and belief still change only through the engine
interpreting what was submitted, never through a client-supplied verdict.

The core stays untouched by all of this: `contextmesh` remains Python 3.9+ with
zero dependencies, the MCP SDK arrives only through the `[mcp]` extra, and
`contextmesh_mcp.server` is the only module that imports it — which is why the
safety tests above run on the whole 3.9–3.13 matrix with nothing installed.

## Layout

```
GRAPH.md                     the ontology, parsed at import
contextmesh/
  ontology.py                GRAPH.md → the schema every write is checked against
  model.py  graph.py         typed nodes, typed edges, no untyped code path
                             and the versioned snapshot it saves and reloads as
  resolve.py                 entity resolution — one id per real-world thing,
                             and the alias table it learns and reloads
  pipeline.py                CHUNK → EXTRACT → RESOLVE → LINK → EMBED → PRUNE
  traverse.py                walks, evidence paths, the four dead-end reasons
  assumptions.py             versioned assumptions, blast radius, rejection
  execute.py                 re-runs exactly the closure an invalidation felled
  decisions.py               append-only decision history
  temporal.py                source time vs processing time, and the graph as
                             it stood on a given date
  reconstruct.py             answering from that past graph, and walking back
                             from a decision to the reasons it stood on
  health.py                  the signals that make a graph quietly useless
  metrics.py                 the dashboard payload
  corpus.py  demo.py  cli.py the worked example
contextmesh_mcp/             MCP server: seven read tools, four controlled
                             structural writes (optional extra, 3.10+)
  session.py                 durable sessions: graph + resolver + execution
                             plan + run ledger, on disk
  tools.py  resources.py     read tools, plain Python over the engine; no
                             SDK import
  writes.py                  controlled structural writes; commits through a
                             lossless session clone, no SDK import
  __main__.py                write or inspect a session without the SDK
  server.py                  the only file that needs the SDK
dashboard/                   the rebuilt dashboard
docs/                        the capture spec and the architecture notes
examples/                    the original standalone control-layer sketch
tests/                       805 tests over the invariants above
```

## Staying publishable

This repository is public and carries nothing internal. `ops/leak-check.sh`
makes that checkable rather than asserted — it scans tracked content for
secrets, email addresses and machine-local paths, and commit *metadata* for
private session URLs and non-no-reply author addresses, which is where leaks
of this kind actually hide.

```bash
ops/leak-check.sh              # the whole history
ops/leak-check.sh origin/main  # only what a branch adds
```

CI runs the second form on every pull request.

`ops/accepted-authors.txt` records addresses that are public on purpose, so the
check can tell a decision from an accident instead of failing forever on a known
case. An address that is not listed and not a no-reply still fails.

## Where it fits

ContextMesh complements retrieval and agent memory rather than replacing them.
Retrieval supplies the source material; memory supplies history; ContextMesh
supplies the typed relations and the provenance that make an answer auditable —
and, on this corpus, does it for about 0.7% of the tokens flat top-k would
spend, because it sends the nodes on the path instead of the top forty chunks.

> The graph should remember not only what is connected, but why it is connected.

## Licence

MIT — see [LICENSE](LICENSE).
