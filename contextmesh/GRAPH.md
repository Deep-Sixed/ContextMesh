# GRAPH.md — the ontology file

> Read on every write. If a write does not typecheck against this file, it does
> not enter the graph.

This is not documentation of the code; it is the schema the code loads. See
`contextmesh/ontology.py`, which parses this file at import time.

## Reading this file

**Authority.** This file is authoritative for the node types, the edge types and
their legal pairs, and the rules below. Where an implementation and this file
disagree about any of those, this file is right and the implementation is a bug.

**What is mechanically enforced.** `ontology.py` parses the `Type` and `Symbol`
columns of the node table, the `Must carry` column of the node table, and the
`Edge`, `Legal pairs` and `Invalidation` columns of the edge table; every write
is checked against them, and `tests/test_ontology.py` asserts that the
`NodeType` and `EdgeType` enums equal what this file declares. `Means` is prose
and is not parsed. The rules below are enforced, or not, one at a time — the
*Declared, not yet realized* section names the ones that currently are not.

**What `Must carry` means.** It is a *minimum*, not an exhaustive list: a node
may carry more. It is satisfied by a node `attrs` key **or** by a `Node` field of
that name — `provenance` is a field, not an attribute. It requires the value to
be **present**, not to be well-formed: `contextmesh/execute.py` mints sources
with `retrieved_at="at plan time"`, which satisfies the presence requirement
defined here, and the temporal layer separately classifies such a source as
`UNDATED` rather than guessing a date. `ContextGraph.add_node` enforces this at
the write boundary — presence only, nothing more — and refuses a node missing a
required name with `OntologyError`. Every path that constructs a node,
including restoring a persisted snapshot, goes through `add_node`, so this is
one check rather than one per caller.

**What edge-level assumption binding means.** `Edge.assumption_id` makes one
relationship conditional on one assumption: the edge holds only while that
assumption stands. Rejecting the assumption invalidates the bound edge
directly — `apply_invalidation` marks it regardless of what else falls — but
binding an edge is not a second way to reach a node. `src` and `dst` fall only
if rule 2's propagation reaches them along an edge of a propagating type; a
bound `mentions` or `supports` edge, say, does not pull its endpoints down
just because it was bound. A node that itself needs to fall when an
assumption falls has to say so explicitly with `depends_on` — rule 2 is the
one mechanism for node-to-node invalidation, and edge binding does not open a
second one. `AssumptionLedger.justifies` enforces the binding side of this at
the write boundary: the assumption must exist and be active, the edge must
exist and be live, binding the same pair again is a no-op, and binding an
edge already bound to a *different* assumption is refused rather than
silently replaced. `ContextGraph.add_edge` takes no `assumption_id`, so it
is not a second, unguarded way to bind one; only snapshot restoration sets
the field directly, and it validates what it restores on its own terms.

**What walk-driven PRUNE means.** Rule 5's second half drops a node only once
an observation window has closed: `Pipeline.prune_unwalked_nodes` takes the
`Walker` that produced the telemetry, and is a no-op below
`MIN_OBSERVATION_WALKS` walks in that window, so one question failing to reach
a node is never mistaken for the corpus having judged it irrelevant.
`Node.walks` — incremented once per node an actual walk visits, in
`Walker.walk` — is the only signal a node's own eligibility is judged on;
degree, inherited unchanged from the build-time half this pass has always
shared it with, is the one other factor, and nothing about a node's label,
age, or type enters the decision beyond `source`, `assumption`, and
`evidence` nodes being preserved outright — an assumption nothing has
questioned yet is not thereby irrelevant, and evidence is what rule 7
requires an audit be able to find. A node this pass drops cannot be revived
by a later walk: `Walker.seed` will not seed a pruned node and
`out_edges`/`in_edges` will not traverse into one, so its `walks` count is
frozen at whatever it was the moment it fell. A node already not live for a
different reason — invalidated by a rejected assumption — is left alone
rather than separately marked pruned, so this pass never creates a second
state change on top of one that already happened. The demo's export path
(`contextmesh/demo.py`) never opts in, so this pass changes nothing about the
dashboard unless a caller explicitly asks for it.

**What a node/edge id collision means.** `slug` (`contextmesh/model.py`) truncates
its digest to keep ids short, so it is not injective: two different node
labels, or two different `(src, type, dst)` edge triples, can derive the same
id. `add_node` and `add_edge` are also the merge path for a genuine repeat
observation — the same claim text extracted from a second document, the same
`mentions` edge asserted twice — so a shared id has to be told apart from a
coincidence before it is treated as one. For a node, "the same thing" means
the existing node's `type` and `label` agree with the incoming call; `attrs`
is never part of that comparison, since a repeat observation may legitimately
carry new metadata the first one did not. For an edge, "the same thing" means
the existing edge's `(src, type, dst)` triple — not merely its id — matches.
A collision found either way is refused with `OntologyError` before anything
is written: no node's `label`/`type` is silently overwritten, no edge's
`(src, type, dst)` is silently repointed, and none of the internal indexes
(`edges`, `_edge_key`, `_out`, `_in`) are touched. `from_dict` inherits this
for free, since every node and edge it restores goes through `add_node` and
`add_edge` the same as a live write — but a snapshot's own `nodes`/`edges`
lists are also checked for two rows sharing a literal id before either method
ever runs, since `to_dict` can never produce that either. This is
collision *detection*, not a stronger id scheme: `slug`'s truncation is
unchanged, and a caller that keeps minting distinct content into the same
id space will eventually have a write refused rather than silently lose one.

**What a decision's identity is.** A `decision` node is an immutable event,
never a content address: two calls to `DecisionLog.decide` with byte-identical
`title` and `rationale` are two decisions, not one being re-observed, because
"we chose this again" and "we never stopped choosing this" are different
facts and only the first is `supersedes`'s job to record. A call made without
an explicit `id` therefore always mints a fresh id, discriminated against
*graph* state rather than `DecisionLog`'s own history — the smallest sequence
number whose derived id is not already a node in this graph — so a repeated
title still cannot collide with its own predecessor after a fresh
`DecisionLog` is constructed over the same graph, which is exactly what a
restored `Runner` does when no `decisions=` is given. An in-memory
discriminator that instead counted this object's own records would forget
every decision minted before it existed and hand out an id that object had
never seen but the graph had — an id collision one caller reuses, not two
callers agreeing.

An explicit `id` means something narrower: not "this is the decision this
content names" but "this exact call may have already happened — tell me if
it did." The decision node itself carries this call's evidence, in two attrs
that ride along with `rationale`: `decision_identity` (`"auto"` or
`"explicit"`) and `decision_payload_digest`, a fingerprint of the node's
graph-visible content — `title`, `rationale`, the provenance `source_id`
kept as its own field, and the deduplicated, order-independent target sets
of the edges `decide` creates beyond that source (citation, support,
dependency, production, supersession). Keeping `source_id` separate from
the additional-citation set matters: reconstructing both as one combined
set from a node's out-edges cannot tell "the edge that is the provenance
source" from "the edge that is an additional cite to the same target", so
a primary source and an additional cite could otherwise be swapped without
the fingerprint noticing — a different agreed payload producing the same
digest. This is graph state, not a `DecisionLog`-local table, so it
survives a snapshot round-trip and a fresh `DecisionLog` the same way the
id-freshness check does.

An explicit `id` naming an existing node is an idempotent retry only when
that node is a decision minted with an explicit id of its own — naming an
auto-minted decision, or any non-decision node, is refused, because it is
not this call's event to retry. That check does not stop at reading the
`decision_identity` attr, either: every auto-minted id lands in a reserved
namespace (the `decision:auto:` slug prefix) that no explicit id may ever
use, and `decide` refuses up front to hand out a brand-new explicit id that
collides with it — so membership in that namespace is a permanent fact
about an id string, settled the moment it is minted, never re-derived and
never dependent on anything else in the graph. A decision whose id lands in
that namespace is never accepted as an explicit retry target, even when its
`decision_identity` attr is tampered to say `"explicit"`. This is what
closes the gap a digest check alone cannot: an auto-minted decision's
stored digest is already correct for its real content, so flipping only its
mode label leaves nothing for a digest comparison to disagree with. It is
also why the namespace test cannot be a search bounded by the graph's
current size — an early version of this tried exactly that (recomputing
`slug(f"{title}|{n}", "decision")` for `n` up to the node count), and a
bound that changes as the graph grows is not stable: an explicit id one
write legitimately accepted could fail the identical check moments later,
once that very write (or a snapshot round-trip of it) changed the bound,
with no tampering involved. A plain prefix has no such bound to drift.
Once identity agrees, the retry check compares
the incoming call against the fingerprint of the *existing node's actual
provenance and edges*, not against whatever its attrs claim. The same `id`
with content that produces the same fingerprint is a true no-op that
returns the existing node untouched, and the same `id` with any different
content is refused with `OntologyError` before a single node or edge is
written — the id is not free to start meaning something else partway
through a run, or after a restart. This is an idempotency key, not a
stronger identity scheme: it only protects a caller who reuses one id on
purpose, the same way `add_node`'s type/label check only protects against
`slug`'s truncation, and the two checks answer different questions (one
caller-declared, one derived) that can both apply to the same write.
Because the check reads the node's actual edges, an explicit-id decision's
fingerprinted edges are fixed once `decide` returns: `add_edge` refuses a
new `cites`, `derived_from`, `produces`, `supersedes`, or `depends_on` into
an assumption out of it, since accepting one would make the next load
refuse a correct snapshot and the next identical retry refuse the original
call. (A `depends_on` into another decision is a task ordering, not part of
the decision, and is still added freely.) A time-travel projection that
drops such an edge because its other end came later re-derives the digest
for what it kept, the same way it rewinds assumption state.

A decision's immutability does not stop at `DecisionLog`, and neither does
the trust boundary around its identity metadata. `add_node` refuses to mint
a brand-new decision node at all — only `DecisionLog.decide` and the
snapshot-restoration path in `from_dict` may create one — so `attrs` like
`decision_identity` and `decision_payload_digest` can never be handed to a
node that did not genuinely come from `decide`; a caller with only the
public API cannot forge a decision that a later retry would mistake for a
prior event. `add_node` also refuses a second call for an *existing*
decision id even when type and label match, unlike a claim or entity, which
legitimately merge new attrs on a repeat observation — a decision has no
such second sighting, so reaching that collision means some other caller is
trying to rewrite a decision's rationale through the generic merge path, and
that is refused rather than silently allowed to succeed. And because a
snapshot is untrusted input regardless of what created it, `from_dict`
independently recomputes every `"explicit"` decision's fingerprint from its
restored provenance/edges and refuses to load a file where the stored digest
disagrees with reality, and separately refuses to load one where an
`"explicit"`-labeled decision's id could have been auto-minted for its own
title — a hand-edited snapshot cannot claim a decision holds content its
actual edges do not support, nor turn a real auto-minted event into an
explicit one by editing a single attr.

`decide` is atomic under either path: a decision and its edges (citation,
support, dependency, production, supersession) either all land or none do,
rolled back the same way `AssumptionLedger.assume` rolls back a
partially-justified assumption.

**Code is evidence, not authority.** A behaviour may be written into this file
when the implementation *structurally guarantees* it — when no caller can make
it false without changing the implementation's own contract. A behaviour that is
merely one unconstrained implementation choice among several does not become
normative by having been written first; it requires a decision recorded here.

## Node types

| Type | Symbol | Means | Must carry |
|---|---|---|---|
| Entity | `entity` | A resolved real-world thing. One id per thing, forever. | `canonical`, `aliases` |
| Claim | `claim` | A statement lifted from a source. | `provenance` |
| Source | `source` | A document, span, run, or artifact that text came from. | `origin`, `retrieved_at` |
| Decision | `decision` | A choice that was made, with the reasoning attached. | `rationale`, `provenance` |
| Assumption | `assumption` | Something taken as true so work could proceed. Versioned. | `status`, `version` |
| Evidence | `evidence` | An observation that bears on a claim, decision, or assumption. | `kind` |

## Edge types

An edge is legal only if the pair `(source type, target type)` appears in its row.

`Invalidation` is rule 2's propagation direction, machine-readable: `backward`
means the edge is walked from target to source when an assumption falls (the
thing that depends falls when its ground falls), `forward` means source to
target (an artefact falls with the decision that made it), and `none` means
the edge never propagates a fall. `contextmesh/assumptions.py` reads this
column through `ontology.py` rather than restating it, so the two cannot
disagree.

| Edge | Legal pairs | Invalidation | Means |
|---|---|---|---|
| `mentions` | source→entity, claim→entity | none | The text refers to this resolved entity. |
| `derived_from` | claim→source, decision→claim, entity→source | backward | This exists because that exists. |
| `cites` | decision→source, claim→claim | none | Explicit reference made by the author. |
| `contradicts` | claim→claim, evidence→claim, evidence→assumption | none | Both cannot hold. |
| `supports` | evidence→claim, claim→decision, evidence→decision | none | Raises confidence in the target. |
| `depends_on` | decision→assumption, decision→decision, claim→assumption | backward | Target's failure invalidates the source. |
| `produces` | decision→entity, decision→source | forward | The decision brought the target into being. |
| `supersedes` | decision→decision, assumption→assumption, claim→claim | none | Replaces, without deleting. |
| `justified_by` | decision→evidence, claim→evidence, assumption→evidence | none | The reason this was allowed to stand. |
| `resolves_to` | entity→entity | none | An alias node folding into its canonical id. |

## Rules

1. `untyped_edges == 0`. There is no constructor for an edge without a type.
2. Invalidation propagates along three edges and no others, and direction
   matters: `depends_on` and `derived_from` are followed *backwards* (the thing
   that depends falls when its ground falls), `produces` is followed *forwards*
   (an artefact falls with the decision that made it). `mentions`, `cites`,
   `supports` and `supersedes` never propagate — that is what makes
   invalidation selective rather than a purge.
3. `supersedes` never deletes. Decision history is append-only.
4. Every `claim` and `decision` carries provenance to a `source`. A node without
   a path to a source is an orphan and is reported by graph health.
5. A node is walkable only after `PRUNE`. `PRUNE` has two halves: the build-time
   pass drops what nothing *linked* (degree zero), and a second pass,
   `Pipeline.prune_unwalked_nodes`, drops what an observation window of walks
   left untouched — see *What walk-driven PRUNE means*.
6. Re-running invalidated work never revives a `decision` — it appends a new one
   that `supersedes` it, so rule 3 holds through re-execution. An `entity` *is*
   revived when the decision that produces it runs again, because the artefact
   was rebuilt; one the rerun stops producing stays invalidated.
7. An `assumption` is only ever rejected by `evidence` that `contradicts` it. A
   caller may not mark one false directly, because "why did this fall over" has
   to have an answer inside the graph.
8. A node or edge id names one semantic identity. A derived id that would
   collide with a different node or edge is refused, never silently merged
   into it — see *What a node/edge id collision means*.
9. A `decision` is never content-addressed. Two decisions with the same
   title and rationale are two events, not one; an explicit `id` on
   `decide` is an idempotency key for one call, not a way to name a
   decision by its content — see *What a decision's identity is*.

## Declared, not yet realized

Each of these is declared here or in the type system, is modelled and in some
cases persisted and validated, and has **no production path**. They are recorded
so a reader does not mistake a declaration for a live capability, and so that
deciding their intent is a visible act rather than an accident of first use.
None of them may be relied on until it appears above this line.

| Declared | State | Missing |
|---|---|---|
| `resolves_to` | In the edge table; in `EdgeType`; weighted `0.3` by the walker | No realized lifecycle today: no production writer creates it, and no node-level justification, contradiction, supersession, support, dependency or citation path is legal for `entity` endpoints. |

## Open items

Questions this file does not yet answer. Each is a decision, not a bug; none is
resolved by pointing at current behaviour.

1. **A decision's processing-time field has no name here.** The `Must carry`
   cell for Decision previously read `at_build`, which names nothing on a
   decision node: the node carries `Node.build`, while `at_build` lives on
   `DecisionRecord`, which is not part of the graph snapshot. The false token is
   removed; naming the intended field is open.

Resolved since the list above was first written, kept here as a record rather
than deleted, since a reader following an old reference should land somewhere
that says what happened to it:

- **`Symbol` was unverified.** `ontology.py` now reads the column and refuses
  to load if it disagrees with `Type`; see *What is mechanically enforced*.
- **Rule 2's propagation set was restated in code.** The edge table's
  `Invalidation` column is now that machine-readable form, and
  `contextmesh/assumptions.py` reads it through `ontology.py` instead of
  restating it.
- **`AssumptionLedger.justifies` propagated further than its wording.** Its
  docstring marked an *edge* as standing on an assumption, but rejection
  reached through the edge to invalidate `edge.dst` too — a second,
  undeclared invalidation path alongside rule 2. Binding an edge no longer
  seeds its endpoints into the blast radius; see *What edge-level assumption
  binding means*. A node that needs to fall with the assumption still has to
  say so with `depends_on`, the same as every other node-to-node case.
- **`Must carry` had no enforcement path.** `ContextGraph.add_node` now
  enforces it; see *What `Must carry` means*.
- **Walk-driven `PRUNE` had no caller.** `Pipeline.prune_unwalked_nodes` now
  takes the `Walker` that produced the telemetry and runs for real, gated on
  an observation window rather than firing on the first walk that happened
  not to reach a node; see rule 5 and *What walk-driven PRUNE means*.

Out of scope for this file today: how entity identity should be asserted,
recorded or retracted. `resolves_to` exists as a declaration only, and the
design that would give it a lifecycle is not settled.
