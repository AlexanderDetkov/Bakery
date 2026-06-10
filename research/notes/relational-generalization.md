---
title: A proper formulation of the knowledge-propagation instrument — knowledge worlds as least-fixpoint closures
kind: note            # framing / theory note (not a finding; carries no run evidence)
status: framing
created: 2026-06-08
links: [[q-graph-structure-diamonds-multipremise]], [[baking-is-associative-prompting-is-directional]], [[yes-saturation-is-fact-general-converse-amplification-is-not]], [[baked-propagation-tracks-trained-depth-no-compositional-bonus]]
---

## Purpose

The current instrument is a Hilbert-style proof system over a directed acyclic graph: one binary
"is-a" relation, one inference rule (transitivity), truth = transitive closure, "hop"/"proof depth" =
shortest directed path. This note gives the **proper, neutral abstraction** of which that instrument is
one instance, so that generalizations (to general relations and equivalence classes) are principled rather
than ad hoc. It is a *framework*, deliberately method-agnostic: prompting, baking, and SFT are all just
methods we score against the framework. The specific "baking installs an associative shadow" conjecture
appears below only as **one application** (§5a), not as the organizing spine.

No code changes accompany this note. It names the existing code objects each abstraction generalizes.

---

## §0. The object — a knowledge world and its fixpoint semantics

A **knowledge world** is a tuple `W = (A, Σ, E, P)`:

- `A` — a finite set of entities (atoms / individuals).
- `Σ` — a relational **signature**: relation symbols with arities. *Current instrument:* a single binary
  symbol, `isa`.
- `E` — the **axioms**: ground atoms over `Σ, A` stated in the prompt. *Current:* the generating edges
  `isa(X, Y)` ("Every X is a Y").
- `P` — a finite set of **Horn rules** `h :- b₁, …, b_k` (`k ≥ 1`). *Current:* the single transitivity
  rule `isa(X, Z) :- isa(X, Y), isa(Y, Z)`.

`P` induces a monotone operator on sets of ground atoms,

> `T_P(S) = E ∪ { ground heads firable from S under some rule of P }`.

Because `T_P` is monotone, it has a **least fixpoint** `lfp(T_P) = ⋃ₙ T_P↑n` (with `T_P↑0 = ∅`,
`T_P↑(n+1) = T_P(T_P↑n)`). The **theorems** of `W` are exactly `lfp(T_P)`; everything else is false
(closed-world). Every theorem has a finite **derivation tree** whose leaves are axioms and whose internal
nodes are rule firings. This is the standard Datalog / Hilbert-system semantics, and it is what
distinguishes a "knowledge world" from a mere graph: the truth set is *generated* by a closure, not listed.

**This is already what `bakery/logic/` implements.** `World` carries `(A, E)` with `Σ = {isa}`;
`ProofEngine` computes the least fixpoint of the transitivity rule (as shortest-path BFS);
`brute_force_reachability` is an independent oracle (Floyd–Warshall) cross-checking that fixpoint on every
pair. Reading the codebase through this lens: the proof engine is a *fixpoint solver for one specific `P`*,
and a generalization is a fixpoint solver for a richer `P`.

---

## §1. Difficulty index — fixpoint stage = derivation height (generalizes "depth"/"hop")

Define the **stage** of a theorem `a` as the least `n` with `a ∈ T_P↑n`. Equivalently, the minimal
**height of a derivation tree** for `a`. This is the proper generalization of "hop count":

- **Single-premise rules** (`k = 1`, current): a derivation is a *chain*, so stage = path length =
  today's `proof_depth`. The single-inheritance forests in `data/worlds/lw_*.json` are exactly this case
  (every concept has ≤ 1 parent; positive entailments are single paths).
- **Conjunctive bodies** (`k ≥ 2`): a derivation is a *tree*, so stage = `max_i stage(bᵢ) + 1` =
  proof-tree height. A second, orthogonal axis appears — **width / arity** = how many premises must be held
  *simultaneously* to fire a head. Chain-depth and conjunctive-width are independent difficulty knobs.

So "n-hop" becomes "stage-n," and AND-composition is a distinct width dimension layered on top of stage.

---

## §2. Negative index — minimal truth-flipping perturbations, one family per ABSENT property

Under the closed-world reading, "false" = "outside `lfp(T_P)`." The *informative* negatives — the ones a
probe should ask — are atoms sitting on the **boundary** of the fixpoint: minimal perturbations of a
theorem that fall just outside the closure. Crucially, the boundary's structure is dictated by the
algebraic properties the closure does **not** enforce. One negative family per absent property:

| Property the closure lacks | Hard-negative family | Current `neg_type` |
|---|---|---|
| not symmetric | **converse** — swap a theorem's arguments | `converse` |
| closure is exact (no spurious edges) | **off-by-one-derivation** — stage `d−1` reachable, final rule doesn't fire | `missing_edge` |
| domain is partitioned | **cross-partition** — entities in different components | `cross` |
| multiple relation symbols | **wrong-relation** — right pair, wrong symbol | *(new under multi-`Σ`)* |

So the existing `neg_type ∈ {converse, cross, missing_edge}` is *not* an ad-hoc list — it is precisely the
catalog of ways a learner could **wrongly extend** a strict order's closure, one probe family per missing
property. `relations.candidate_pools` is the constructor of these families for the current `t`-slice; the
general constructor emits the families implied by `W`'s actual properties.

---

## §3. The relation zoo — the closure-operator lattice (the core, method-agnostic catalog)

Restrict for a moment to a single binary relation `R` (the "relation-algebra slice"). The named relation
types are exactly the standard **closure operators** applied to the axioms `E`:

| Closure of `E` | Relation type | Truth set | Negatives that survive |
|---|---|---|---|
| `t(E)` (transitive) — acyclic | **strict partial order** *(current)* | directed reachability | converse, cross, missing-edge |
| `rt(E)` (reflexive-transitive) | **preorder** | reachability incl. identity | converse (across blobs), cross, missing-edge |
| `rst(E)` (refl.-symm.-trans.) | **equivalence relation** | the connected components `A/∼` | cross only (converse is now a *positive*) |
| `s(E)` (symmetric) | one-hop undirected adjacency | neighbours only | non-neighbours (no transitivity) |

Each operator is a *different rule set* `P` (e.g. adding the symmetry rule `R(Y,X) :- R(X,Y)` turns a
preorder into an equivalence relation) and hence a *different world type* — but all are instances of §0.
Note `World.components` is already the quotient `A/∼` of the `rst`-world, so the equivalence-relation
instrument is a **truth-function relabel** of the existing machinery ("same class?" instead of
"reachable?"), not a rebuild.

**Generalizing the signature `Σ`** (beyond one relation):

- **Multiple typed relations + bridge rules** — e.g. `isa` + `has-prop` with
  `has-prop(X,P) :- isa(X,Y), has-prop(Y,P)` (property inheritance). Tests chaining *across relation
  types*, not just along one relation. Introduces the `wrong-relation` negative family.
- **Higher-arity Horn** — relational atoms `R(a,b)` with general Horn rules (kinship:
  `grandparent(A,C) :- parent(A,B), parent(B,C)`). The fully general "knowledge graph reasoning" setting;
  the current instrument is its simplest fragment.

This zoo is the framework's menu of worlds — defined entirely independently of which method is studied on
them. **Two of these instances are already enqueued** as [[q-graph-structure-diamonds-multipremise]]:
*diamonds* (convergent DAGs — still the `t`-slice, but stage = shortest of several paths) and
*multi-premise* (the `k ≥ 2` width axis of §1). That open question lives entirely within
`isa`/transitivity; this note's contribution beyond it is the **algebraic axis** — varying *which closure
operator* generates truth (symmetry → equivalence/preorder), which diamonds and multi-premise do not touch.

---

## §4. Readout — what any method's behavior is scored against

For each (non-)theorem, indexed by **stage `d`** (§1) and **negative-family `f`** (§2), score a method's
belief with the existing bias-immune readout (d′ / AUROC). The framework is method-agnostic: the identical
readout applies to the **prior** (base, no prompt), **prompting** (base + axioms in context), **baking**
(adapter, no prompt), **SFT**, or any future method.

The headline object is the **stage × family agreement matrix** between a method's induced truth function
and a *chosen target closure*, with prompting as the practical upper bound. Today's two metric indices —
`proof_depth` and `neg_type` — are exactly the `d` and `f` axes of this matrix, specialized to the
`t`-slice.

---

## §5. Applications (hypotheses the framework lets you state precisely)

The framework does not privilege any method. These are hypotheses it can phrase sharply; (a) is where our
current evidence sits, but it is one entry, not the spine.

- **(a) Inductive-bias / closure-approximation.** *For a method M, which fixed closure does M's induced
  truth function approximate, independent of `P`?* An attractive early conjecture was **baking ≈ `rst(E)`**
  (the equivalence closure — "learns comparability, not order"), motivated by the pre-Hilbert belief-metric
  results ([[baking-is-associative-prompting-is-directional]],
  [[yes-saturation-is-fact-general-converse-amplification-is-not]]) showing baking over-affirming the
  converse. **A bias-immune per-negative-family re-read on the strict-order world REFUTES that specific
  form** ([[converse-collapse-does-not-survive-bias-immune-instrument]]): baking does *not* over-affirm the
  converse (baked converse-AUROC ≥ prompting; FA far from saturation — at d1 prompting affirms it *more*).
  What survives is **over-connection across the partition** — baking spuriously affirms genuinely-unrelated
  `cross`-component pairs (AUROC 0.66–0.77 vs prompting 0.93–0.95). So the measured deviation is *spurious
  extra connectivity*, not *symmetrization of direction* — there is no clean named closure for "true
  relation + spurious cross edges," so the strong fixed-closure conjecture is downgraded to a falsifiable,
  per-family question. Predictions read off §3, updated:
    - strict-order world → divergence from prompting concentrates on the **`cross`** family (over-connection),
      NOT the converse (observed, bias-immune); the earlier converse-collapse was chain-specific / a
      belief-metric artifact;
    - equivalence-relation world → was the **sharp test**, now ✅ **RUN & CONFIRMED**
      ([[equivalence-world-confirms-cross-over-connection-mechanism]]): `cross` is the only false family;
      baking aced positives (incl. would-be-converse: d1 AUROC 1.00) and concentrated its whole error on cross
      (baked d2 cross-FA 0.93 vs prompted 0.06) — cleanly separating "over-connection" from "symmetrization".
      Refines the formal claim: baking **inflates the closure** (over-permissive/low-precision), it does not
      symmetrise direction;
    - cyclic preorder → predicted **within-world dissociation**: correct *inside* strongly-connected blobs,
      over-connecting *across* them (still untested).
- **(b) Reach-vs-stage.** Characterize prompting's (or any method's) accuracy as a function of stage `d` on
  a fixed world — the propagation-decay curve — now well-defined for trees and widths, not just chains.
- **(c) Method contrasts under one world.** prompting vs baking vs SFT on the same `W` and readout, to
  isolate objective effects (cf. [[q-sft-vs-bake-reversal-curse]]).
- **(d) Property installation.** Which closure properties (symmetry, transitivity, AND-gating, cross-type
  chaining) a given method installs, read off the negative-family rows of the §4 matrix.

---

## §6. Categorical gloss (one remark, intuition only)

Let `G` be the free category on the axiom graph: objects = atoms, morphisms = derivations, composition =
transitivity, so `Hom(x, z) ≠ ∅ ⇔ x ⊑ z`. Its **connectedness groupoid** `|G|` (freely adjoin an inverse
to every morphism) has `π₀(|G|) =` the connected components `= World.components`. Conjunctive rules make
`G` a free **multicategory** (multi-input morphisms ⇒ proof trees); multiple relation symbols make it
**colored**. In this language a method that "forgets orientation" is one approximating `Hom_{|G|}`
(connectedness) rather than `Hom_G` (the oriented relation) — the converse curse is the
groupoidification unit `η : G → |G|` failing to be faithful. Offered as intuition; the
fixpoint / relation-algebra view above (§0–§5) is the working formalism.

---

## §7. Subsumption of the current instrument (explicit)

The current instrument is the **`t`-slice with `Σ = {isa}`, `k = 1`, acyclic** — a strict partial order.
Concretely:

- `World` / `ProofEngine` = a knowledge world `W` whose `P` is the single transitivity rule and whose
  fixpoint solver is shortest-path BFS; `brute_force_reachability` = the independent fixpoint oracle.
- `proof_depth` (= shortest directed path) = the **stage** of §1, specialized to chains.
- `neg_type ∈ {converse, cross, missing_edge}` = the **absent-property families** of §2 for a strict order.
- `World.components` = the quotient `A/∼` of the `rst`-world = `π₀` of the connectedness groupoid (§6).
- `relations.candidate_pools` = the §2 negative constructor for this slice.

Nothing in the current gate, metric, or proof-engine self-test is contradicted — they are the special-case
realization of the framework. Any generalization (equivalence/preorder worlds; diamonds; conjunctive rules;
multi-relation signatures) is "instantiate a richer `(Σ, P)`, extend the fixpoint solver + its independent
oracle, and let §2 emit the property-matched negatives" — with the validity gate and d′/AUROC readout
unchanged.
