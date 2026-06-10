---
title: The reversal curse, placed inside the fixpoint framework — and why baking is predicted to differ from SFT
kind: note            # framing / theory note (no run evidence of its own)
status: framing
created: 2026-06-10
links: [[relational-generalization]], [[propagation-bounded-by-trajectory-coverage]], [[trajectory-type-is-a-binary-coverage-gate]], [[contrastive-trajectories-do-not-fix-the-converse]], [[CORRECTED-picture-robust-metric]], [[converse-collapse-does-not-survive-bias-immune-instrument]], [[equivalence-world-confirms-cross-over-connection-mechanism]], [[grokking-null-no-late-propagation-transition]], [[q-sft-vs-bake-reversal-curse]], [[q-grokking-converse-via-longer-training]]
---

## Purpose
The sister project `~/Invertibility` (branch `research/scaling`) is a controlled, full-SFT study of the
**reversal curse** (Berglund et al. 2023: a model trained on "A is B" fails "B is A"). This note forms a
*rigorous* connection between it and our baking-vs-prompting findings: it (i) states their result precisely,
(ii) places the reversal curse inside our least-fixpoint formalism ([[relational-generalization]]), (iii)
draws the sharp line between what our findings already establish and what they do **not** (the converse vs.
the true inverse — a duality that is easy to conflate), (iv) derives the curse as a corollary of our
coverage-bound, and (v) states the one place baking is predicted to behave **differently** from the toy SFT
model, with falsifiable predictions and a specified (un-run) experiment. No code accompanies this note.

---

## §1. The reversal curse, precisely (what Invertibility established)
Setup (`~/Invertibility`, full **SFT** = next-token cross-entropy on a ~90k-param 4-layer Llama; no LoRA, no
prompting, no distillation): states `0..N-1`, a random bijection `f`, two action tokens — forward `R`
(`s ↦ f(s)`) and inverse `L` (`s ↦ f⁻¹(s)`). For each held-out state exactly **one** direction is withheld
from training (the other stays); the model must produce the withheld direction. `T` = path length (how many
actions are chained in a training sequence); `k` = number of independent relations. Metric: held-out inverse
accuracy `test_map` (chance ≈ `1/N`); "generalizes" ≥ 0.10, "cursed" ≤ 0.03; plateau = mean of last-5 evals.
Forward train accuracy is ≈ 1.0 throughout, so any failure is *purely* on the withheld direction.

Four quantitative laws (their `research/findings/`):
1. **Isolated forward edges never inverts; compositional paths do.** `T=1` best ≤ **0.025** across
   `N∈{500,1k,2k}`, 5 seeds, 32k epochs, *and* at matched supervised-token budget; `T≥2` groks (plateau
   0.24–0.45 at N=1000). (`path-depth-matched-{epoch,token}.md`, `t1-strong-negative.md`.) ~10× separation.
2. **The curse scales with domain size `N`.** At `T=2`, plateau **0.01 (N=250, cursed) → 0.94 (N=4000,
   ~solved)**; the small-`N` curse is intrinsic, not a held-out-fraction artifact. (`n-scaling{,-heldfraction}.md`.)
3. **Grokking is late, non-monotonic, weight-decay-driven.** Inverse generalization onsets ~ep1000–2000
   (well after train fit), peaks then decays to a plateau (peak > plateau), under `weight_decay=0.1`; deeper
   paths grok earlier and more stably; optimal `lr` decreases with depth. (`depth-scaling.md`,
   `peak-vs-plateau-stability.md`, `lr-depth-interaction.md`.)
4. **Curse-break depth scales with relational complexity:** `T ≈ k+1` (k independent relations), and *mixing*
   relations is super-additive. (`grid-threshold-scales-with-relations.md`.)
They offer no closure/fixpoint formalism (their account is empirical: "paths embed the compositional
structure that reveals the inverse").

---

## §2. Formal placement — the reversal curse is a missing closure rule
In the [[relational-generalization]] formalism a world is `W=(A,Σ,E,P)` with truth `= lfp(T_P)`. The reversal
curse needs a world whose signature carries a **designated inverse**: a relation `R` *and* its inverse symbol
`R⁻¹`, with truth generated (in part) by the **inverse rule**

> `ρ_inv :  R⁻¹(Y, X)  :-  R(X, Y)`.

The axioms `E` state only forward atoms `R(a,b)`; the inverse atoms `R⁻¹(b,a)` are **true** (entailed by
`ρ_inv`) but *never stated*. The reversal curse is then exactly:

> **a learner reproduces the forward atoms `R` but its learned rule set `P̂` omits `ρ_inv`** — it fails the
> `R⁻¹` queries despite `R⁻¹` being true in `W`.

This is the multi-`Σ` "wrong-relation" row of [[relational-generalization]] §2 made into a *positive*: under a
designated inverse, the reversed pair is no longer a false converse but a **true** atom of a second relation.
Invertibility's `T≥2` result is, in this language, that `ρ_inv` (a rule *schema* — the systematic inverse map,
not a per-pair fact) is **not inducible from isolated forward atoms** but **is** inducible from derivations
that exhibit the map composing across states (paths). Stage/derivation-height ([[relational-generalization]]
§1) is their `T`. Their "relation-count law `T≈k+1`" is: each extra relation adds a rule whose induction
needs one more level of compositional context.

---

## §3. The duality — what our findings establish, and what they do NOT (the key rigor point)
Our directed `lw_*` worlds have **no inverse rule** (closed-world, `Σ={isa}`, only transitivity). So the
reversed pair `(z,x)` of a true `isa(x,z)` is **FALSE**, and our `converse` negative family asks the learner
to **reject** it. That is the **dual** of the reversal curse:

| | reversed pair's truth | correct behaviour | failure mode measured | our result |
|---|---|---|---|---|
| our `converse` family (no inverse rule) | **FALSE** | reject | **over**-generalisation (wrongly affirm) | baking rejects it — directional |
| reversal curse (inverse rule present) | **TRUE** | affirm | **under**-generalisation (fail a true inverse) | **UNTESTED on our instrument** |

So our high-confidence "baking is **not** direction-blind / does not collapse on the converse"
([[converse-collapse-does-not-survive-bias-immune-instrument]]) measures the *over*-generalisation axis and
says **nothing directly** about the reversal curse, which lives on the *under*-generalisation axis. We have
never put a true-but-unstated inverse in front of a baked model.

**The unification (why this is one phenomenon, not two).** A learner that copies the *trained-direction
single-pass conditional and nothing about the reverse* will simultaneously (a) **reject the false converse**
(it never installed any `z→x` mapping) and (b) **fail the true inverse** (same reason). (a) is the *benefit*
of directionality; (b) is its *cost*. "Baking is directional" and "baking has the reversal curse" are thus the
**same prediction seen from two sides**. Our converse result is therefore *consistent with* baking exhibiting
the reversal curse — it does not refute it; it is the other face of the same coin.

**A trap to avoid.** The equivalence world's perfect within-class score (d1 AUROC 1.00, incl. the
would-be-converse, [[equivalence-world-confirms-cross-over-connection-mechanism]]) is **not** reversal-curse
escape: there the *phrasing is symmetric* ("are X and Y the same kind?") and the closure is `rst` (symmetry is
given), so there is no untrained inverse to generalise. The reversal curse requires an **asymmetric** relation
whose inverse is true but surfaced under a *different* form than the trained forward form.

---

## §4. The bridge — the reversal curse is a corollary of our coverage-bound
Our most robust, instrument-independent result is the **coverage-bound**: baking installs a conditional **iff
the trajectory distribution exercises it**, and `eval_kl` is decoupled from what propagates
([[propagation-bounded-by-trajectory-coverage]], [[trajectory-type-is-a-binary-coverage-gate]]). Apply it to
`ρ_inv`: the inverse conditional `P(· | "R⁻¹ of b is ?")` is in the baked model's supervised span **iff the
trajectories exhibit the inverse direction**. Forward-only trajectories never do ⇒ `ρ_inv` is never installed
⇒ reversal curse — **as a corollary, not a new phenomenon.** And it predicts the *same fix* Invertibility
found: trajectories that exhibit the inverse rule firing (compositional paths, or explicit inverse statements)
put `ρ_inv` into coverage and lift the curse. In fixpoint terms: **baking learns `lfp` over the
trajectory-covered generators; `ρ_inv ∈ P̂` iff the trajectories exhibit it.** Invertibility's `T≥2` paths are
exactly "trajectories that exhibit `ρ_inv`"; their `T=1` is "forward-only coverage."

---

## §5. Where baking is predicted to DIFFER from SFT (the novel, testable claim)
Invertibility is **full SFT** — there is no reasoner in the loop, so the only way `ρ_inv` enters the data is
the path structure. **Baking is KL-distillation of a *prompted teacher*** that is a pretrained 8B model and can
**infer the inverse in-context** (given "A is B's parent" in the prompt, it can answer "B's child is A" by
reasoning, even with the inverse never a training token). Baking's trajectories are *sampled from that
teacher*. Therefore:

> **Baking's reversal curse is governed by the teacher's *generative* coverage of the inverse, not by weight
> geometry.** If the prompted teacher's sampled generations exercise `ρ_inv` (it reasons the inverse in its
> CoT), baking can **escape** the curse *even from forward facts* — where toy SFT, lacking a reasoner, cannot.
> If the teacher only ever states forward facts, baking inherits the curse exactly as SFT does.

This is consistent with — and sharpens — two existing findings: [[contrastive-trajectories-do-not-fix-the-converse]]
(baking inherits the *teacher's generative bias*, not an abstract symmetry) and [[CORRECTED-picture-robust-metric]]
(the prompted 8B teacher is directional / can reason). It reframes the reversal curse for distillation: it is a
property of *the teacher's trajectory distribution*, which is exactly the lever our coverage-bound identifies.

---

## §6. Falsifiable predictions (each with the teacher-coverage wrinkle)
1. **Path-depth (their law 1):** SFT on forward-only trajectories shows the curse (inverse ≈ chance); SFT on
   path trajectories (`T≥2`) lifts it. **Baking** lifts it at lower apparent `T` *to the extent the teacher's
   CoT supplies inverse reasoning* — possibly even from `T=1` forward facts. A divergence here = the
   distillation framing matters; SFT≈bake = "it was the reversal curse all along."
2. **N-scaling (law 2):** our worlds are small (`lw_*`/`eq_*` ≈ 42 atoms) — Invertibility's law 2 predicts this
   is the **cursed regime**. This is both a *caveat* (don't expect inverse generalisation at our N even if the
   mechanism is benign) and a *prediction* (scale `N` up and the inverse should improve, for SFT and baking).
3. **Grokking (law 3):** does the inverse grok late (ep~1–2k) under weight decay in the *distillation* setting?
   Our [[grokking-null-no-late-propagation-transition]] is a strong null **but for DEPTH propagation, not the
   inverse** — it does **not** settle the inverse question. A direct inverse-grokking arm (with wd, long
   training) is required; predicted: SFT may grok the inverse late (as the toy model does), baking's onset
   depends on teacher coverage.
4. **Relation-count (law 4):** in a multi-relation baked world, curse-break depth should scale `≈k+1` for the
   SFT arm; baking's effective threshold should be *lower* by whatever the teacher's reasoning supplies.

---

## §7. The experiment that would test this (specified, NOT run — see plan)
A `relation_mode="inverse"` world: a relation `R` with a **designated true inverse** `R⁻¹` (e.g. an `isa`
edge plus an `has-instance` inverse query phrased under a *different* surface form), trajectories state **only
forward `R`**, and the **inverse `R⁻¹` is held out as a TRUE positive** (not a false converse). The gate
already enforces forward-train / inverse-eval disjointness (`bakery/trajectories/base.py:177-189`), and the
`sft` objective already exists as the toy-model bridge (`bakery/objectives/sft.py`, `--train.objective sft`,
identical trajectories to `bake`). Three arms on identical data:

- **prompting** (base+u, read from the metric's prompted state) — the in-context reasoner; upper bound.
- **SFT** (`--train.objective sft`) — the toy-model analogue at 8B/LoRA scale; predicted to show the curse.
- **bake** (KL) — predicted to track the *teacher's generative coverage* of the inverse.

Readouts: forward_acc vs **inverse_acc** (reuse the d′/AUROC per-form machinery, adding an `inverse` form),
plus a **grokking arm** (wd, long training: does the inverse grok in the distillation setting?) and a small
**N sweep** (law 2). This resolves [[q-sft-vs-bake-reversal-curse]] and the inverse half of
[[q-grokking-converse-via-longer-training]]. Building it is a known quantity (the `relation_mode` pattern from
the equivalence world); deferred to a later cycle per the current scope.

---

## TL;DR
The reversal curse = a learner reproducing forward atoms `R` but failing to install the inverse rule
`ρ_inv: R⁻¹(Y,X):-R(X,Y)` despite `R⁻¹` being true. In our fixpoint framework it is a *missing closure rule*;
in baking terms it is a *corollary of the coverage-bound* (`ρ_inv` is installed iff trajectories exercise it).
Our "baking is directional / no converse-collapse" result is the **dual** (rejecting a *false* converse) and
is **consistent with**, not contrary to, baking having the curse — both are "copy the trained-direction
conditional, nothing about the reverse." The one genuinely new prediction: because baking distills a
**reasoning teacher**, its reversal curse is set by the teacher's *generative coverage* of the inverse, so
baking can escape where toy SFT cannot — a clean, falsifiable SFT-vs-bake test that we have specified but not
yet run, and which our small-`N` worlds are (per Invertibility's law 2) in the *cursed* regime for.
