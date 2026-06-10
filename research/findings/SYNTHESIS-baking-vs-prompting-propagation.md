---
title: SYNTHESIS (v2, aggregate) — how prompt baking differs from prompting in knowledge propagation
outcome: positive
confidence: high            # proof-system core is 12-graph + 4-seed replicated; belief-metric era kept as corrected history
created: 2026-06-06
updated: 2026-06-09
question: [[q-propagation-prompt-vs-bake]]
metric: dprime
run_ids: [graph-lw_alpha-n1, graph-lw_alpha-n2, qa-bake-n1-s10, qa-bake-n2-s10, prop-veld-8b-mixed, prop-tellus-8b-mixed, prop-tsunami-1b-r16-mixed]
# headline run sets: 12-graph sweep graph-lw_{alpha..mu}-n{1,2} (24 runs) + 4-seed qa-bake-n{1,2}-s{10..13} (8 runs)
# + belief-metric era prop-{veld,tellus,tsunami,size,type2}-* (corrected). Numbers: analysis/neg_family_auroc.py.
---

## The question
Inject a new fact / relation either by PROMPTING (put it in context) or by BAKING it into LoRA weights
(distil the prompted model's behaviour into an adapter), then ask questions `n` reasoning hops away. How far
does the update propagate, and how does that differ between the two? This note aggregates everything we have:
two measurement eras (a fictional-chain belief readout, then a controlled proof-system instrument), 32+
headline 8B runs, 12 independent graphs, 4 seeds, plus the trajectory/recipe/regularization controls.

## TL;DR — the current aggregate picture
**Baking reproduces the prompted teacher's knowledge but as a noisier, distance-limited copy, and its one
graph-general failure mode is over-connecting UNRELATED concepts — not getting logical direction wrong.**
Concretely, against the bias-immune proof-system instrument, aggregated over 12 graphs:
- **At the source (depth 1)** baking ≈ prompting but consistently a notch lower — a fidelity gap, not a wall
  (baked AUROC 0.83→0.88 for n=1→n=2 vs prompted 0.92).
- **One hop out (depth 2)** baking propagates about as well as prompting overall, and the n=2 curriculum
  *exceeds* it (0.75 vs 0.69) — **so "baking doesn't propagate" is false on this instrument.**
- **The only robust divergence is the `cross` family**: baking spuriously affirms different-component
  (genuinely unrelated) pairs — baked d2 cross-AUROC **0.58 vs prompted 0.90** (Δ0.32, n=1), with
  P(answer "Yes" | unrelated pair) **0.61 baked vs 0.08 prompted**. The n=2 curriculum shrinks this to 0.77.
- **Baking is NOT direction-blind / does NOT collapse on the converse** — at d2 baked converse-AUROC
  (0.75–0.79) is *above* prompting (0.63). This **refutes** the pre-Hilbert headline (see era reconciliation).
- **Neither propagates past the teacher's reach (depth 3 ≈ chance for both).** Baking inherits the teacher's
  depth ceiling and adds no free compositional hop; no grokking observed through long training.

Figures: `results/_fig_relation_worlds.png` (what the instrument is — implication vs equivalence worlds);
`results/bake_theorem_qa/_fig_propagation_aggregate.png` (the prompted-vs-baked-vs-curriculum bars below).

## Aggregate evidence — AUROC(true@depth vs negative), mean ± popsd over 12 graphs (lw_*, Llama-3.1-8B)
[`analysis/neg_family_auroc.py results/bake_theorem_qa/graph-lw_*-n{1,2}`; prompted is n-independent (one teacher/graph)]

| depth | family | prompted | baked n=1 | baked n=2 | reading |
|---|---|---|---|---|---|
| **d1** | ALL | 0.92±.04 | 0.83±.05 | 0.88±.05 | fidelity gap: baking copies the source a notch noisier |
| d1 | converse | 0.84±.07 | 0.82±.08 | 0.86±.08 | ≈ prompting — **no converse collapse even at the source** |
| d1 | cross | 0.99±.02 | 0.88±.07 | 0.91±.06 | small cross gap already present |
| d1 | missing_edge | 0.95±.06 | 0.79±.09 | 0.86±.08 | |
| **d2** | ALL | 0.69±.07 | 0.63±.09 | **0.75±.07** | n=2 baking *beats* prompting one hop out |
| d2 | **converse** | 0.63±.10 | **0.75±.09** | **0.79±.08** | **baked ABOVE prompted — direction is fine** |
| d2 | **cross** | **0.90±.06** | **0.58±.12** | 0.77±.08 | **THE deficit: over-connecting unrelated pairs; n=2 heals it** |
| d2 | missing_edge | 0.55±.16 | 0.53±.15 | 0.69±.15 | both weak (near a known teacher blind spot) |
| **d3** | ALL | 0.53±.09 | 0.57±.12 | 0.60±.12 | **teacher ceiling: ≈ chance for both states** |
| d3 | cross | 0.66±.09 | 0.52±.21 | 0.55±.16 | cross deficit persists where anything is learned |

**Absolute affirmation — FA = P(answer "Yes" | negative) at d2** (yes-saturation signature, →1.0 = says yes to everything):
prompted converse/cross/missing = 0.07 / 0.08 / 0.21; baked n=1 = 0.40 / **0.61** / 0.62; baked n=2 = 0.17 / 0.21 / 0.31.
→ Baking's error is *affirmation of unrelated pairs*, and the n=2 curriculum more than halves it.

The 4-seed set (`qa-bake-n{1,2}-s{10..13}`, fixed graph, varied split+init) reproduces the same shape with
near-zero seed spread (e.g. d2 cross baked n=1 0.66±.04 vs prompted 0.93) — the effect is graph-general AND seed-stable.

## The seven aggregated claims (with their findings)
1. **Fidelity gap at the source, not a wall** — baked d1 AUROC 0.83–0.88 vs prompted 0.92; more curriculum
   (n=2) and more trajectories narrow but don't close it ([[size-helps-fidelity-not-the-propagation-gap]],
   [[baked-propagation-tracks-trained-depth-no-compositional-bonus]]).
2. **Propagation tracks TRAINED depth; no free hop; hard teacher ceiling** — baking installs
   max(trained-depth, teacher-reach); d2 propagates (n=2 > prompting), d3 ≈ chance for prompted AND baked
   because the *teacher itself* is at chance there. No grokking through long training
   ([[baked-propagation-tracks-trained-depth-no-compositional-bonus]]; grokking-null in flight, finding #19).
3. **The one graph-general divergence is CROSS-COMPONENT OVER-CONNECTION** — baked d2 cross-AUROC 0.58 vs
   prompted 0.90; FA 0.61 vs 0.08. Baking links unrelated concepts; prompting keeps the partition. n=2 heals
   it (0.58→0.77). This is the surviving sense of baking's "associative shadow"
   ([[converse-collapse-does-not-survive-bias-immune-instrument]], confidence **high**, 12 graphs + 4 seeds).
4. **Baking is approximately DIRECTIONAL — the converse-collapse headline is REFUTED on this instrument** —
   baked converse-AUROC ≥ prompting at d1, *above* it at d2; converse-FA far from saturation. At d1 *prompting*
   affirms the converse more (FA 0.50 vs 0.39). ([[converse-collapse-does-not-survive-bias-immune-instrument]]
   corrects [[baking-is-associative-prompting-is-directional]] / [[yes-saturation-is-fact-general-converse-amplification-is-not]]).
5. **eval_kl ⟂ propagation; coverage is the lever** — the lowest-eval_kl (off-topic) bake injects ~nothing;
   eval_kl plateaus while belief is still moving. Trajectory type is a BINARY on/off-topic coverage gate, not
   graded by reasoning-richness (restate ≈ consequence). Survives BOTH eras
   ([[propagation-bounded-by-trajectory-coverage]], [[trajectory-type-is-a-binary-coverage-gate]]).
6. **Cheap regularization preserves behaviour at no propagation cost** — SQuAD anchor trajectories drive
   behaviour_drift 0.023→0.011 monotonically (32→256 anchors), d2/d3 AUROC unchanged
   ([[regularization-buys-behavior-preservation-cheaply]]).
7. **Recipe / measurement discipline (settled)** — lead with AUROC not raw d′ (~8× lower variance,
   [[paired-matched-seed-protocol]]); fast bake = lr 3e-4 constant + teacher-logit cache ≈ 7× faster, quant is
   memory-only ([[faster-training-recipe]], [[teacher-logit-cache]]); sampled-teacher trajectories keep CoT so
   held-out d≥2 is partly recall-of-recited ([[sampled-teacher-trajectories-keep-cot]] — see threats).

## Era reconciliation — why the headline flipped (belief-metric → proof-system)
The pre-Hilbert era (fictional Veld/Tellus/Tsunami chains, no-CoT forced-choice belief logP(pos)−logP(neg))
concluded baking was **direction-blind / yes-saturated / affirms the converse** (Veld baked converse belief
−6.61 vs prompted −1.88; baked fracYes ≈1.00), replicated across 3 seeds. Three corrections walked it back:
- **Tokenization artifact** ([[tokenization-artifact-corrected-prompting-is-directional]]): the metric scored
  " Yes"/" No" (leading space) but Llama-3 chat emits "Yes"/"No" (distinct ids 9642/2822) → it fabricated much
  of the *prompted* converse failure. Corrected against generated answers, **prompting rejects the converse**;
  yes-saturation was relocated to being BAKING-specific.
- **Chain-specificity + capacity** ([[CORRECTED-picture-robust-metric]], [[yes-saturation-is-fact-general-converse-amplification-is-not]]):
  the dramatic converse-amplification was Veld-specific (Tellus baked rejects the converse 0.80 ≈ prompting);
  forward propagation (prior ~0.05 → baked ~0.90) is the robust fact-general positive; the 1B is capacity-gated.
- **Bias-immune controlled instrument** ([[converse-collapse-does-not-survive-bias-immune-instrument]]): on
  lw_alpha + 11 sibling graphs with a d′/AUROC readout *built so yes-saturation forces AUROC→0.5*, the
  converse-collapse **does not reproduce**. The genuine associative error survives but **moves families**:
  from "symmetrizes direction" to "over-connects across the partition" (cross). This also refutes the simple
  symmetric-closure `rst(E)` conjecture (relational-generalization note §5a), which predicted the opposite.

**What survived intact across both eras:** eval_kl ⟂ propagation; the on/off-topic coverage gate; baking is
bounded by what the TEACHER generates; forward propagation is real and fact-general; baking is a noisier copy
that does not exceed the teacher's depth reach. **What was an instrument artifact:** uniform "direction-blind /
yes-saturation" (tokenization bug + yes-bias + Veld quirk). **What was relocated:** the associative signature →
cross-component over-connection. **Residual quantitative fix:** "n=1 baking beats prompting at d2" was a
d′-vs-AUROC mismatch — under AUROC only the n=2 curriculum exceeds prompting at d2.

## Limitations / threats (honest, program-wide)
- **Sampled-teacher CoT leak** ([[sampled-teacher-trajectories-keep-cot]]): 26–57 held-out probes are recited
  in the teacher's sampled CoT, so absolute held-out d≥2 numbers are partly recall-of-recited. The leak is
  ~constant across prior/prompted/baked, so the *relative* prompted-vs-baked claims (incl. the cross gap) hold;
  the clean-d2 reference is the unfinished teacher-forced arm ([[q-teacher-ceiling-vs-objective-limit]]).
- The belief-metric era is single-fact-per-chain and small probe banks (4–6/hop); the proof-system era is the
  hardened evidence base (12 graphs, 4 seeds, 16+16 probes/cell). Lead with the proof-system numbers.
- Behavioural (not mechanistic) readouts throughout; LoRA r/α=16 only; 1B vs 8B is not a controlled scale sweep
  ([[q-propagation-model-scale]]).

## Settled vs open
**Settled:** the instrument + validity gate; eval_kl⟂propagation; the on/off-topic coverage gate;
**baking is a noisier, depth-limited, directionally-faithful copy whose one graph-general error is
cross-component over-connection** (12 graphs, 4 seeds, high confidence); the recipe/measurement discipline.
**Open / in flight:** the **equivalence-relation world** (finding #20, sweep chained) — the sharp test, since it
collapses the negatives to `cross` ONLY and makes the converse a free positive; if over-connection is the
mechanism, baking's entire deficit should concentrate there. The **grokking null** (finding #19, long bakes
running). The **teacher-forced clean-d2** arm (removes the CoT leak). Knowledge-baking sequential composition
(agenda #2) still untouched.
