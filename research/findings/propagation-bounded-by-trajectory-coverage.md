---
title: Baking only injects a fact as far as the trajectories exercise it; eval_kl does not certify the fact was learned
outcome: positive          # positive (headline C3 + the on-topic/off-topic gate); C1/C4 suggestive only
confidence: medium         # high for C3 + gate; low for the per-hop "propagation distance" and size claims
created: 2026-06-06
question: [[q-propagation-prompt-vs-bake]]
metric: propagation        # no-CoT forced-choice belief shift vs prior; eval_kl secondary
run_ids: [prop-tsunami-1b-r16-mixed, prop-tsunami-8b-r16-mixed, prop-tsunami-1b-restate-m12, prop-tsunami-1b-consequence-m12, prop-tsunami-1b-neutral-m12]
---

## Insight
Baking distills a prompted fact into LoRA weights only *partially* and only for the consequences the
trajectory distribution actually exercises; `eval_kl` measures trajectory-distribution match and is
**decoupled** from how far the fact propagates to held-out probes — a low `eval_kl` can mean "the
trajectories never carried the fact," not "the fact baked in well."

## Method (the instrument built this cycle)
Inject one fact `u` (a magnitude-9 tsunami hitting Japan today) by PROMPTING (`base()`+u) vs BAKING
(LoRA adapter, `baked()`+empty), and read **no-CoT** belief on a held-out forced-choice probe bank:
`belief(q) = logP(pos|q) − logP(neg|q)` in ONE forward pass (cannot be reached by reasoning aloud),
for 16 probes across hops 0–3 (4/hop, Yes/No polarity balanced 2:2). Three states are computed on the
SAME checkpoint (full-vocab, paired): prior `base()`+empty, prompted `base()`+u, baked `baked()`+empty;
propagation = belief shift vs prior. `fidelity = baked − prompted` (~0 ⇒ baking reproduces prompting).
Trajectory TYPE is a knob via a categorized context bank (restate / consequence / neutral / mixed).
Figures: `results/bake_fact/_fig_prompt_vs_bake.png`, `results/bake_fact/_fig_by_type.png`.
Measurement code independently re-verified (an adversarial agent re-loaded the 1B+adapter and recomputed
beliefs; all table cells reproduce metrics.json to 2 dp; the metric is full-vocab, sign-correct, uses
`ASSISTANT_SHIFT` logits[p−1]→token p).

## Evidence (final epoch; shift vs prior, in nats of log-odds)
**C3 — eval_kl ⟂ propagation (CONFIRMED, high):** at matched 48-traj count on 1B,
- neutral (off-topic): eval_kl **0.0124 (lowest)** → baked_shift h0..h3 = +0.02/−0.14/+0.61/−0.72 (≈0).
- restate (on-topic): eval_kl **0.1395 (highest)** → reaches h2 (+1.64).
- consequence (on-topic): eval_kl 0.0546 → reaches h2 (+1.64).
The KL rank-orders **opposite** to propagation: lowest KL = weakest fact injection.

**On-topic/off-topic GATE (robust):** the *most-converged* run (neutral, eval_kl 0.012, train_kl 0.003)
injects ≈0, so its null is genuine (not under-training); on-topic runs inject a clear positive shift
through ~h2. Baking can only distill consequences the trajectories express.

**C1 — prompted−baked gap (SUGGESTIVE, 8B-only):** 8B baked_fidelity h0..h3 = +0.73/−1.56/−2.12/−2.95
(monotone-decreasing); baked collapses at h3 (baked_shift +0.42 vs prompted +3.38). 8B baking matches/
overshoots prompting at h0–h2, then opens a gap with hop distance.

**C4 — model scale (DESCRIPTIVE, n=2):** 8B prompted-normalized retention (baked_shift/prompted_shift)
≈0.71 at h0–h2 vs 1B ≈0.42; 1B baked lags prompted even at h0 (fidelity −2.52).

## Counter-arguments / threats to validity
- **C1 reverses on 1B.** On 1B the gap SHRINKS toward h3 (fidelity −2.52/−1.92/−1.66/−0.25) and baked_shift
  PEAKS at h2 (+1.75). The monotone-growing gap is an **8B-only** single-run pattern, not a propagation law.
- **h3 "collapse" is a low-headroom artifact, not baking-specific.** The PROMPTED teacher itself collapses
  at h3 (8B +3.38 vs ~+9 at h1/h2) and the h3 prior is already saturated (8B prior_h3 +6.31). Little headroom
  for any intervention, so "baking fails to reach far hops" is partly ceiling/floor noise.
- **C3/type comparisons confounded.** The type sweep used 48 traj vs the mixed run's 120 (count-confounded);
  restate is the least-converged run (eval_kl 0.140, still descending) with the fewest supervised tokens, so
  the finer restate-vs-consequence ordering is unidentifiable (plausibly under-training, not type).
- **C4 is not causal.** The 1B/8B runs differ on FOUR uncontrolled axes — LoRA trainable params (3.4M vs 13.6M),
  max_new_tokens (160 vs 128), supervised tokens, and tokenizer/family — and raw log-odds are cross-model
  incommensurable. Two lenses rated C4 refuted as a causal claim.
- **Statistical power.** n=1 fact, 1 seed, 4 probes/hop, NO CIs; per-probe within-hop spread (e.g. baked h0
  −0.56..+2.81) is comparable to several between-hop/between-run differences — sub-~0.5-nat distinctions are
  noise. The neutral null cannot be *proven* from the (now-fixed) per-hop-mean-only logs.

## Implications
`eval_kl` alone is an **insufficient** success criterion for knowledge baking — pair it with propagation
probes and/or an on-topic-coverage check on the trajectory bank. The central, well-supported claim for the
agenda: **baking ≠ prompting because baking is bounded by the trajectory distribution**; prompting carries
`u` into every forward pass for free, baking carries only what the trajectories sampled. This makes
"trajectory type/coverage" the primary lever for knowledge baking — directly answering the user's question.
Instrumentation upgraded this cycle (kept tests green): the `propagation` metric now logs **per-probe**
beliefs (enables bootstrap CIs + rules out polarity cancellation), and the manifest now records the run seed.

## Next steps
See enqueued open-questions: [[q-propagation-trajectory-type]] (count- & convergence-matched 4-way sweep),
[[q-propagation-trajectory-size]] (size scaling), [[q-propagation-hardening]] (≥2 facts, ≥3 seeds, ≥12
probes/hop, bootstrap CIs; demote C1 to a law only if CIs exclude 0), [[q-propagation-model-scale]]
(controlled one-family size sweep, matched LoRA params, normalized retention), [[q-propagation-cot-confound]].
