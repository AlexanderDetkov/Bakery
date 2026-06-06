---
title: More trajectories lower eval_kl but don't close the prompting–baking propagation gap; and eval_kl converges BEFORE belief does
outcome: inconclusive       # headline "propagation scales with size" NOT established; solid sub-results + a methodological finding
confidence: medium          # eval_kl↓ and below-ceiling-at-h0/h1 are solid; the scaling/plateau claim is confounded
created: 2026-06-06
question: [[q-propagation-trajectory-size]]
metric: propagation
run_ids: [prop-size-tpc1-1b, prop-size-tpc4-1b, prop-size-tpc8-1b, prop-size-tpc16-1b]
---

## Insight
Increasing the number of baked trajectories (30→120→240→480 at fixed contexts/type) monotonically lowers
`eval_kl` but does **not** close the prompting–baking belief gap at the near hops, and — critically —
`eval_kl` plateaus while the baked belief is **still rising**: the distillation objective converges before
the downstream knowledge does. The headline "propagation scales with trajectory count then plateaus" is
**not established** (confounded with gradient steps, within n=4 probe noise, and read off an unconverged
propagation curve).

## Method
Held contexts fixed (mixed, 30 ctx) and varied `trajectories_per_context ∈ {1,4,8,16}` → total train
trajectories {30,120,240,480}; 1B, rank 16, 20 epochs, seed 0. Same no-CoT forced-choice probe readout as
[[propagation-bounded-by-trajectory-coverage]]; per-probe beliefs logged → bootstrap 95% CIs over the 4
probes/hop. Two independent adversarial skeptics re-derived the numbers and hunted confounds.

## Evidence
- **eval_kl ↓ monotonically with data (CONFIRMED, clean):** 0.1074 → 0.0687 → 0.0489 → 0.0417. Between-run
  gaps (~0.04/0.02/0.007) dwarf last-5-epoch noise (~0.002). More data ⇒ better distillation (expected).
- **Baking stays BELOW the prompting ceiling at h0/h1 at every size (CONFIRMED via paired test):** the
  paired prompt−baked gap excludes 0 at h0 (+1.78 [+1.19,+2.38]) and h1 (+1.27 [+0.39,+1.81]); baked reaches
  only ~50–70% of the prompted shift. More data did not close this gap.
- **eval_kl converges before propagation (METHODOLOGICAL, important):** at tpc8, eval_kl was flat over the
  last 5 epochs while baked_shift was still climbing (h2 2.23→2.48; h0 +0.16 over epochs 16→20). The
  distillation objective saturates earlier than the held-out belief.
- **Size→propagation "rise then plateau": NOT established.** Point estimates rise (mean h0–h2: tpc1 +1.21,
  tpc4 +1.15, tpc8 +2.08, tpc16 +1.86) but every between-run CI OVERLAPS at n=4; the "rise" is one step
  (tpc4→tpc8) and tpc8→tpc16 declines; and propagation hadn't converged at 20 epochs for the big runs.

## Counter-arguments / threats to validity
- **Steps ≡ data here.** At fixed 20 epochs, tpc16 got ~15× more gradient updates than tpc1 (1200 vs 80
  steps), perfectly collinear with trajectory count — "more data" and "more optimization" are inseparable.
- **The plateau was read off the wrong curve.** eval_kl-plateau ≠ propagation-plateau; propagation was still
  rising at tpc8/tpc16, so any "saturation ~240 traj" is premature.
- **n=4 probes/hop.** Within-hop SEM ≈ 0.2–0.95; the h0 non-monotonicity and most between-run differences
  are within noise. Only eval_kl and the paired h0/h1 sub-ceiling gap are statistically clean.
- **h3 probes are degenerate.** Prompting itself only moves h3 by +0.56 (vs +3–3.5 at h0–h2); 2/4 h3 probes
  have saturated priors (+3.8, +5.2). So "baking never reaches h3" is a probe-design artifact, NOT a
  depth failure — DROP the h3 claim and replace those probes (already captured in [[q-propagation-hardening]]).
- **Below-ceiling ≠ fundamental data limit.** With only rank 16 / 20 epochs / one fact, the gap could be a
  LoRA-capacity or optimization limit; no rank/epoch sweep rules that out.
- Single seed, single fact, single generation seed (so quantity vs sampling-luck is unidentified).

## Implications
Strengthens [[propagation-bounded-by-trajectory-coverage]]: eval_kl is decoupled from propagation not only
in LEVEL (off-topic low-KL but no injection) but in TIME (it converges first). Practical consequences for
the whole agenda: (1) **train to PROPAGATION convergence and early-stop on baked_shift, not eval_kl**;
(2) **a clean size study must hold gradient steps fixed and use multiple seeds + non-saturated probes**;
(3) trajectory quantity is, at best, a weak lever compared to trajectory COVERAGE/TYPE (cycle 1's gate).

## Next steps
- De-confounded size re-run (see refined [[q-propagation-trajectory-size]]): match TOTAL gradient steps
  across tpc, ≥3 generation seeds, train to baked_shift convergence, ≥12 probes/hop with recorded priors.
- Fix the probe bank (low-prior consequences, ≥12/hop) under [[q-propagation-hardening]].
- Prioritize [[q-propagation-trajectory-type]] (coverage) — cycle 1 suggests TYPE dominates SIZE.
