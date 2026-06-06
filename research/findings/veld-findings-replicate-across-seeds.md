---
title: The Veld findings (converse-affirmation, single-pass yes-saturation, forward<prompting) replicate tightly across 3 seeds
outcome: positive
confidence: high            # cross-seed SDs tiny; prior/prompted seed-invariance is a clean internal check; still ONE fact
created: 2026-06-06
question: [[q-propagation-hardening]]
metric: propagation
run_ids: [prop-veld-1b-mixed, prop-veld-1b-mixed-s1, prop-veld-1b-mixed-s2, prop-veld-8b-mixed, prop-veld-8b-mixed-s1, prop-veld-8b-mixed-s2]
---

## Insight
The central cycle-3/5 results on the Veld chain are NOT single-seed flukes: across 3 independent seeds
(0,1,2; different context split, sampled trajectories, and adapter init) the baked model's converse-
affirmation, single-pass yes-saturation, and below-prompting forward shift all replicate with small
cross-seed spread. The qualitative findings are robust; only the multi-fact generalization remains.

## Evidence (mean ± cross-seed SD over seeds 0,1,2; no-CoT belief)
8B:
- forward baked shift +5.72 ± 0.55 [4.94, 6.10, 6.11]; prompted +8.35 (baking < prompting, every seed).
- **converse baked belief −7.00 ± 0.27 [−6.61, −7.18, −7.20]** vs prior −1.86, prompted −1.88 — baking
  strongly affirms the FALSE converse (deficit ≈ −5.1 vs prior), tightly across all 3 seeds.
- baked yes-saturation fracYes 0.99 ± 0.02 — answers "Yes" to ~every probe, every seed.
1B:
- forward baked shift +0.52 ± 0.28; converse baked belief −1.53 ± 0.25; fracYes 1.00 ± 0.00.

Internal check: prior and prompted beliefs are IDENTICAL across seeds (SD 0.00) — correct, since they don't
depend on the adapter — confirming determinism and that the seed only moves the baked (LoRA) behavior.
eval_kl across seeds: 1B {0.093,0.112,0.091}, 8B {0.090,0.088,0.061}.

## Counter-arguments / threats to validity
- **Still ONE fact/chain (Veld)** — the hardening criterion also asks for ≥2 additional facts; seed-robustness
  ≠ fact-generality. A different chain could behave differently (esp. restate-vs-consequence separation).
- Still 6 probes/depth (5 true converse); ≥12/depth not yet done, so per-hop CIs remain wide (this finding uses
  cross-SEED spread of per-condition means, which is tight, but the within-probe-set n is unchanged).
- 8B seed-2 eval_kl (0.061) is lower than seeds 0/1 (~0.09) — mild generation-luck variation; the propagation
  numbers are nonetheless consistent.
- Replication confirms the EFFECTS exist robustly; it does not change the behavioral (non-mechanistic) nature
  of the readout.

## Implications
Upgrades the confidence of [[baking-is-associative-prompting-is-directional]] and
[[cot-chains-baked-rules-but-not-the-converse]] from "single-run/medium" toward "seed-robust": baking's
direction-blind converse-affirmation and single-pass yes-saturation are stable properties (on this fact), not
seed artifacts. Strengthens the capstone [[SYNTHESIS-baking-vs-prompting-propagation]].

## Next steps (remaining hardening, keeps [[q-propagation-hardening]] active)
- Add ≥2 more fact/chain banks (a second synthetic chain + a real-entity counterfactual) × the 3 seeds, to
  test fact-generality of converse-affirmation + yes-saturation.
- Expand probe banks to ≥12/depth (incl. ≥8 converse) for tight per-hop CIs.
- On a fact with NON-obvious consequences, re-test restate-vs-consequence (cf. [[trajectory-type-is-a-binary-coverage-gate]]).
