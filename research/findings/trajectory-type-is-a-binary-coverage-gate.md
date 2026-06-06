---
title: Trajectory type is a BINARY coverage gate (on-topic injects, off-topic doesn't) — not graded by reasoning-richness
outcome: positive
confidence: medium-high      # count+convergence-matched, bootstrap CIs exclude/include 0 as predicted; single fact/seed
created: 2026-06-06
question: [[q-propagation-trajectory-type]]
metric: propagation
run_ids: [prop-type2-restate-1b, prop-type2-consequence-1b, prop-type2-neutral-1b, prop-type2-mixed-1b]
---

## Insight
At MATCHED trajectory count (56) AND matched convergence (40 epochs, all eval_kl-plateaued), with bootstrap
CIs, trajectory TYPE acts as a BINARY gate on how much a fact bakes in: on-topic contexts (restate /
consequence / mixed) inject a clear belief shift (CIs exclude 0); off-topic (neutral) injects ~nothing (CI
includes 0). But type does NOT grade propagation by how much reasoning the contexts elicit — restate ≈
consequence ≈ mixed. The lever is COVERAGE (does the fact surface in the trajectories at all), not the
reasoning-depth of the eliciting prompts.

## Method
The clean version of cycle-1's type sweep, fixing both its confounds: matched count (num_contexts=14,
4 traj/ctx = 56 train traj for ALL categories) and matched convergence (40 epochs; all four runs eval_kl-
plateaued). Tsunami chain, 1B, seed 0, fanned across both GPUs. Per-probe logging → bootstrap 95% CIs.

## Evidence (mean baked belief shift vs prior over h0–h2, the propagating range)
- restate:     +1.38 [+0.60, +2.24]   (eval_kl 0.112)
- consequence: +1.33 [+0.64, +2.12]   (eval_kl 0.050)
- mixed:       +1.35 [+0.80, +1.94]   (eval_kl 0.085)
- **neutral:   +0.21 [−0.18, +0.65]   (eval_kl 0.009)** — CI includes 0: no significant injection.
- prompted reference (same across runs): ~+3.4 at h0–h2 — baking (~+1.35) stays well below prompting.

Two sub-results: (1) the on-topic/off-topic GATE is confirmed with CIs at matched count+convergence (the
acceptance criterion). (2) restate vs consequence are statistically indistinguishable (overlapping CIs,
near-identical means) → the "consequence-rich trajectories propagate further to higher hops" hypothesis is
REFUTED at this scale. Also re-confirms eval_kl⟂propagation: neutral has the lowest eval_kl (0.009) yet ~zero
injection; restate the highest (0.112) yet injects as much as consequence.

## Counter-arguments / threats to validity
- Single fact (tsunami), single seed; ~6 probes/hop. The h0–h2 pooled CIs are the reliable level; per-hop is noisier.
- Neutral shows a small significant h2 bump (+0.88 [+0.3,+1.5]) — likely the tsunami h2 probes ("avoid eastern
  Japan?") being partly reachable by generic priors, not fact injection; a probe-design artifact, not propagation.
- "restate ≈ consequence" may be fact-specific: the tsunami's consequences are common-sense, so restate
  trajectories (which the model elaborates anyway) may already carry them. A fact whose consequences are
  non-obvious (e.g. a synthetic theorem) could separate restate from consequence — worth a follow-up.
- Baking still << prompting at all hops (coverage helps injection but doesn't close the prompting gap; cf.
  [[baking-is-associative-prompting-is-directional]]).

## Implications
Resolves [[q-propagation-trajectory-type]] and sharpens [[propagation-bounded-by-trajectory-coverage]]: the
actionable lever for knowledge baking is ensuring the trajectories COVER the fact (any on-topic elicitation
works); engineering "reasoning-rich" trajectories did not help beyond bare coverage for this fact. Directly
answers the user's "type of trajectories" axis: type matters as a coverage gate, not as a reasoning-depth dial.

## Next steps
- Re-test restate-vs-consequence on a fact with NON-obvious consequences (e.g. the Veld synthetic chain, where
  restate ≠ consequence may finally separate) — folds into [[q-propagation-hardening]].
