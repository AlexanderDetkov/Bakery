---
title: Does the TYPE of trajectory (what contexts elicit) control how far a baked fact propagates?
status: open
priority: high
created: 2026-06-06
hypothesis: Propagation is bounded by what the trajectory distribution exercises. Baking over "restate" contexts (which only elicit the bare fact) injects the 0-hop fact but little else; baking over "consequence/reasoning" contexts (which elicit the model to discuss downstream implications) propagates further to higher hops; baking over "neutral" contexts (generic QA where the fact never comes up) injects almost nothing. I.e. propagation depth tracks the deepest hop the trajectories touch.
acceptance_criteria: "Hold trajectory COUNT *and* convergence (matched eval_kl plateau or matched supervised-token budget) fixed; vary category in {restate, consequence, neutral, mixed}. DECISIVE if the on-topic vs off-topic gap survives count+convergence matching with bootstrap CIs over probes that exclude 0, AND we can (or cannot) separate restate vs consequence at high hops once convergence is matched."
experiment: "bake_fact --data.context_category {restate|consequence|neutral|mixed}, count fixed, on 1B fanned across GPUs"
links: [[q-propagation-prompt-vs-bake]], [[q-propagation-trajectory-size]]
---

## Question
The user asked whether propagation depends on the TYPE of trajectories. This is the sharpest version of
the central claim: baking distills a token distribution, so a consequence the trajectories never express
cannot be distilled — baked propagation should be capped at the deepest hop the trajectory contexts
exercise, whereas prompting carries the fact into every forward pass for free.

## Plan
- The `fact_propagation` builder reads a context bank tagged by category. Run one bake per category at
  matched trajectory count; compare baked_shift_h{n} curves. The mixed/consequence conditions should
  dominate at high hops; restate should collapse to ~0 beyond hop 0.

## Notes
- This is the mechanism behind [[q-propagation-prompt-vs-bake]]: if it holds, "baking propagates less far
  than prompting" is really "the trajectories didn't reach that hop", and is FIXABLE by choosing
  consequence-rich trajectories. A strong, actionable finding if true.
- **Cycle-1 partial result** ([[propagation-bounded-by-trajectory-coverage]]): the categorical on-topic/
  off-topic GATE is confirmed (neutral, the most-converged run, injects ≈0; on-topic reaches h2). BUT the
  cycle-1 sweep used 48 traj (vs the mixed run's 120) and restate was under-converged (eval_kl 0.140), so
  the finer restate-vs-consequence ordering is NOT yet identified. This question now = the count+convergence-
  matched re-run. Expand the context bank if needed for a larger matched count.
