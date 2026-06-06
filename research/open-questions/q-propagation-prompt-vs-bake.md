---
title: Does a baked fact propagate to n-hop consequences as far as the same fact in the prompt?
status: resolved         # open | active | resolved | parked  (first-pass; deeper claims -> new questions)
priority: high
created: 2026-06-06
hypothesis: A prompted fact u influences the model's beliefs about its n-hop consequences with a propagation curve that decays with hop distance n; baking u into LoRA weights reproduces the 0-hop fact but propagates LESS far to higher-hop consequences than prompting does, because baking only distills the teacher's token distribution over the trajectory contexts and never sees the deeper consequences unless the trajectories exercise them.
acceptance_criteria: "On a held-out forced-choice (no-CoT) probe bank with hops 0..3, measure belief shift Δ=logP(fact-consistent)-logP(contrast) relative to the no-fact prior, for three model states on ONE checkpoint: prior (base,no u), prompted (base,+u), baked (adapter,no u). DECISIVE if (a) prompted shift > 0 at hop 0 (sanity: prompting works) and (b) we can compare the prompted vs baked per-hop shift curves with the propagation metric and eval_kl < 0.1 (faithful bake)."
experiment: bake_fact --model.name meta-llama/Llama-3.1-8B-Instruct --generation.base_prompt data/prompts/tsunami_u.md
links: [[propagation-bounded-by-trajectory-coverage]], [[q-propagation-trajectory-size]], [[q-propagation-trajectory-type]], [[q-propagation-hardening]], [[q-propagation-model-scale]], [[q-propagation-cot-confound]]
---

## RESOLUTION (cycle 1 — see [[propagation-bounded-by-trajectory-coverage]])
Acceptance criteria met: prompted shift > 0 at h0 (1B +3.48, 8B +7.07; prompting works), prompted vs baked
per-hop curves compared, faithful bake (8B eval_kl 0.053 < 0.1). **Answer:** baking does NOT propagate as
far as prompting in general — but the dominant effect is that **baking only injects what the trajectories
exercise** (the headline confirmed result C3 + the on-topic/off-topic gate), not a clean "decay with
reasoning distance." The per-hop "gap grows with hops" pattern (C1) is real on 8B but reverses on 1B and is
confounded by far-hop prior saturation, so it is demoted to *suggestive, single-run* and moved to
[[q-propagation-hardening]]. Model-scale faithfulness (C4) moved to [[q-propagation-model-scale]].

## Question
The central question the user posed: when we inject a new fact (tsunami in Japan; or, ideally, a new
theorem) either by PROMPTING or by BAKING, how far does the update propagate to questions that are
"n reasoning hops" away? Zero-hop = the fact restated ("is there a tsunami in Japan?"); one-hop =
immediate consequence ("is Japan having a disaster?"); two-hop = derived consequence ("should I avoid
traveling there?"). We want to measure INTERNAL propagation (does the consequence emerge in one forward
pass) and separate it from CoT chaining (which would let a model reach the answer by reasoning aloud).

This matters for the central idea because baking is a KL-distillation of the prompted model over a
trajectory distribution. If propagation is bounded by what the trajectories exercise, then "how deep a
fact bakes in" is a property of the trajectory distribution, not just the fact — a sharp, testable claim.

## Plan
- Measure propagation with a forced-choice, single-/few-token belief readout (NO chain-of-thought):
  for each probe q with a fact-consistent answer `pos` and a contrastive `neg`, belief(q) =
  logP(pos|q) - logP(neg|q). This is a one-forward-pass internal readout, so it cannot be reached by
  CoT chaining. Average over several probes per hop; balance Yes/No polarity within each hop to cancel
  any yes-bias.
- Compute belief for THREE states from ONE bake on ONE checkpoint (paired, full-vocab):
  prior = base()+empty, prompted = base()+u, baked = baked()+empty. The propagation curve is the
  per-hop shift relative to prior.
- First fact: a vivid event ("magnitude-9 tsunami struck Japan today") with rich real-world priors so
  multi-hop consequences are well-defined. Trajectories generated from on-topic elicitation contexts so
  the fact actually bakes in (a neutral context distribution would never inject it — see
  [[q-propagation-trajectory-type]]). Trajectory contexts are DISJOINT from the probe bank.
- Small/cheap: Llama-3.1-8B-Instruct (cached), ~60 train / 20 eval contexts, 4 traj/context, rank 16,
  ~15 epochs. eval_kl tracks bake fidelity; `propagation` metric is the research payload.

## Notes
- Instrument built this cycle: `propagation` metric, `fact_propagation` builder, `bake_fact` experiment,
  assets under data/prompts,contexts,probes/.
- Confound to watch: if the fact never enters the trajectories, baked shift ~ 0 at ALL hops (including 0)
  — that's a trajectory-type result, not a propagation-depth result. Check hop-0 baked shift first.
