---
title: Does baked-fact propagation depth scale with the number/size of trajectories?
status: open
priority: high
created: 2026-06-06
hypothesis: Propagation depth (the largest hop n at which the baked belief shift remains a meaningful fraction of the prompted shift) increases monotonically with the amount of trajectory data baked, with diminishing returns — more trajectories give the LoRA more chances to internalize the fact and its frequently-co-occurring consequences.
acceptance_criteria: "Sweep generation.num_contexts in {15, 30, 60, 120} (traj/context fixed). DECISIVE if the baked per-hop shift curve rises with trajectory count at fixed hop, OR clearly plateaus — either is a result. Report propagation.baked_shift_h{0..3} and eval_kl per point."
experiment: "configs/sweeps/propagation_size.yaml (bake_fact, grid over generation.num_contexts), fanned across GPU 0/1"
links: [[q-propagation-prompt-vs-bake]]
---

## Question
The user asked whether propagation scales with the SIZE of trajectories. Baking is distillation over a
finite trajectory sample; more samples = a denser estimate of the prompted model's behavior, which may
let deeper consequences (that appear only occasionally in trajectories) get internalized.

## Plan
- After the primary comparison establishes the metric, sweep `generation.num_contexts` (and optionally
  `trajectories_per_context`) on the cheap 1B model, fanned across both GPUs (CUDA_VISIBLE_DEVICES).
- Plot baked_shift_h{n} vs num_contexts for each hop n; compare against the (constant) prompted_shift.

## Notes
- Watch the confound: more trajectories also lowers eval_kl generally; we want propagation depth, not
  just fidelity on the trajectory distribution. Separate the two by reporting both.
