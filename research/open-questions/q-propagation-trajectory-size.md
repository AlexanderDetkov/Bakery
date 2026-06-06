---
title: Does baked-fact propagation depth scale with the number/size of trajectories?
status: open
priority: medium       # cycle-2 first attempt inconclusive + confounded; cycle 1 suggests TYPE dominates SIZE
links_finding: [[size-helps-fidelity-not-the-propagation-gap]]
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
- **Cycle-2 first attempt INCONCLUSIVE** ([[size-helps-fidelity-not-the-propagation-gap]]): varied
  trajectories_per_context {1,4,8,16} at fixed 20 epochs. eval_kl fell monotonically (0.107→0.042) and
  baking stayed below the prompting ceiling at h0/h1, but the "propagation rises then plateaus" claim was
  confounded and unconvincing: (a) at fixed epochs, gradient STEPS were collinear with trajectory count
  (15× span); (b) eval_kl had plateaued but PROPAGATION was still rising at 20 epochs (so the plateau was
  read off the wrong curve); (c) n=4 probes/hop → all between-run CIs overlap.
- **De-confounded re-design (do this before re-claiming):** match TOTAL gradient steps across tpc (adjust
  epochs so each gets the same #updates), train to baked_shift convergence (early-stop on propagation, not
  eval_kl), ≥3 generation seeds, ≥12 probes/hop with recorded non-saturated priors. Vary num_contexts
  (coverage) as a separate arm from trajectories_per_context (sample density).
