---
title: The fidelity↔sharpness frontier — can an intermediate objective trade teacher-fidelity (eval_kl) for probe-discrimination (d′), and which point best reproduces the teacher's held-out BEHAVIOR?
status: active
priority: medium
created: 2026-06-09
hypothesis: >
  [[bake-tracks-teacher-sft-sharpens]] found two extremes on the SAME data: baking (KL→teacher) sits at low
  eval_kl / moderate d′ (faithful but inherits the teacher's weak single-pass propagation); one-hot SFT sits at
  high eval_kl / high d′ (sharp but far from the teacher). These look like the endpoints of a frontier. An
  objective that interpolates — KL at temperature T>1, label-SMOOTHED SFT, or the existing HALF-BAKING α-scaled
  adapter (B(θ,u) with 0<α<1) — should trace an eval_kl↔d′ curve between them. Prediction: as we move from
  pure-KL toward pure-CE, eval_kl rises monotonically while probe d′ rises; the question is whether there is a
  knee where d′ is already near SFT's while eval_kl is still near baking's (a "best of both"), or whether the
  two are strictly traded. Crucially, the RIGHT operating point is the one whose held-out GENERATED answers best
  match the prompted teacher's generated answers — eval_kl and d′ are both proxies; validate against generation.
acceptance_criteria: >
  On theorem_qa n1-s0 (matched cached trajectories), sweep an interpolation knob — minimally {bake (KL),
  half-bake α∈{0.25,0.5,0.75}, sft (CE)}, ideally also KL-temperature — and plot eval_kl vs held-out d′
  (forward + converse) for each. DECISIVE-POSITIVE if some intermediate point achieves d′ within ~10% of SFT's
  while eval_kl stays within ~2× of baking's (a usable knee). DECISIVE-NEGATIVE if eval_kl and d′ move together
  with no knee (strict tradeoff). Either way, free-generate teacher vs each-arm continuations on a few held-out
  contexts and report which arm's GENERATED yes/no answers best match the teacher's.
experiment: >
  Reuse the matched n1-s0 cached trajectories. half-bake is the α-scaled-adapter extension seam (CLAUDE.md
  "half-baking"); if not yet implemented as an objective, scaffold it via /scaffold-new-variant (an Objective
  that scales the adapter delta or mixes KL+CE with weight α) with its own alignment test. Keep it small
  (1000 ep, 8B, seed 0). Plot with analysis/plot_grokking depths + a new eval_kl-vs-d′ scatter.
links: [[bake-tracks-teacher-sft-sharpens]], [[q-sft-vs-bake-reversal-curse]], [[q-grokking-converse-via-longer-training]]
---

## Why this matters
It reframes "bake vs SFT" from a binary into a tunable objective and asks the design question directly: if you
want a LoRA that BOTH mimics the prompted teacher AND discriminates sharply, does such a point exist, or is
prompt-baking's teacher-fidelity inherently in tension with raw task sharpness? Answers whether half-baking
(already a named Bakery seam) buys anything on knowledge tasks.

## IMPLEMENTED + RUNNING (2026-06-11, S4c5)
Scaffolded the `mix_bake` objective (bakery/objectives/mix_bake.py): loss=(1-w)·aligned_KL + w·CE_on_span,
w=`train.mix_ce_weight` (w=0≡bake, w=1≡sft — proven in tests/test_objective_alignment.py; 136 tests + smoke green).
This is the TRAINING-time loss mix (distinct from eval-time `model.half_bake_alpha` adapter scaling).
Launched the α-frontier on GPU1: qa-mix-w{025,05,075}-n1-s0 (8B, lw_alpha, matched recipe). With the existing
endpoints (w=0=qa-sbake-n1-s0 eval_kl 0.47/conv 0.89; w=1=qa-ssft-n1-s0 eval_kl 4.43/conv 1.79) this gives a
5-point eval_kl↔d′ frontier. DECISIVE-POSITIVE if an interior w has converse d′ within ~10% of SFT while eval_kl
stays within ~2× of bake (a usable knee); DECISIVE-NEGATIVE if eval_kl and d′ move together (strict tradeoff).
Run recipe: `--train.objective mix_bake --train.mix_ce_weight <w>` (rest matched to qa-sbake-n1-s0).
