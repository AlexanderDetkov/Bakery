---
title: Does regularizing the bake with base-anchored irrelevant-question trajectories preserve general behavior without hurting propagation?
status: open
priority: medium       # new capability (shipped this session); queued behind the running n-sweep
created: 2026-06-07
hypothesis: Mixing "anchor" trajectories — teacher = base(no prompt, adapter OFF), student = baked(no prompt, adapter ON), supervised on tokens the BASE generated — into the bake pulls the adapter toward IDENTITY on irrelevant (SQuAD) inputs. As the anchor count rises, behavior_drift on held-out irrelevant questions should FALL (less degeneration) while held-out propagation (dprime at proof-depth ≥ 2) should be roughly preserved until anchors start crowding out the logic signal. The interesting regime is whether there is a strength where drift drops materially with little propagation cost.
acceptance_criteria: "Run configs/sweeps/regularization_strength.yaml (bake_theorem_qa, n=1, num_train_contexts ∈ {0,32,128}, 8B, 300 ep). DECISIVE if behavior_drift falls monotonically with anchor count AND the held-out baked dprime (d≥2) of the best-drift arm is within noise of the 0 arm — i.e. you can buy behavior preservation cheaply. Also a result if drift is already ~0 at 0 anchors (baking n=1 axioms doesn't degrade general behavior) or if anchors visibly suppress propagation (a cost). Report eval_kl, dprime (+ dprime_baked_d1_baseline, held-out d≥2), behavior_drift per arm/epoch."
experiment: "configs/sweeps/regularization_strength.yaml — run when a GPU frees; do NOT preempt the n-sweep"
links: [[q-propagation-prompt-vs-bake]] [[q-propagation-trajectory-size]]
---

## Question
Baking distills the prompted base (teacher = base+u) into an unprompted adapter. Left unconstrained,
that can drift the model's behavior on inputs unrelated to `u` (a catastrophic-forgetting effect). The
regularization feature (shipped this session) adds anchor trajectories that say "on irrelevant
questions, behave exactly like the original base (no prompt, no adapter)". Does this preserve general
behavior, and at what cost to knowledge propagation?

## Mechanism (already implemented + tested)
- Anchor trajectory = `FramedTrajectory` with `base_input_ids == baked_input_ids` (both EMPTY prompt),
  supervised on tokens the BASE model generated (greedy, adapter OFF). The one KL primitive then
  computes `KL(base_no_prompt ‖ baked_no_prompt)` → drives the adapter toward identity there. Zero
  objective/runner/gate changes (see `bakery/trajectories/regularization.py`).
- Strength knob: `--regularization.num_train_contexts` (count of anchors; 0 = OFF). Source = SQuAD.
- Eval: `behavior_drift` metric = mean `KL(base ‖ baked)` on a HELD-OUT slice of the anchor pool
  (disjoint from the trained anchors), greedy reference (fixed target across epochs). Lower = closer
  to the base. Held-out propagation read off the existing `dprime` (d≥2).

## Plan
- 3 arms (num_train_contexts ∈ {0, 32, 128}) on the primary world (lw_alpha), n=1 (axioms only — the
  cleanest baseline to read behavior preservation against), 8B, 300 epochs.
- Plot, per arm: behavior_drift vs epoch (does it fall with anchor count?), held-out baked dprime(d≥2)
  vs epoch (is propagation preserved?), eval_kl vs epoch (fidelity). Trade-off curve: behavior_drift
  vs dprime across arms.

## Notes
- Confound to watch: more anchors = larger anchor SHARE of the per-step mean KL → effectively
  down-weights the logic signal. If propagation drops at 128, separate "anchors crowd out logic" from
  "anchors actively suppress logic" by also reporting train_kl split (not currently logged) or an arm
  that scales epochs to hold logic gradient mass fixed.
- n=1 is the natural first arm; if interesting, repeat at n=2 (where propagation is non-trivial) to see
  whether anchoring interacts with the curriculum.
- behavior_drift is only meaningful with `do_sample=False` (greedy → fixed reference). Keep it greedy.
