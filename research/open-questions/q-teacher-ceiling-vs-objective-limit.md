---
title: Is baking's depth-3 ceiling the TEACHER's reach or the BAKING OBJECTIVE's limit? (teacher-forced ground truth beyond the teacher + matched-trajectory-count control)
status: blocked        # 2026-06-10: teacher-forced clean arm BLOCKED by the gate — needs a builder split fix (see note below)
priority: high
created: 2026-06-07

## 2026-06-10 — teacher-forced clean arm is BLOCKED by the gate (needs a disjoint relation-level split)
Tried `--data.sample_trajectories false` on lw_alpha + eq_alpha, n∈{1,2}. In teacher-forced mode the gate
ENFORCES train/eval disjointness (sampled mode only RECORDS it), and it correctly **refused 3 of 4 builds**:
`assert_probes_heldout` fired because the teacher-forced training pairs at `train_max_depth` overlap held-out
probes (tf-eq_alpha-n2: "probes tagged expect_heldout are STATED by the trajectories: ['In eq_alpha, are Guva
and Rotezu the same kind?', …]"). Only **tf-eq_alpha-n1** had a naturally-disjoint split and ran clean.
**Actionable next dev step:** the teacher-forced builder must EXCLUDE the held-out probe pairs from the
enumerated training pairs (a relation-level disjoint split), not just rely on `split_seed`; needs a builder
change + a test. The gate is behaving correctly — this is a validity win, not a bug. Until fixed, a clean
leak-free reference exists only for cells where the split happens to be disjoint (e.g. eq_alpha-n1).
hypothesis: >
  [[baked-propagation-tracks-trained-depth-no-compositional-bonus]] showed baked d3 ≤ 0 with no grokking,
  AND prompted d3 = 0 — so sampled baking can't reach d3. But that conflates two limits: (i) the TEACHER
  can't generate d3 (sampled trajectories never state a correct d3 fact), vs (ii) the soft-KL objective /
  LoRA can't INSTALL a fact even when given it. Teacher-forcing the engine-verified ground-truth answers
  (`data.sample_trajectories=False`) injects correct d2/d3 the teacher itself lacks. HYPOTHESIS: with
  teacher-forced d≤n ground truth, baked d′ at the TRAINED depths rises toward 1 (objective CAN install
  given facts), but held-out d=n+1 STILL does not generalize (no compositional bonus) — i.e. the ceiling is
  "trained/teacher coverage", and teacher-forcing raises the coverage bound but not the generalization bound.
  Also: n=2 trains on MORE relations than n=1, so cycle-1's n2>n1 d2 may be a data-QUANTITY effect; a
  matched-count control (cap n=1 trajectories to n=2's count, or vice-versa) isolates curriculum-DEPTH from
  data-quantity.
acceptance_criteria: >
  bake_theorem_qa 8B, lw_alpha, seeds {10,11}. Arm A (teacher-forced): --data.sample_trajectories=False,
  n∈{1,2}. Arm B (matched count): n=1 vs n=2 with per_depth_train_cap tuned so total trained trajectories
  match (log n_train_traj from data_stats; equalize). DECISIVE: (1) if teacher-forced baked d′ at trained
  depths ≫ sampled (cycle-1) → the objective installs given facts and the sampled-d3 ceiling is the TEACHER
  (not the objective); (2) if teacher-forced held-out d=n+1 still ≈ 0 → confirms NO compositional bonus
  independent of teacher; (3) if the n2>n1 d2 gap survives count-matching → curriculum-depth effect is real,
  not data-quantity.
experiment: >
  bake_theorem_qa, Llama-3.1-8B, bake, lora r/α 16, 200 ep, eval_period 10, bs4 ga1.
  A: --data.sample_trajectories False --data.train_max_depth {1,2} --seed {10,11}.
  B: count-matched n=1 vs n=2 (adjust --data.per_depth_train_cap; verify n_train_traj parity in data_stats).
links: [[baked-propagation-tracks-trained-depth-no-compositional-bonus]], [[q-n-curriculum-propagation-dynamics]], [[q-sft-vs-bake-reversal-curse]], [[propagation-bounded-by-trajectory-coverage]]
---

## Question
Cycle-1 showed the propagation frontier sits at max(trained-depth, teacher-reach) with no +1 hop. This
question localizes WHY d3 fails: teacher can't supply it, or baking can't install it. Teacher-forcing is the
clean knife — it hands the bake correct facts the prompted teacher never generates. It also doubles as the
SFT-adjacent arm (teacher-forced KL ≈ distilling ground truth), connecting to [[q-sft-vs-bake-reversal-curse]]
without a separate SFT run (the user deprioritized SFT, so frame this as "teacher-forced bake", not SFT).

## Plan
Queue AFTER the seed-CI (seeds 12,13) frees GPUs. 4 GPUs: teacher-forced n∈{1,2} × seed∈{10,11} (Arm A) in
one wave; the matched-count control (Arm B) in a second wave. Heavy (8B) but each ≤ ~5–8 h.

## Notes
- This is the control that turns the cycle-1 finding from "baking is teacher-bounded" into a mechanistic
  claim about WHERE the bound comes from (data coverage vs objective capacity).
