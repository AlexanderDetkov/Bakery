---
title: Does the n-hop training curriculum (≤n-hop trajectories) extend BAKING's propagation depth toward PROMPTING's, and how does the per-depth d′ evolve over grokking-length training across seeds?
status: active
priority: high
created: 2026-06-07
hypothesis: >
  Baking propagation is bounded by what the trajectories exercise ([[propagation-bounded-by-trajectory-coverage]],
  [[trajectory-type-is-a-binary-coverage-gate]]) and by the prompted teacher's own reach (8B prompted d′
  propagates ~2 hops, d3+≈0 — lw_alpha sanity). So training on a DEEPER curriculum (≤2-hop vs ≤1-hop
  atomic-only trajectories) should EXTEND the baked model's held-out propagation depth — but only up to the
  teacher ceiling under sampled (canonical) baking, since the teacher generates the answers being distilled.
  Concretely: (H1) n=2 baked d′ at held-out depth ≥3 exceeds n=1 baked d′ at the same depths (curriculum
  pushes the frontier out by ~1 hop); (H2) baked propagation lags PROMPTING at every depth the teacher itself
  reaches (bake ≤ teacher ceiling); (H3) per-depth d′ is a GROKKING curve — it keeps rising for many epochs
  AFTER eval_kl plateaus ([[size-helps-fidelity-not-the-propagation-gap]]: "eval_kl converges before belief"),
  so short (15-40 ep) bakes under-read the achievable depth.
acceptance_criteria: >
  On lw_alpha 8B, bake objective, sampled teacher, seeds {10,11}, n∈{1,2}, 600 epochs (eval_period 10):
  DECISIVE-POSITIVE for H1 if mean(n=2) held-out d′ at depth≥3 exceeds mean(n=1) by an amount whose 2-seed
  spread excludes 0 (and prop_distance_dprime(n=2) > prop_distance_dprime(n=1)). DECISIVE for H3 if the epoch
  at which per-depth d′ first crosses ~1.0 is materially LATER than the epoch eval_kl plateaus (quantify the
  gap). Report prompted d′ (the same teacher for both n) as the propagation ceiling; baking "approaches
  prompting" iff baked d′ → prompted d′ at the depths the teacher reaches. NEGATIVE/NULL (still a finding) if
  n has no effect on held-out depth across seeds, or baked d′ stays ≈0 at depth≥2 to 600 ep despite eval_kl→0.
experiment: >
  bake_theorem_qa, Llama-3.1-8B-Instruct, lora_rank/alpha 16, objective=bake, sample_trajectories=True
  (canonical), num_epochs 600, eval_period 10, save_every 150, batch_size 2 grad_accum 2, lr 1e-4,
  per_depth_train_cap 16, regularization OFF. CYCLE-1 LAUNCH (all 4 GPUs):
  qa-bake-n{1,2}-s{10,11} with --data.train_max_depth ∈ {1,2}, --seed = --data.split_seed ∈ {10,11}.
  Follow-ups (own questions / later cycles): trajectory-regularization axis
  ([[q-regularization-preserves-behavior]], --regularization.num_train_contexts {0,32,128}); more seeds for
  CIs; teacher-forced arm (data.sample_trajectories=False) to separate "bake the teacher's behavior" from
  "inject ground truth"; model scale.
links: [[propagation-bounded-by-trajectory-coverage]], [[size-helps-fidelity-not-the-propagation-gap]], [[trajectory-type-is-a-binary-coverage-gate]], [[q-grokking-converse-via-longer-training]], [[q-regularization-preserves-behavior]], [[q-propagation-model-scale]]
---

## Question
The central agenda question is "how does baking differ from prompting, and how does it depend on the
trajectories." This question makes the dependence on the TRAJECTORY CURRICULUM precise on the rigorous
proof-system instrument (`bake_theorem_qa`: proof-depth = hop, bias-immune d′, contamination-guarded,
held-out-by-depth). Train depth-1 (all axioms) only (n=1) vs depth-1 plus a balanced depth-2 subset (n=2),
hold out everything deeper, and ask: does the deeper curriculum let the BAKED (unprompted) model answer
held-out probes deeper, and does it ever match PROMPTING (base + the axiom prompt) — which is the same
teacher in both arms? The user's framing: "the difference between prompting knowledge propagation and
baking on datasets with different n=1,2 … I am interested in training dynamics … baking has grokking-like
behavior so we may need to bake for a while." Hence long training + per-depth d′ logged every eval =
grokking curves.

## Plan
CYCLE 1 (launched, heavy — 4× 8B, 600 ep): the n×seed factorial above, regularization off, the clean
baseline. Read held-out d′ vs depth per arm + the per-epoch d′ trajectory (grokking) overlaid on eval_kl.
LATER: layer in trajectory-regularization (anchor SQuAD trajectories) as a second axis; add seeds for CIs;
add the teacher-forced contrast. Keep all GPUs busy — relaunch the next factorial slice as runs free.

## Cycle-1 result (S4c1, 2026-06-07) → [[baked-propagation-tracks-trained-depth-no-compositional-bonus]]
n∈{1,2} × seed∈{10,11}, 8B, bake, stopped epoch 110–170 (dynamics saturate ~ep50). DECISIVE: prompting
ceiling `[1.12,0.53,0,0]` (d1..d4); baked d3 ≤ 0 every arm, no grokking through ep170 → baking bounded by
max(trained-depth, teacher-reach), **no +1 compositional hop**, deeper hop does NOT grok. SUGGESTIVE but
seed-confounded: n=2 d2 (1.05) > n=1 d2 (0.74) on average, but per-seed Δ ∈ {−0.01, +0.63} (n1-s10 anomaly).
→ **seed-CI RUNNING** (seeds 12,13) to settle the d2 gap. Question stays **active**.

## Notes
- Reseeding everything (split_seed = seed) is intentional: a propagation phenomenon that only holds for one
  held-out split is not robust. 2 seeds is a weak CI — cycle-1 reads the SHAPE; CIs come with more seeds.
- Sampled teacher (canonical baking) is the faithful "bake what prompting does" choice; see decision
  [[sampled-teacher-trajectories-keep-cot]] for the recall-of-recited caveat (mitigated here by short answers
  + the DAG contamination filter). A teacher-forced arm is the natural control.
- Watch behavior_drift too (logged) even with reg off, as the baseline the regularization axis improves on.
