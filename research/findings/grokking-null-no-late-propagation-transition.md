---
title: No compositional "grokking" — deep-depth propagation does NOT rise late on the proof-system instrument (flat through ep2000); a low-rank distillation adapter installs trained-depth propagation early and never gains a free hop
outcome: negative          # decisive-negative for the grokking hypothesis (a clean null)
confidence: high           # 2000 epochs (≈12× the prior ep170 null), n∈{1,2}, two weight-decay settings
created: 2026-06-10
question: [[q-grokking-converse-via-longer-training]]
metric: dprime
run_ids: [grok-n1-wd0.0, grok-n1-wd0.1, grok-n2-wd0.0, grok-n2-wd0.1]
---

## Insight
Baking installs whatever propagation depth it is going to reach **early** (by the time eval_kl plateaus,
~ep50) and **never gains a deeper hop with extended training** — there is no grokking-style late transition.
Training the same bakes ~12× longer than any prior run (to **ep2000**, vs the ep170 null in
[[baked-propagation-tracks-trained-depth-no-compositional-bonus]]), across curricula n∈{1,2} and weight-decay
∈{0, 0.1}, the deep-depth (d2/d3) baked AUROC stays **flat-to-declining** while eval_kl is long-since flat.
This converts the earlier "no grokking through ep170" into a **decisive negative**: a rank-16 additive LoRA
distilled to copy single-pass teacher behaviour does not, with more optimisation, develop the +1 compositional
hop it didn't have at convergence. (Fig: `results/bake_theorem_qa/_fig_grok_depth.png`.)

## Evidence (grok-n{1,2}-wd{0,0.1}, lw_alpha, Llama-3.1-8B, fast recipe + long training; baked AUROC by epoch)
| arm | eval_kl ep50→ep2000 | d2_ALL ep50→late | d2_cross ep50→late | d3_ALL ep50→late |
|---|---|---|---|---|
| n1 wd0.0 | 0.232 → 0.292 (flat) | 0.758 → 0.676 | 0.750 → 0.575 | 0.633 → 0.578 |
| n1 wd0.1 | 0.218 → 0.236 (flat) | 0.734 → 0.691 | 0.700 → 0.562 | 0.555 → 0.598 |
| n2 wd0.1 | 0.140 → 0.160 (flat, to ep1500) | 0.809 → 0.777 | 0.838 → 0.762 | 0.402 → 0.324 |
| n2 wd0.0 | 0.148 → **1.45 (DIVERGED @ ~ep1375)** | 0.805 → 0.668 | 0.750 → 0.750 | 0.387 → 0.379 |

- **No late rise at ANY propagation depth.** Across all stable arms, d2 and d3 baked AUROC are flat or slightly
  *declining* from ep50 to ep2000 — the opposite of a grokking transition. d3 for the n=2 arms sits *below*
  chance (~0.32–0.40) the entire time and never recovers: the trained-depth-2 model confidently mis-ranks some
  3-hop negatives above truth, and 1500–2000 epochs do not fix it.
- **eval_kl plateaus by ~ep50 and stays there** (the distillation loss is "done" ~40× before training stops),
  while propagation never improves → reconfirms eval_kl ⟂ propagation, now in the long-training limit.
- **Weight decay does not induce grokking** (its classic role): wd=0.1 vs wd=0.0 are indistinguishable on
  d2/d3 trajectory shape; neither shows a transition.

## Secondary result — weight decay buys STABILITY at long bake lengths (refines "wd is a no-op")
The wd=0.0 **n=2** arm diverged: eval_kl 0.186 (ep1350) → 1.65 (ep1400) → stayed ~1.4–1.6 — a training blow-up
under the higher-capacity 2-hop curriculum with no regularisation. The wd=0.1 n=2 arm stayed flat (0.157→0.160)
through ep1500. So [[faster-training-recipe]]'s "weight decay is a no-op" holds only in the **short-bake regime**
(≤~200 ep, where the recipe lives); at 1000+ epochs **wd=0.1 is the difference between a stable and a divergent
n=2 bake**. Not a propagation lever, but a stability one for anyone running very long bakes.

## Counter-arguments / threats to validity
- **This tests DEPTH/compositional grokking on the proof-system instrument, NOT the converse-on-Veld grokking**
  that [[q-grokking-converse-via-longer-training]] originally framed (bake_fact Veld, `converse_acc_baked`).
  That said, the proof-system instrument already shows **no converse failure to grok**
  ([[converse-collapse-does-not-survive-bias-immune-instrument]]: baked converse-AUROC ≥ prompting), so the
  "converse is the reversal curse, learnable late" hypothesis has no deficit to rescue *here*; the Veld
  belief-metric version remains formally untested (E2 path/converse trajectories never run).
- **n2-wd0.0 diverged**, so its grok arm is inconclusive past ep1375 — but n2-**wd0.1** is the clean stable n=2
  null, and both n1 arms are stable to ep2000, so the negative does not rest on the diverged arm.
- **CoT leak** ([[sampled-teacher-trajectories-keep-cot]]) inflates absolute d2/d3 but is constant across epochs,
  so it cannot hide a *late rise* — the null is about the trajectory shape, which is unaffected.
- Single world (lw_alpha) for the long runs; the depth ceiling itself is already 12-graph-general
  ([[converse-collapse-does-not-survive-bias-immune-instrument]]), so structure-generality is inherited.

## Implications
- **Closes the "undertraining" escape hatch** for [[baked-propagation-tracks-trained-depth-no-compositional-bonus]]:
  baking's depth ceiling (= max(trained-depth, teacher-reach), no free +1 hop) is a property of the
  low-rank-copy objective, NOT an artifact of stopping at 15–200 epochs. More compute does not buy a deeper hop.
- Strengthens the v2 synthesis ([[SYNTHESIS-baking-vs-prompting-propagation]]) claim #2 (hard teacher ceiling)
  to the long-training limit.
- Practical: keep bakes short (the fast recipe is not just cheaper, it's all you get); if you must run long
  (e.g. for a different reason), use wd=0.1 to avoid divergence on n≥2 curricula.

## Next steps
- The remaining grokking question is now narrowly the **converse-on-paths** version (E2): bake on
  converse/"path" trajectories that actually exercise the reverse direction, on a chain where the teacher
  rejects the converse, and watch `converse_acc` late. Lower priority — the proof-system instrument shows no
  converse deficit to fix, so this is about the Veld belief-metric chain specifically.
- The equivalence-world sweep (in flight, [[converse-collapse-does-not-survive-bias-immune-instrument]] §next)
  is the higher-value next test of the surviving mechanism (cross-component over-connection).
