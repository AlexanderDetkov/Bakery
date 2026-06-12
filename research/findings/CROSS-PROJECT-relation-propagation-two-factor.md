---
title: "Cross-project synthesis (~/Invertibility × Bakery): relational knowledge propagation factorizes into a GLOBAL relation operator × LOCAL entity bindings — and the two projects study the two factors"
outcome: synthesis          # connects the Invertibility reversal-curse results to the Bakery baking-vs-prompting findings
confidence: medium          # the mapping is principled and each cell is backed by data, but the unification itself is interpretive
created: 2026-06-12
question: cross-project (links q-grokking-converse-via-longer-training, q-sft-vs-bake-reversal-curse, q-regularization-preserves-behavior)
metric: —
run_ids: [qa-reg0, qa-reg32]   # the live block-diagonal confirmation; Invertibility runs cited by path below
---

## The Invertibility results being connected (results/, manual pre-loop study; KB there is still empty)
Inverse world: learn forward edges `[a, R, f(a)]`; test held-out `[f(a), L, a]` (`test_map_element_acc_inv`).
- **Consistency, not count:** pure-inverse held-out accuracy rises with N (0.69@500 → 0.87@1k → 0.94@2k; 3 seeds),
  but at FIXED N_inv=500 adding entangled random-reverse pairs collapses it (0.69 → 0.09@N=1k). Driver = inverse
  fraction p=N_inv/N (results/grid_final/figure_N_vs_Ninv.png, aggregated_inv_accuracy.csv).
- **Global reinforcement, locally gated:** K disjoint clean blocks each generalize at the level of the TOTAL
  consistent count N, not their own block size M (observed ≈ global pred ≫ local pred;
  results/disentangle/figure_disentangle.png).
- **Entanglement kills, disjointness doesn't:** the same inconsistent pairs collapse the inverse when entangled
  in-vocabulary (→0.07) but leave it intact (~0.87) in a disjoint block (results/grid_block/figure_block_vs_entangled.png).
- **Composition required:** T=1 isolated edges never generalize; T≥2 paths grok (agenda focus #1; grokking there
  is late/transient/regularization-driven).

## The two-factor claim
**Propagation through a relation = (i) an abstract relation OPERATOR in the weights × (ii) entity-specific
BINDINGS injected locally.** Invertibility studies how factor (i) is BUILT from scratch (needs high consistency
p, composition T≥2, global pooling; emerges by grokking; destroyed by entangled inconsistency). Bakery studies
the regime where factor (i) ALREADY EXISTS (pretrained 8B inherited its relation operators from pretraining —
effectively enormous N_consistent). Prompting EXERCISES the operator at inference time; baking COPIES the
operator's output distribution into a LoRA; SFT RE-BINDS entities directly against ground-truth labels.

## How this explains the cross-project asymmetries (each cell is a recorded result)
| Invertibility (from-scratch) | Bakery (pretrained 8B) | Two-factor reading |
|---|---|---|
| Reversal curse at T=1 | NO curse for bake or SFT ([[bake-tracks-teacher-sft-sharpens]]) | the curse = ABSENCE of operator (i); the 8B's operator pre-exists, so neither objective recreates the curse |
| Inverse groks LATE (transient) | NO grokking to 10k ep ([[no-grokking-converse-installed-early]]) | grokking is the cost of BUILDING an operator; distillation/binding into a pretrained model builds nothing |
| held-out acc tracks consistent-data scale | baked d′ tracks TEACHER capacity 1B→3B→8B (size trend in [[bake-tracks-teacher-sft-sharpens]]) | both measure the strength of factor (i): built-by-training there, inherited-by-pretraining here |
| T≥2 paths required | coverage gate ([[propagation-bounded-by-trajectory-coverage]], [[trajectory-type-is-a-binary-coverage-gate]]); CoT rescues single-pass | composition in the training/inference signal is what engages or extends the operator |
| entangled inconsistency collapses the inverse | contrastive trajectories failed because the TEACHER emits converse-affirmations ([[contrastive-trajectories-do-not-fix-the-converse]]) | inconsistent relation signal in-distribution destroys directionality at either layer |
| disjoint blocks don't interfere | **LIVE:** SQuAD anchors (a disjoint block) leave propagation intact — qa-reg32 conv d′ 1.16 ≈ control 1.06 @ep~140 | the block-diagonal no-interference prediction, holding in the LoRA/baking regime |

## Predictions this generates (falsifiable, cheap)
1. **mix_bake near-step should soften from-scratch / at small scale:** [[fidelity-sharpness-frontier-no-knee]]
   found the KL→CE mix degenerate on the 8B because CE has no operator to build (only bindings to sharpen). In a
   regime where CE must BUILD the operator (from-scratch Invertibility model, or a 1B where the teacher is
   useless), the KL leash to a weak teacher should fight operator-construction continuously → a real curve, maybe
   even KL-hurts-monotonically. Testable in Invertibility by adding a distillation arm.
2. **Bakery's converse health should track pretraining-relation frequency:** chains phrased with relations that
   are rare/inconsistent in pretraining text should reproduce the Veld-style converse failure even on the clean
   instrument (operator weak → baked converse collapses), connecting back to "converse reliability is
   chain-specific" ([[CORRECTED-picture-robust-metric]]).
3. **Anchors entangled with the world's vocabulary should hurt:** rerun the regularization sweep with anchors that
   REUSE lw_alpha entity tokens in inconsistent statements — grid_block predicts interference where disjoint
   SQuAD anchors showed none.

## Counter-arguments / threats to validity
- The unification is interpretive: "operator × binding" is a reading, not a measured decomposition; no shared
  metric crosses the projects (d′ vs held-out edge accuracy).
- Invertibility evidence is from the pre-loop manual study (results/ + figures; its research/ KB is empty — no
  committed finding pages with counter-arguments yet); the 82%→2% framing earlier cited in Bakery's grokking
  question matches figure_N_vs_Ninv qualitatively but the canonical numbers are the CSV's (0.69→0.09 at N_inv=500).
- The live block-diagonal cell (qa-reg32) is mid-training (ep~140/300) at n=1 seed; could shift by completion.
- Scale confound: "pretraining supplies the operator" is inferred from the 1B/3B/8B trend, not from inspecting
  circuits; an alternative reading is generic capacity rather than relation-specific operators.

## Implications
The reversal curse and prompt-baking's teacher-ceiling are the SAME constraint seen from opposite ends: you can
only propagate through a relation to the extent an abstract relation operator exists in the weights. Training
builds it slowly (consistency × composition × scale, grokking); pretraining amortizes it; prompting invokes it;
baking faithfully copies its output (and so inherits its limits); SFT bypasses it for bindings but cannot exceed
it where it's absent (Invertibility's curse) — and need not where it's strong (Bakery's no-curse).
