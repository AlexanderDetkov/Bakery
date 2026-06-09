---
title: Prompting vs SFT vs Baking on the reversal curse — does KL-distillation inherit, worsen, or escape forward-only SFT's converse failure?
status: active
priority: high
created: 2026-06-06
hypothesis: >
  ~/Invertibility (and Berglund et al.'s reversal curse) is fundamentally an SFT phenomenon: a transformer
  trained with next-token CE on forward edges fails the inverse. Bakery's `bake` objective is NOT plain SFT
  — it is KL-distillation of the prompted teacher into a LoRA. Adding a matched `sft` objective (CE on
  exactly the same gate-validated trajectory tokens, no teacher) isolates what the distillation framing
  adds. Predictions: (1) SFT on forward-laden trajectories shows the classic reversal curse — forward_acc
  high, converse_acc ~0 — mirroring the toy model; (2) baking (KL) may differ because the teacher's
  full-vocab distribution carries directional information SFT's hard targets discard. If SFT ≈ baking on the
  converse -> the failure is a property of training on forward text (the reversal curse), not the KL framing.
  If they diverge -> the distillation framing matters. Long training (grokking) may rescue SFT's converse
  just as it does the toy model's inverse.
acceptance_criteria: >
  Veld (and Tellus) 8B, three arms on identical data: prompting (base+u, already a state in the propagation
  metric), SFT (--train.objective sft), baking (--train.objective bake). Report forward_acc and converse_acc
  (tokenization-robust, per-form) for each. DECISIVE on (a) whether SFT and baking diverge on converse_acc at
  matched data/training, and (b) whether extending SFT to grokking length moves SFT's converse_acc. Cross
  against E1/E2 ([[q-grokking-converse-via-longer-training]]).
experiment: >
  bake_fact on Veld 8B with --train.objective {sft, bake} at matched trajectories (mixed) and at the long
  (grokking) recipe; prompting is read directly from the propagation metric's prompted state. Same probe
  bank (data/probes/veld_probes.json, form-labelled). Optionally repeat on Tellus for fact-generality.
  The `sft` objective is implemented in bakery/objectives/sft.py (CE on the supervised span via the audited
  shift; tests in tests/test_sft_objective.py).
links: [[q-grokking-converse-via-longer-training]], [[CORRECTED-picture-robust-metric]], [[propagation-bounded-by-trajectory-coverage]]
---

## Why this is the cleanest bridge to ~/Invertibility
The toy model is literally SFT (next-token CE on sequences). Putting an SFT arm next to baking on the SAME
facts/probes makes the comparison apples-to-apples and tells us whether "baking's converse problem" is just
"SFT's reversal curse" wearing a distillation hat, or something the KL objective specifically does (better
or worse). It also gives a third reference point against prompting (which is directional on 8B, no training).

## Notes
- Matched-data discipline: SFT and bake must train on the SAME cached trajectories (same sampler
  base_disable_adapter) so only the loss differs. The runner enforces objective.sampler == data.sampler.
- eval_kl is still computed for SFT runs (it's objective-agnostic: KL of the SFT'd model to the prompted
  teacher) — a useful "how close did SFT land to the teacher" readout even though it is not the SFT loss.
