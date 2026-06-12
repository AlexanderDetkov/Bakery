---
title: Does baking faithfulness/propagation increase with model scale, in a CONTROLLED size sweep?
status: open
priority: medium
created: 2026-06-06
hypothesis: Larger models bake a fact more faithfully (higher prompted-normalized retention, deeper propagation) because richer priors let the LoRA bind the fact to more existing structure. Cycle-1 saw 8B retention ≈0.71 vs 1B ≈0.42 at h0–h2, but across 4 uncontrolled axes — so this needs a controlled sweep.
acceptance_criteria: "Single model family (Llama-3.2-1B / 3B / Llama-3.1-8B), MATCHED LoRA trainable-param budget (tune rank per size, not fixed rank), matched max_new_tokens + supervised-token budget, same facts/probes/seeds. Compare prompted-NORMALIZED retention (baked_shift/prompted_shift) per hop with CIs (raw log-odds are cross-model incommensurable). DECISIVE if normalized retention rises monotonically with size and CIs separate sizes."
experiment: "bake_fact across {1B,3B,8B} one family, rank chosen for matched trainable params, fanned across GPU 0/1"
links: [[q-propagation-prompt-vs-bake]], [[propagation-bounded-by-trajectory-coverage]]
---

## Question
Cycle-1 C4 (8B bakes more faithfully than 1B) was rated descriptive-only: the two runs differed in LoRA
params (3.4M vs 13.6M), token budget (160 vs 128), supervised tokens, and tokenizer/family simultaneously,
and raw nats are not comparable across models. This question runs the controlled version.

## Plan (heavy)
- Pull Llama-3.2-3B-Instruct (1B + 8B already cached). Pick LoRA rank per size to match trainable-param
  count. Fix max_new_tokens + per-run supervised-token budget. Same fact/probe banks + seeds.
- Report prompted-normalized retention per hop with bootstrap CIs from `propagation.per_probe`.

## Notes
- If retention is flat or non-monotonic under control, C4 was a confound — itself a useful negative result.

## Related evidence (cross-ref, 2026-06-12 lint)
The 1B/3B/8B size trend in [[bake-tracks-teacher-sft-sharpens]] is partial model-scale data: BAKED propagation tracks the prompted-teacher capacity (1B baked-converse≈prior, 3B 0.35, 8B 0.89). It does NOT fully answer this question (single chain/seed, bake-vs-SFT framing rather than a controlled matched-LoRA prompted-vs-baked retention sweep), so this stays OPEN — but design the sweep to extend that trend.
