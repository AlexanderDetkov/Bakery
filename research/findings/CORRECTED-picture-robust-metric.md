---
title: CORRECTED picture (tokenization-robust, generation-validated) — baking propagates FORWARD well; converse failure is chain-specific; prompting is directional; 1B capacity-gated
outcome: positive          # the definitive corrected result; supersedes converse/yes-saturation claims from the artifactual metric
confidence: high           # robust belief matches GENERATED answers; but n=5 converse, ~1 seed/chain, 2 chains
created: 2026-06-06
question: [[q-propagation-prompt-vs-bake]]
metric: propagation
run_ids: [prop-veld-8b-mixed, prop-tellus-8b-mixed, prop-veld-1b-mixed]
---

## Insight
Re-scoring the existing adapters under the tokenization-robust metric (validated to match the model's GENERATED
answers, [[tokenization-artifact-corrected-prompting-is-directional]]) gives the definitive corrected picture,
which REVISES the earlier "baking is associative / direction-blind / yes-saturated" headline (that rested on the
artifactual space-only metric): (1) baking reliably propagates the FORWARD entailments (large, fact-general
gain over prior); (2) baking's CONVERSE failure is CHAIN-SPECIFIC (bad on Veld, fine on Tellus), not uniform
direction-blindness; (3) prompting is fully directional on 8B; (4) the 1B cannot do single-pass propagation even
when prompted (capacity-gated, defaults to "No").

## Evidence (tokenization-robust no-CoT accuracy; correct = favors the logically-correct answer)
| run | forward-correct (prior→prompted→baked) | converse-reject (prior→prompted→baked) | overall |
|---|---|---|---|
| Veld 8B   | 0.07 → 0.93 → **0.93** | 1.00 → 0.80 → **0.00** | 0.30 → 0.90 → 0.70 |
| Tellus 8B | 0.00 → 0.93 → **0.87** | 0.80 → 0.80 → **0.80** | 0.13 → 0.80 → 0.73 |
| Veld 1B   | 0.07 → **0.07** → 0.00 | 1.00 → 1.00 → 1.00 | 0.50 → 0.53 → 0.50 |
- **Forward propagation by baking is strong & fact-general:** prior 0.00–0.07 → baked 0.87–0.93 on both chains.
  Baking DOES internalize the forward consequences (single-pass, no CoT).
- **Converse handling is chain-specific:** Veld baked 0.00 (over-affirms, fails) vs Tellus baked 0.80 (rejects,
  matches prompting). So baking is NOT uniformly direction-blind; it depends on the chain.
- **Prompting is directional on 8B:** forward 0.93 AND converse 0.80 on both chains (overall 0.80–0.90).
- **1B capacity gate:** even PROMPTED, forward-correct is 0.07 (the 1B says "No" to "is a Zorv a Plonk?" no-CoT);
  it never propagates forward in a single pass (CoT rescues it — cycle 5). Its converse-reject 1.00 is just its
  global "No" bias, not discrimination (overall 0.50).

## What this SUPERSEDES / corrects
- "Single-pass YES-saturation" ([[yes-saturation-is-fact-general-converse-amplification-is-not]]) and
  "baking is associative/direction-blind" ([[baking-is-associative-prompting-is-directional]]): the UNIFORM
  yes-saturation was a metric artifact (the 1B was actually NO-biased; 8B-Tellus baked discriminates fine).
  The robust fact-general claim is the POSITIVE one: baking propagates FORWARD entailments; converse failure is
  chain-specific (Veld).
- The forward-propagation + coverage-gate + eval_kl⟂propagation results (cycles 1, 6) are qualitatively
  unaffected (they concern forward/Yes answers and SHIFTs, where the artifact is minor) — but their absolute
  magnitudes should be read under the robust metric.

## Counter-arguments / threats to validity
- n=5 converse / 15 forward per chain; ~1 seed per chain in this re-score; 2 chains → "chain-specific converse"
  is n=2 (Veld fails, Tellus fine); needs more chains/seeds to characterize WHEN baking fails the converse.
- Robust metric validated against generation on 8B Veld; assumed to transfer to Tellus/1B (same tokenizer family).
- Single-pass, behavioral; CoT changes the 1B/baked picture (cycle 5). "Capacity-gated" = single-pass only.
- Why does baking fail Veld converse but not Tellus? Unknown (entity tokenization? teacher generative bias differs
  per chain — cf. [[contrastive-trajectories-do-not-fix-the-converse]]). Open.

## Implications — the corrected answer to the user's question
**Prompting** carries the fact's directional logic into a single forward pass (forward + converse correct, 8B).
**Baking** reliably propagates the fact's FORWARD consequences (big gain, fact-general) — so "the theorem's
consequences DO emerge internally" for forward entailments — but its discrimination on NON-entailed (converse)
questions is unreliable and chain-dependent (can collapse to over-affirmation, as on Veld). **Internal
single-pass propagation is capacity-gated** (1B fails even prompted; needs CoT). And **eval_kl does not certify
any of this** (cycle 1). The earlier "direction-blind/associative" framing was too strong — corrected to
"reliable forward propagation + chain-dependent converse reliability."

## Next steps
- Re-score the remaining conditions (size sweep, type sweep, seeds, contrastive/directional) under the robust
  metric to refresh all magnitudes; recompute "yes-saturation" as a per-model response-bias (8B→Yes, 1B→No).
- Characterize WHEN baking fails the converse (more chains; correlate with teacher generative converse-affirmation).
