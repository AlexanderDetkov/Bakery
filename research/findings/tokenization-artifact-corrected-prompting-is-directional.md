---
title: A tokenization artifact hid that PROMPTING rejects the converse; corrected, the prompting-vs-baking contrast is CLEANER (prompting directional, baking direction-blind)
outcome: positive          # sharpens the central thesis + fixes an instrument bug; high confidence in the correction
confidence: high           # the robust belief matches GENERATED answers exactly on the held-out probes
created: 2026-06-06
question: [[q-fix-converse-stronger]]
metric: propagation
run_ids: [prop-veld-8b-mixed, prop-veldD-8b, prop-veldD-1b]
---

## Insight
The no-CoT forced-choice BELIEF metric scored the space-prefixed tokens " Yes"/" No" (ids 7566/2360), but
chat models emit the FIRST assistant token WITHOUT a leading space — "Yes"/"No" are DIFFERENT ids (9642/2822).
This systematically under-credited "No", inflating apparent yes-affirmation. Correcting it (logsumexp over the
space/no-space variants) makes belief match the model's actual GENERATED answer — and reveals that PROMPTING
correctly REJECTS the converse (it is directional), while BAKING affirms it (direction-blind). The central
thesis holds and is cleaner; the earlier "prompting also fails the converse / partly yes-saturates" sub-claims
were the artifact.

## Evidence (8B, 5 Veld converse probes; correct answer = reject = "No")
Comparing belief-sign to the model's greedy GENERATED first token:
- **PROMPTED (base+u): generates "No" on 4/5** (correct rejection). The OLD belief metric read "affirms" on 3
  of those (it scored " No" id 2360, but the model's mass is on "No" id 2822). Tokenization-robust belief →
  4/5 correct, matching generation.
- **BAKED (adapter): generates "Yes" on 5/5** (affirms converse). Old & robust belief & generation ALL agree:
  baked genuinely affirms (0/5). 
- FORWARD entailments: robust belief 5/5 correct for BOTH prompted and baked (forward was never affected — the
  sign was right when the model favors Yes).
- Robust-belief correct-rejection: PROMPTED 4/5, BAKED 0/5 (matches generated answers exactly). Token ids
  confirmed distinct: " Yes"=7566 vs "Yes"=9642; " No"=2360 vs "No"=2822.

## What this CORRECTS in prior findings
- [[cot-chains-baked-rules-but-not-the-converse]] / [[baking-is-associative-prompting-is-directional]]: the
  claim that the PROMPTED model also affirms the converse no-CoT (−1.88) was an ARTIFACT; prompting REJECTS it
  (4/5 generated). So the prompting-vs-baking directional contrast is STRONGER/cleaner than recorded.
- [[yes-saturation-is-fact-general-converse-amplification-is-not]]: "single-pass yes-saturation" is
  BAKING-SPECIFIC (baked generates Yes to everything, 0/5 on converse — real), NOT a property of the prompted
  model (which discriminates). Prompted "fracYes" numbers were inflated by the artifact.
- BAKED results are unchanged (baked is so Yes-dominant that even the space-only token won → belief, generation,
  and robust belief all agree). FORWARD-entailment results unchanged (sign correct).
- Vindicates the cycle-5 free-generation readout ([[cot-chains-baked-rules-but-not-the-converse]] used parsed
  generated answers) — that methodology was already artifact-free; the belief metric was the weak link.

## Fix applied
`bakery/eval/metrics/propagation.py`: new `_answer_logprob` aggregates leading-space/no-space answer variants
(logsumexp); `_belief_per_probe` and `cot_probe._noCoT_belief` now use it. Tests green (the fake tokenizer
dedupes variants, so the unit test is unchanged). FUTURE runs are artifact-free; PRIOR belief-based magnitudes
(esp. converse / yes-saturation for prompted & prior) should be re-read under the fixed metric — the QUALITATIVE
corrections above are already established by the generated-answer comparison.

## Counter-arguments / threats to validity
- Demonstrated on 8B, 5 converse probes, plain u (and u'); a full re-measurement of all conditions/chains under
  the fixed metric is a follow-up (the generated-answer evidence already fixes the qualitative picture).
- The artifact's exact bite depends on the chat template's first-token spacing (Llama-3 here); other tokenizers
  may differ — but the robust scorer is template-agnostic.
- Also note (from this cycle) the explicit-directional u' did NOT fix the BAKED converse (still 0/5 generated,
  fracYes 1.0) even though base+u' rejects it — so baking's direction-blindness is a real LoRA/distillation
  limit, not just the metric: the adapter copies the teacher's strongly-affirming single-pass GENERATION on
  chain probes regardless of the prompt's explicit directionality.

## Implications
The corrected, sharpened thesis: **prompting carries the fact's directional logic into a single forward pass
(rejects invalid converses); baking installs a direction-blind, yes-saturated trace (affirms them).** And it is
NOT fixed by explicit-directional prompting of the teacher (u') — a genuine limit of distilling single-pass
behavior into a low-rank adapter. Methodological: ALWAYS validate a forced-choice belief metric against the
model's generated answer; prefer generated-answer parsing or tokenization-robust scoring.

## Next steps
- Re-run the key conditions (Veld/Tellus prior/prompted/baked) under the fixed metric to refresh the recorded
  magnitudes (converse rejection rate + a corrected, deflated "yes-saturation" for prompted/prior).
- Arm (b) of [[q-fix-converse-stronger]] — trajectory filtering to converse-rejecting samples — to fully pin
  "LoRA limit" (the u' result already strongly suggests it).
