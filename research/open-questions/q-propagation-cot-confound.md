---
title: Does chain-of-thought inflate apparent propagation, and does baking change the no-CoT vs CoT gap?
status: active
priority: high          # promoted: the definitive internal-vs-chaining test on the Veld chain (cycle-3 follow-up)
created: 2026-06-06
hypothesis: With CoT allowed, a model can reach an n-hop answer by chaining one-hop steps aloud, inflating "apparent" propagation regardless of whether the fact is internalized. The no-CoT forced-choice belief readout measures INTERNAL propagation only. Prediction: (a) CoT raises high-hop accuracy for BOTH prompted and baked; (b) the no-CoT gap between prompted and baked is the true measure of internalization depth; (c) baking may narrow the no-CoT/CoT gap relative to prompting if the LoRA compresses the reasoning into the forward pass.
acceptance_criteria: "Add a CoT-allowed variant of the propagation readout (generate a short rationale, then read the forced-choice answer) and compare to the no-CoT readout used elsewhere. DECISIVE if CoT raises high-hop belief shift markedly while no-CoT does not — confirming the readout cleanly separates internal propagation from CoT chaining."
experiment: "bake_fact with a `propagation_cot` metric variant (allow_cot=true); compare to no-CoT propagation on the same probes"
links: [[q-propagation-prompt-vs-bake]], [[baking-is-associative-prompting-is-directional]], [[q-propagation-deductive-chain]]
---

## CYCLE-4 UPDATE — first CoT readout was ARTIFACTUAL; redesign required (still active)
Cycle 4 ([[cot-cue-scoring-is-artifactual]]) built a CoT readout (greedy rationale → "Final answer:" cue →
score 1 token). A cue-only ablation proved the cue ALONE causes the apparent deflation (rule-verbatim probe
+14.3 → +2.5 with zero reasoning), so the CoT-vs-no-CoT comparison is inconclusive. CORRECTED DESIGN for the
re-run: free-generate the answer and PARSE Yes/No from the text (not a forced cue token), with k-sample
self-consistency; keep the cue-only run as the bias baseline; rationales are now persisted. Code:
`bakery/eval/cot_probe.py` (now supports `--max_cot_tokens 0`). Then re-ask the questions below.

## CYCLE-3 UPDATE — now the highest-value next experiment
Cycle 3 ([[baking-is-associative-prompting-is-directional]]) found NO-CoT that baking installs an undirected
chain (affirms the converse) and the 1B barely propagates even prompted. The decisive disambiguation: re-run
the SAME Veld probes WITH a short rationale allowed before the forced choice. Predictions to test:
(a) does CoT let the BAKED 8B recover the converse / deep entailments it fails no-CoT (i.e. is the baked trace
usable-with-reasoning but not internalized)? (b) does CoT rescue the 1B's failed no-CoT propagation (capacity
is about single-pass, not knowledge)? Build a `propagation_cot` metric variant (generate bounded rationale per
state, then score pos/neg). Use the Veld chain (clean priors) as the primary substrate.

## Question
The user's key subtlety: CoT lets a model fake propagation by reasoning out loud, which tests reasoning,
not fact internalization. We need to confirm our no-CoT forced-choice readout actually isolates internal
propagation, and to quantify how much CoT inflates the apparent depth — separately for prompting vs baking.

## Plan
- Reuse the probe bank. Add a metric variant that, before the forced-choice readout, lets the model emit
  a brief rationale (bounded tokens) under each state, then scores the same pos/neg answer.
- Compare CoT vs no-CoT belief shift per hop, per state. Defer until the no-CoT instrument is validated.

## Notes
- This is the control that legitimizes the whole program: if no-CoT and CoT track identically, our readout
  isn't isolating internal propagation and the design needs revisiting.
