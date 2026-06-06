---
title: Does chain-of-thought inflate apparent propagation, and does baking change the no-CoT vs CoT gap?
status: open
priority: medium
created: 2026-06-06
hypothesis: With CoT allowed, a model can reach an n-hop answer by chaining one-hop steps aloud, inflating "apparent" propagation regardless of whether the fact is internalized. The no-CoT forced-choice belief readout measures INTERNAL propagation only. Prediction: (a) CoT raises high-hop accuracy for BOTH prompted and baked; (b) the no-CoT gap between prompted and baked is the true measure of internalization depth; (c) baking may narrow the no-CoT/CoT gap relative to prompting if the LoRA compresses the reasoning into the forward pass.
acceptance_criteria: "Add a CoT-allowed variant of the propagation readout (generate a short rationale, then read the forced-choice answer) and compare to the no-CoT readout used elsewhere. DECISIVE if CoT raises high-hop belief shift markedly while no-CoT does not — confirming the readout cleanly separates internal propagation from CoT chaining."
experiment: "bake_fact with a `propagation_cot` metric variant (allow_cot=true); compare to no-CoT propagation on the same probes"
links: [[q-propagation-prompt-vs-bake]]
---

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
