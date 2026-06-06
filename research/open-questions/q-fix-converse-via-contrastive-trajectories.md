---
title: Is baking's direction-blindness (converse-affirmation) FIXABLE by trajectories that exercise directionality?
status: resolved        # NO — contrastive elicitation alone doesn't fix it (teacher pollutes); stronger interventions → q-fix-converse-stronger
priority: high
links_finding: [[contrastive-trajectories-do-not-fix-the-converse]]
created: 2026-06-06
hypothesis: Per the coverage principle ([[propagation-bounded-by-trajectory-coverage]]), baking only installs what the trajectories exercise. Standard on-topic trajectories state the forward rules ("a Zorv is a Wexil") but rarely the directional caveat ("not every Wexil is a Zorv"), so baking yields yes-saturation + converse-affirmation. PREDICTION: baking over CONTRASTIVE trajectories (which explicitly exercise that the rules are one-way) installs converse-REJECTION — i.e. the converse failure is a fixable coverage gap, not a fundamental limit of baking.
acceptance_criteria: "At matched count (12 ctx), compare baked CONVERSE belief for context_category=contrastive vs mixed (baseline), on 8B (+1B). DECISIVE if contrastive baking moves the baked converse belief UP (toward correct rejection, less negative / positive) materially vs mixed, while keeping forward entailments correct. If contrastive ≈ mixed on the converse, direction-blindness is more fundamental."
experiment: "bake_fact --data.context_bank data/contexts/veld_contrastive_contexts.json vs veld_contexts.json, mixed, 12 ctx, 8B+1B"
links: [[yes-saturation-is-fact-general-converse-amplification-is-not]], [[baking-is-associative-prompting-is-directional]], [[propagation-bounded-by-trajectory-coverage]]
---

## Question
The most actionable follow-up to the yes-saturation finding: can we DESIGN trajectories that fix baking's
direction-blindness? If contrastive trajectories (eliciting "X is Y but not every Y is X") let baking install
converse-rejection, then yes-saturation is a coverage artifact (fixable), and the prescription for knowledge
baking becomes "cover the directional caveats in the trajectories." If not, direction-blindness is intrinsic
to the KL-distillation-into-LoRA objective.

## Plan
- New `contrastive` context bank (directional-elicitation prompts; disjoint from probes). Bake at matched
  count (12 ctx) for contrastive vs mixed, 8B + 1B. Compare baked converse belief + forward belief.
