---
title: Can a STRONGER intervention (explicit-directional u, or trajectory filtering) install converse-rejection via baking?
status: open
priority: high
created: 2026-06-06
hypothesis: Contrastive elicitation failed because the teacher (base+u) itself generates converse-affirmation, so trajectories carry little clean rejection signal ([[contrastive-trajectories-do-not-fix-the-converse]]). A stronger teacher signal — u that EXPLICITLY states one-directionality ("the converse is false"), or FILTERING sampled trajectories to only converse-rejecting ones before baking — should give the adapter a clean rejection signal. If the baked converse STILL doesn't flip, direction-blindness is a LoRA/objective limit, not a teacher-signal limit.
acceptance_criteria: "Two arms: (a) u' = veld chain + explicit 'these rules are one-directional; the converse does NOT hold'; (b) filter sampled trajectories to those containing converse-rejection, then bake. Measure baked converse correct-rejection rate + fracYes vs the cycle-3 baselines. DECISIVE if either arm raises baked converse rejection materially above 0 — isolating 'teacher signal' from 'LoRA limit'."
experiment: "bake_fact with u'=explicit-directional prompt; and/or a trajectory-filtering builder hook; Veld, 8B+1B"
links: [[contrastive-trajectories-do-not-fix-the-converse]], [[yes-saturation-is-fact-general-converse-amplification-is-not]]
---

## Question
The decisive disambiguation of WHY baking is direction-blind: is it because the teacher doesn't generate clean
rejection (a fixable signal problem) or because a low-rank additive adapter trained to match an affirming
distribution cannot represent input-conditional suppression (a fundamental limit)? The two arms separate these.

## Plan
- Arm (a): write u' adding an explicit one-directionality clause; re-bake; cheapest test of "stronger teacher".
- Arm (b): a trajectory-filter (keep only sampled continuations whose text rejects a converse) — needs a small
  builder hook or a post-generation filter; then bake on the filtered set.
- Score baked converse rejection + fracYes; compare to cycle-3 contrastive/mixed.
