---
title: Contrastive trajectories do NOT fix baking's converse-blindness — because the teacher itself generates converse-affirmation
outcome: negative           # the intervention failed; but it revealed WHY (teacher generative bias) — a valuable mechanistic result
confidence: medium-high     # clean 0.00 rejection + fracYes 1.0 in both conditions; teacher-bias shown by generation samples (n=2, qualitative)
created: 2026-06-06
question: [[q-fix-converse-via-contrastive-trajectories]]
metric: propagation
run_ids: [prop-veldC-8b, prop-veldM-8b, prop-veldC-1b, prop-veldM-1b]
---

## Insight
Designing trajectories that explicitly exercise directionality (CONTRASTIVE contexts eliciting "X is Y but
NOT every Y is X") does NOT repair baking's converse failure: the baked model still saturates to "Yes"
(fracYes 1.0) and never rejects the converse (correct-rejection rate 0.00), same as the matched mixed
baseline. The reason — confirmed by generation — is that the base+u TEACHER itself often generates
converse-AFFIRMING text, so the contrastive trajectories don't cleanly carry converse-rejection to distill.
Baking faithfully inherits the teacher's GENERATIVE directional bias, which is worse than its forced-choice
probe answer suggested.

## Evidence
Matched count (12 ctx, 8 traj/ctx, 30 ep), Veld chain, contrastive (C) vs mixed (M):
- 8B converse baked belief: C −5.14 vs M −6.54 (both ≪ 0 = affirm the false converse); correct-rejection
  rate C 0.00 / M 0.00; yes-saturation fracYes C 1.00 / M 1.00; forward shift C +5.25 / M +4.77.
- 1B: converse C −1.69 / M −1.16; rejection 0.00/0.00; fracYes 1.00/1.00.
- So contrastive gives only a SMALL nudge (8B −6.54→−5.14) — real but swamped; it does not flip any converse probe.
- **Confound check (the key):** generating from base+u on the contrastive prompts, the teacher is INCONSISTENT
  — on "do the rules work both ways?" it answers *"The rules work in both directions… every Zorv is a Plonk,
  so if something is a Plonk it must be a Zorv"* (WRONG, affirms converse); on "a student assumes every Plonk
  is a Zorv" it correctly says *"there is no rule that states every Plonk is a Zorv."* So the trajectories carry
  a MIX dominated by affirmation → baking distills affirmation. The test isn't "contrastive prompts don't help";
  it's "the teacher doesn't reliably generate converse-rejection, so there's little clean signal to bake."

## Counter-arguments / threats to validity
- The teacher-bias evidence is 2 greedy generations (qualitative existence proof, not a rate). A proper version
  would score, over many sampled trajectories, the fraction containing converse-rejection vs -affirmation.
- A STRONGER intervention (e.g. u itself stating "these rules are one-directional; the converse is false", or
  filtering trajectories to only converse-rejecting ones) was not tried — so "not fixable by trajectory design"
  is too strong; the supported claim is "not fixed by contrastive ELICITATION alone, because the teacher pollutes it."
- prop-veldM-1b under-converged (eval_kl 0.141); 8B is the clean comparison. Single chain, seed 0, 5 converse probes.

## Implications
Refines the coverage principle ([[propagation-bounded-by-trajectory-coverage]]): baking is bounded not by what
the CONTEXTS invite but by what the TEACHER actually GENERATES. Since the base+u model affirms converses in
free generation (more than its forced-choice probe answer −1.88 implies), distillation inherits that — so
[[yes-saturation-is-fact-general-converse-amplification-is-not]] is partly the teacher's generative yes/affirm
bias, faithfully copied. Prescription for cleaner directional baking: fix the TEACHER's generative behavior
(stronger/explicit u, or trajectory filtering/curation to converse-rejecting samples), not just the prompts
that elicit trajectories. This is a sharper statement of "baking copies the prompted model's BEHAVIOR, not its
latent logic."

## Next steps
- Score the fraction of (sampled) trajectories that contain converse-rejection vs -affirmation, per condition.
- Try a STRONGER teacher: u that explicitly states one-directionality, OR filter trajectories to converse-
  rejecting ones, then bake — does the baked converse finally flip? That isolates "teacher signal" from "LoRA limit".
