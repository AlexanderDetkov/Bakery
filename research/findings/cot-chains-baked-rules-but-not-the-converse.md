---
title: Single-pass baking saturates a yes-bias (affirms everything); CoT forward-chains the installed rules but reasons over hallucinated REVERSED rules on the converse
outcome: positive          # robust core (yes-saturation; CoT forward-chaining via rationale evidence); converse-CoT direction suggestive
confidence: medium         # converse n=5, parse rate ~52%, single chain/seed; key qualitative claim rests on reading rationales
created: 2026-06-06
question: [[q-propagation-cot-confound]]
metric: propagation
run_ids: [prop-veld-8b-mixed, prop-veld-1b-mixed]
---

## Insight
In a SINGLE forward pass, baking the Veld rule chain collapses the model toward answering "Yes" to every
chain-related question (it amplifies a pre-existing yes-bias to saturation), so it installs undiscriminated
AFFIRMATION rather than directional discrimination. With chain-of-thought, the baked model genuinely recites
and FORWARD-chains its installed rules (Zorv→Plonk→…→venomous ⟹ Yes), so the consequences ARE
reasoning-accessible — but on the CONVERSE it reasons over HALLUCINATED reversed rules ("every Wexil is a
Zorv") and stays wrong. So CoT answers the user's worry directly: it reaches n-hop answers by chaining, and
the baked model can do this forward, but baking's undirected corruption poisons even its reasoning backward.

## Method
Corrected CoT readout (`bakery/eval/cot_probe.py --mode freegen`): free-generate k=5 sampled rationales,
PARSE Yes/No from the text, majority-vote (self-consistency) — no answer-cue artifact (cf.
[[cot-cue-scoring-is-artifactual]]). Reused the cycle-3 Veld adapters (no re-bake). Accuracy = the model
favours the logically-correct answer; we report it ALONGSIDE the yes-bias (fraction favouring the literal
"Yes" token) because on a polarity-balanced set those two together expose degenerate all-Yes behaviour that
bare accuracy hides.

## Evidence (no-CoT = single-pass belief sign; CoT = parsed majority of 5 samples)
| state | no-CoT acc | yes-bias | CoT acc | converse no-CoT / CoT |
|---|---|---|---|---|
| 8B prompted | 0.77 | 0.73 | 0.89 | 0.20 / 0.50 |
| **8B baked** | **0.50** | **1.00** | 0.76 | 0.00 / **0.00** |
| 1B prompted | 0.50 | 1.00 | 0.80 | 0.00 / 0.60 |
| 1B baked | 0.50 | 1.00 | 0.72 | 0.00 / 0.40 |
- **Yes-saturation:** 8B/1B baked answer "Yes" to ALL 30 probes (fracYes=1.00) → 0.50 acc = right on the 15
  forward entailments, wrong on the 15 reverse. The base prior already leans Yes (8B 0.90); baking maxes it.
  8B PROMPTING keeps discrimination (fracYes 0.73, acc 0.77); the 1B collapses to all-Yes even prompted (capacity).
- **CoT > no-CoT for baked**, robust to the parse penalty: 8B baked CoT 0.76 (0.73 if every unparsed=wrong) vs
  no-CoT 0.50; 1B baked 0.72 (0.70). Reading the rationales: 1B baked explicitly chains the installed rules
  ("Since a Zorv is a Plonk (rule 1)… Wexils are venomous (rule 4)… a Zorv is venomous. Final answer: Yes") —
  genuine multi-hop deduction over rules it recites from its OWN weights (the no-fact prior confabulates instead).
- **Converse stays wrong under CoT** (baked 8B 0.00 / 1B 0.40 vs prompted 0.50/0.60, prior 1.00): the baked
  rationales reason over HALLUCINATED reversed rules ("Every Wexil is a Zorv (rule 3) → every Plonk is also a
  Zorv. Yes"). Baking's undirected trace propagates into its reasoning.

## Counter-arguments / threats to validity
- **The adversarial panel itself erred here** — TWO lenses scored the baked converse-affirmation as "correct"
  (a sign flip), concluding "baked no-CoT = 1.00". Settled by per-probe inspection: e.g. "is every Marn a
  Zorv?" has noCoT_belief = −7.56 (favours Yes = WRONG). The panel's *rationale reading* (independent of the
  sign error) is what supports the CoT-chaining claims. Lesson: verify the verifier; print per-probe.
- **No-CoT forward "correctness" is confounded with the yes-bias** (all-Yes is right on forward by default), so
  the no-CoT data alone cannot prove baking "installs the forward chain"; the CoT rationale evidence is what does.
- **Small n / low parse:** converse n=5 (some CoT majorities rest on 1–2 parsed samples; converse-ALL pos=No
  n≈14 gives baked CoT ≈0.64 — materially softer than the converse-5 0.00). Sample parse rate ~52% (8B)/55%
  (1B). Single fictional chain, single seed; 8B bake under-converged (cycle-4). Treat magnitudes as suggestive.
- belief (single-pass logit diff) and CoT-majority (multi-pass text vote, temp 0.7) are different scales; the
  comparison is accuracy-vs-accuracy, not belief-vs-belief.

## Implications
Resolves the cycle-4 CoT question with a corrected readout and answers the user's central concern: CoT does
reach multi-hop answers by explicit chaining, and the baked model CAN forward-chain its installed rules — so
"the theorem's consequences emerge" under reasoning. But two failure modes distinguish baking from prompting:
(1) single-pass, baking yields undiscriminated affirmation (all-Yes), not internalized direction-sensitive
belief; (2) baking corrupts the converse so deeply that CoT hallucinates reversed rules. Combined with
[[baking-is-associative-prompting-is-directional]]: baking transfers an associative, direction-blind shadow of
the prompted model's knowledge. Methodological: always report accuracy WITH the yes-bias and split
forward/converse/negated; and print per-probe before trusting any aggregate (or any verifier).

## Next steps
- Raise parse rate (force a final "Answer: <Yes/No>" token via constrained/long generation), add ≥10
  converse probes, ≥3 seeds and ≥2 more chains, to firm the converse-CoT degradation and the all-Yes claim.
- Disentangle baking's yes-saturation from the base prior's yes-bias (subtract per-probe prior; report the
  ADDED affirmation on reverse probes specifically).
- Tie back to the trajectory-TYPE lever ([[q-propagation-trajectory-type]]): does consequence-rich vs
  restate-only baking change the all-Yes saturation / converse corruption?
