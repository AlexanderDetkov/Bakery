---
title: Across 2 chains, single-pass YES-SATURATION is the fact-general baking effect; the dramatic converse-amplification was chain-specific
outcome: positive          # confirms the mechanism, qualifies the headline number — both valuable
confidence: medium-high    # 2 chains × seeds, consistent on the mechanism; n=2 chains still small
created: 2026-06-06
question: [[q-propagation-hardening]]
metric: propagation
run_ids: [prop-tellus-8b-mixed, prop-tellus-1b-mixed, prop-tellus-1b-mixed-s1, prop-veld-8b-mixed, prop-veld-1b-mixed]
---

## Insight
Replicating the Veld result on a second, structurally-identical synthetic chain (Tellus:
Grummel→Fenn→Drask→Yorl→glows) shows the ROBUST fact-general effect of baking is **single-pass
yes-saturation** — the baked model answers "Yes" to ~every chain-related probe (fracYes 0.97–1.00 on both
chains, both seeds) — which mechanically yields forward-entailments-correct + reverse(converse)-wrong. But the
*dramatic* cycle-3 finding that "baking affirms the false converse FAR more than prompting" (Veld 8B baked
converse −6.61 vs prompted −1.88) does NOT generalize: on Tellus, baking ≈ prompting on the converse
(−2.16 vs −2.61) and even the prior already strongly affirms it (−4.21). So the headline NUMBER was
chain-specific; the underlying MECHANISM (direction-blind yes-saturation) is fact-general.

## Evidence (8B; belief, correct>0; + baked fracYes = single-pass yes-rate)
| chain | fwd baked shift | fwd prompted shift | converse prior | converse prompted | converse baked | baked fracYes |
|---|---|---|---|---|---|---|
| Veld   | +4.94 | +8.35 | −1.86 | −1.88 | **−6.61** | 1.00 |
| Tellus | +2.34 | +6.16 | −4.21 | −2.61 | **−2.16** | 0.97 |
- **Fact-general (both chains, incl. 1B seeds 0/1, fracYes 1.00):** baked yes-saturation; forward shift
  positive but < prompting; baked converse belief NEGATIVE (model fails to reject the converse).
- **Chain-specific:** the SIZE/DIRECTION of the converse effect vs prompting. Veld: baking worsens converse
  far beyond prompting (−6.61 vs −1.88). Tellus: baking ≈ prompting (−2.16 vs −2.61), both modestly wrong; the
  Tellus prior is already very converse-affirming (−4.21), leaving little for baking to "amplify".

## Counter-arguments / threats to validity
- Still only n=2 chains (both transitive 4-link syllogisms, fictional). The yes-saturation generalization is
  2-for-2; "fact-general" beyond syllogistic chains (e.g. the tsunami event-fact) is not directly tested here.
- The converse difference between chains may reflect tokenization/semantics of the specific entity names (how
  "interchangeable" Drask/Grummel feel vs Marn/Zorv) — uncontrolled.
- 6 probes/depth (5 converse) per chain; per-condition means are stable across seeds but within-probe n is small.
- Behavioral readout; yes-saturation is inferred from belief-sign + fracYes, not adapter internals.

## Implications — RECAST the headline mechanism
The fact-general statement of "baking is associative / direction-blind"
([[baking-is-associative-prompting-is-directional]], [[cot-chains-baked-rules-but-not-the-converse]]) is best
phrased as **single-pass yes-saturation**: baking drives the unprompted model to affirm ~everything related to
the baked content, so forward entailments come out right and reverse/converse questions come out wrong — on
every chain tested. Converse-affirmation is a *symptom* of yes-saturation whose severity-relative-to-prompting
is chain-dependent (do NOT cite the −4.75 Veld amplification as a general number). This is a cleaner, more
unified, and better-supported version of the capstone claim — and a reminder that single-fact magnitudes need
multi-fact replication before they become headline numbers.

## Next steps
- A 3rd chain + the tsunami event-fact under the same fracYes lens, to test yes-saturation beyond syllogisms.
- Disentangle baked yes-saturation from the base prior's yes-bias per-probe (report ADDED affirmation on
  reverse probes specifically), across chains.
- Update [[SYNTHESIS-baking-vs-prompting-propagation]] and [[baking-is-associative-prompting-is-directional]]
  to lead with yes-saturation (done: cross-links added).
