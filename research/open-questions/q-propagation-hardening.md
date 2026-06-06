---
title: Is the "prompted−baked gap grows with hop distance" pattern real, with CIs, across facts and seeds?
status: active
priority: high
created: 2026-06-06
hypothesis: The cycle-1 C1 pattern (on 8B, baked reproduces prompting at near hops but the fidelity gap grows with hop distance and baked collapses at the far hop) is a genuine propagation-distance effect, not an artifact of far-hop prior saturation, a collapsing teacher signal, single-seed noise, or the one tsunami fact.
acceptance_criteria: "With ≥12 polarity-balanced probes/hop, ≥2 additional facts beyond tsunami, ≥3 seeds, report fidelity_h{n} and prompted-normalized retention with bootstrap 95% CIs (now computable from the per-probe beliefs the metric logs). DECISIVE if the monotone fidelity decay + far-hop collapse holds with CIs excluding 0 across seeds AND facts, AND survives a far-hop headroom control (probes whose prior is not saturated). Otherwise C1 is demoted to a single-run curiosity."
experiment: "bake_fact on 8B (and 1B for contrast), new facts + expanded probe banks, 3 seeds; analyze per-probe arrays in metrics.json for CIs"
links: [[q-propagation-prompt-vs-bake]], [[propagation-bounded-by-trajectory-coverage]], [[veld-findings-replicate-across-seeds]]
---

## PROGRESS (S2 cycle 1) — SEEDS done, FACTS + probe-count remain
[[veld-findings-replicate-across-seeds]]: ran the Veld chain at seeds {0,1,2} × 1B+8B. The converse-affirmation
(8B baked −7.00 ± 0.27), single-pass yes-saturation (fracYes ≈ 1.0), and forward<prompting results all
REPLICATE tightly across seeds → upgraded to seed-robust. STILL TODO for full closure: (a) ≥2 additional
fact/chain banks × 3 seeds (fact-generality), (b) ≥12 probes/depth (incl. ≥8 converse) for tight per-hop CIs,
(c) the C1 per-hop fidelity-decay claim specifically (this cycle hardened the converse/yes-sat results, not C1).

## PROGRESS (S2 cycle 2) — 2nd FACT added (Tellus chain)
[[yes-saturation-is-fact-general-converse-amplification-is-not]]: ran a 2nd syllogism chain (Tellus) ×
seeds. Result: **single-pass yes-saturation is FACT-GENERAL** (fracYes≈1.0 both chains → forward-correct,
converse-wrong), but the dramatic Veld converse-amplification (−4.75 vs prompting) is CHAIN-SPECIFIC (on
Tellus baking ≈ prompting on the converse). Recast the headline as yes-saturation. Still TODO: a 3rd fact +
the tsunami event-fact under the fracYes lens; ≥12 probes/depth; the C1 per-hop decay claim.

## Question
C1 was the most eye-catching cycle-1 result but the adversarial panel demoted it to suggestive: it reverses
on 1B, the far-hop gap is confounded by a saturated prior (8B prior_h3 +6.31) and a teacher signal that
itself collapses at h3, and there are no CIs (4 probes/hop, 1 seed, 1 fact). This question hardens or kills it.

## Plan (heavy: 8B × 3 seeds × ≥3 facts)
- Build 2–3 more fact+probe banks (e.g. a fictional scientific claim with a clean deductive consequence
  chain, and a counterfactual about a well-known entity) with ≥12 polarity-balanced probes/hop and a
  recorded prior so saturated probes can be excluded.
- Run 8B (and 1B) × 3 seeds; compute fidelity & normalized-retention curves with bootstrap CIs from the
  per-probe beliefs now logged in metrics.json (`propagation.per_probe`).
- Add the headroom control: report shift vs prior AND vs the prompted ceiling; drop/rebalance saturated probes.

## Notes
- A clean deductive/mathematical fact (per the user's "new theorem ⇒ its consequences emerge internally")
  would be the strongest test: the consequence is logically entailed, so prior is genuinely ~0 and CoT is
  the only alternative route (which the no-CoT readout blocks).
