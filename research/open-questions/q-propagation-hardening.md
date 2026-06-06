---
title: Is the "prompted−baked gap grows with hop distance" pattern real, with CIs, across facts and seeds?
status: open
priority: high
created: 2026-06-06
hypothesis: The cycle-1 C1 pattern (on 8B, baked reproduces prompting at near hops but the fidelity gap grows with hop distance and baked collapses at the far hop) is a genuine propagation-distance effect, not an artifact of far-hop prior saturation, a collapsing teacher signal, single-seed noise, or the one tsunami fact.
acceptance_criteria: "With ≥12 polarity-balanced probes/hop, ≥2 additional facts beyond tsunami, ≥3 seeds, report fidelity_h{n} and prompted-normalized retention with bootstrap 95% CIs (now computable from the per-probe beliefs the metric logs). DECISIVE if the monotone fidelity decay + far-hop collapse holds with CIs excluding 0 across seeds AND facts, AND survives a far-hop headroom control (probes whose prior is not saturated). Otherwise C1 is demoted to a single-run curiosity."
experiment: "bake_fact on 8B (and 1B for contrast), new facts + expanded probe banks, 3 seeds; analyze per-probe arrays in metrics.json for CIs"
links: [[q-propagation-prompt-vs-bake]], [[propagation-bounded-by-trajectory-coverage]]
---

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
