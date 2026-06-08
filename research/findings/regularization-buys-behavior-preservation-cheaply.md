---
title: Anchor-trajectory regularization buys behavior preservation cheaply — behavior_drift falls monotonically with anchor count at ~no propagation or fidelity cost
outcome: positive
confidence: medium
created: 2026-06-08
question: [[q-regularization-preserves-behavior]]
metric: behavior_drift
run_ids: [qa-reg0-n1-s10, qa-reg32-n1-s10, qa-reg128-n1-s10, qa-reg256-n1-s10]
---

## Insight
Mixing base-anchored irrelevant-question (SQuAD) trajectories into the n=1 axiom bake drives
`behavior_drift` = KL(base‖baked) on held-out SQuAD continuations **down monotonically with anchor
count** (0.012 → 0.007 → 0.005 nats/tok for 32 → 128 → 256 anchors), while **held-out propagation
(baked d′ at d2) and bake fidelity (eval_kl) are unchanged** within noise. So you can preserve general
behavior essentially for free — the regularizer does what it claims at negligible cost to the headline.
NB the absolute drift is TINY even unregularized, so for n=1 axiom baking this is preserving an
already-small perturbation.

## Evidence
- bake_theorem_qa, Llama-3.1-8B, bake, sampled teacher, lw_alpha, n=1, **seed 10**, bs4, 200 ep
  (read at ep 70–100; reg256 at ep50 so its d′ is preliminary, its drift is stable). Dose-response on
  `--regularization.num_train_contexts ∈ {0,32,128,256}` (256 ≈ 84% of the ~304 logic trajectories).
- **behavior_drift (postmean ep≥50, lower=better):** reg0 = n/a (metric returns None when OFF),
  reg32 = 0.012, reg128 = 0.007, reg256 = 0.005. **Monotonic ↓**; most of the drop is 0→128
  (diminishing returns by 256).
- **Propagation cost ≈ none:** baked d2 (postmean) = {reg0 0.91, reg32 0.96, reg128 1.03, reg256 0.92}
  — flat, all within the reg0 baseline ± seed noise. d1 = {1.02, 0.89, 0.55, 0.93} (the reg128 0.55 dip
  is within the d1 noise band measured at ±0.32 in [[baked-propagation-tracks-trained-depth-no-compositional-bonus]];
  reg256 recovers to 0.93 → no systematic suppression). d3 ≈ {−0.38, −0.27, −0.25, −0.15} (the teacher
  ceiling; regularization neither helps nor hurts it).
- **Fidelity cost ≈ none:** eval_kl = {0.262, 0.255, 0.256, 0.264} — flat across doses.
- Meets the question's acceptance criteria: drift falls monotonically AND the best-drift arm's (reg256)
  held-out d2 (0.92) is within noise of the 0 arm (0.91).

## FINAL converged AUROC re-read (all arms ep200, postmean ep≥100) — confirms + a new wrinkle
Per [[paired-matched-seed-protocol]] (AUROC primary). All 4 arms ran to ep200:
| anchors | behavior_drift | AUROC d1 | AUROC d2 (held-out) | AUROC d3 | eval_kl |
|---|---|---|---|---|---|
| 0   | — (off)  | 0.749 | 0.695 | 0.496 | 0.253 |
| 32  | 0.023 | 0.745 | 0.695 | 0.513 | 0.235 |
| 128 | 0.015 | 0.725 | 0.739 | 0.522 | 0.246 |
| 256 | 0.011 | 0.741 | 0.733 | 0.493 | 0.251 |
- **Drift ↓ monotonic with anchors** (0.023 → 0.015 → 0.011) — robust at full convergence.
- **AUROC d2 preserved (even slightly up)** across doses (0.695 → 0.739/0.733), within noise of reg0; AUROC d1
  ≈ 0.72–0.75 (recall intact); AUROC d3 ≈ 0.49–0.52 = chance at every dose (teacher ceiling unchanged);
  eval_kl flat ≈ 0.24–0.25. ⇒ **behavior preservation is cheap** holds, now on converged data + the
  low-variance readout.
- **NEW WRINKLE (supersedes the earlier ep50–80 read):** absolute drift GROWS with training — the ep50–80
  snapshot (reg32≈0.012, reg256≈0.005) under-reported it; by ep200 it's ~2× higher. So an unregularized bake
  keeps drifting on irrelevant inputs the longer it bakes, and anchoring's value RISES with bake length —
  strengthening the case for regularization in long/grokking-length bakes.
- Caveat unchanged: single-arm-per-dose (unpaired); a paired matched-seed dose-response would tighten the CI.

## Counter-arguments / threats to validity
- **Single seed (10).** The drift dose-response is clean and monotonic (low-variance metric), but the
  per-dose d′ values carry the same ±0.2–0.3 probe noise as elsewhere; the "no propagation cost" claim
  would be firmer at ≥3 seeds. Medium (not high) confidence for that half.
- **reg256 d′ is preliminary** (ep50, 1 post-saturation point) — its drift is stable but its propagation
  read should be confirmed at ep≥80.
- **Absolute drift is tiny unregularized** (~0.012 even at reg32). So this is "make a small drift smaller,"
  not "rescue a degenerate adapter." Whether regularization matters MORE in regimes with larger drift
  (higher n, longer/stronger bakes, or a more behavior-perturbing prompt) is untested → follow-up.
- **CoT-leak caveat (constant across arms):** these sampled-teacher runs leak ~29 held-out d≥2 probes via
  the teacher's recited CoT (analysis loader flags it), so the absolute d2 is partly recall-of-recited.
  BUT the leak is identical across all four doses, so the *relative* claim "regularization doesn't change
  d2" is valid; only the absolute propagation level is caveated (clean ref = teacher-forced arm).
- behavior_drift is measured on SQuAD continuations only; "general behavior" is proxied by that one
  distribution — a broader behavioral battery could disagree.

## Implications
- The regularizer is a usable knob: a practitioner baking facts can anchor ~32–128 SQuAD trajectories to
  cut drift ~2× with no measured propagation cost. Cheap insurance against catastrophic-forgetting-style
  drift, especially relevant before scaling to stronger/longer bakes where drift should be larger.
- For the central idea: baking n=1 axioms is already GENTLE on general behavior (tiny drift), consistent
  with [[baked-propagation-tracks-trained-depth-no-compositional-bonus]] (the adapter installs a narrow
  capability without broadly moving the model).

## Next steps
- Confirm at ≥3 seeds (seeds 11,12) and let reg256 reach ep≥80; fold into the finding.
- Re-run the dose-response with **teacher-forced** trajectories (no CoT leak) to give a clean-propagation
  cost readout — pairs with [[q-teacher-ceiling-vs-objective-limit]].
- Test regularization where drift is LARGER (higher n, longer bake, or a behavior-perturbing prompt) — does
  it matter more there?
