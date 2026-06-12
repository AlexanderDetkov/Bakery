---
title: No free-lunch knee — the fidelity↔sharpness frontier (KL→CE loss mix) is a near-STEP, not a smooth tradeoff; the KL leash pins baking at the teacher until fully removed
outcome: negative          # DECISIVE-NEGATIVE on a usable knee; the tradeoff is degenerate (two clusters)
confidence: high           # 7-point alpha frontier + w=0.5 mid-point confirmed bake-like on seed-1 AND lw_beta; principled mechanism. Single interpolation family (convex loss-mix) is the remaining caveat.
created: 2026-06-12
question: [[q-fidelity-vs-sharpness-frontier]]
metric: dprime
run_ids: [qa-sbake-n1-s0, qa-mix-w025-n1-s0, qa-mix-w05-n1-s0, qa-mix-w075-n1-s0, qa-mix-w09-n1-s0, qa-ssft-n1-s0]
---

## Insight
Sweeping the `mix_bake` objective `loss = (1−w)·KL(teacher‖student) + w·CE(tokens)` from bake (w=0) to SFT
(w=1) does NOT trace a smooth eval_kl↔d′ tradeoff with a usable knee. Instead it is a **near-STEP**: for ALL
w<1 the adapter stays at the bake optimum — low eval_kl AND bake-level d′ — and only at pure CE (w=1) does it
jump to SFT's behavior (high eval_kl AND high d′) *together*. There is **no intermediate point that buys SFT's
sharpness while keeping near-bake fidelity**. Mechanism: the KL term grows fast as the student diverges from the
teacher, so even a small KL weight (0.1 at w=0.9) is a strong leash that holds the solution at the teacher;
removing it entirely (w=1) is what lets the student escape to the sharp, teacher-divergent CE optimum. So SFT's
extra sharpness is INSEPARABLE from abandoning the teacher — exactly what [[bake-tracks-teacher-sft-sharpens]]
implies ("baking's ceiling IS the teacher"): you cannot exceed the teacher's discrimination without leaving it.

## Evidence (8B lw_alpha n1-s0; held-out = depths 2–6; all 7 points final)
| w (CE weight) | eval_kl | converse d′ | fwd held-out d′ |
|---|---|---|---|
| 0.0 (bake) | 0.465 | 0.89 | 0.52 |
| 0.25 | 0.468 | 0.96 | 0.54 |
| 0.5  | 0.48  | 0.76 | 0.24 |
| 0.75 | 0.61  | 0.91 | 0.38 |
| 0.9  | 0.76  | 1.07 | 0.47 |
| 0.95 | 0.94  | 0.97 | 0.40 |
| **1.0 (sft)** | **4.43** | **1.79** | **1.21** |
- **eval_kl is a hockey stick:** ~flat (0.47→0.76) for every w≤0.9, then ~6× jump to 4.43 at w=1. A 0.1 KL
  weight keeps eval_kl within ~1.6× of bake; only zero KL weight diverges (w=0.95 = 5% KL weight → eval_kl still 0.94, d′ still bake-like 0.97).
- **d′ stays bake-like across the interior:** converse 0.76–1.07 (vs bake 0.89), forward 0.24–0.54 (vs bake 0.52)
  — NOT climbing toward SFT's 1.79 / 1.21. The sharpness is concentrated entirely at w=1.
- **Robustness (2nd seed + 2nd chain):** the w=0.5 mid-point stays bake-like in BOTH eval_kl and d′ on alpha-seed1
  (eval_kl 0.38 vs bake 0.31 vs sft 4.40; conv 0.61 vs bake 0.56 vs sft 2.15) AND on lw_beta (eval_kl 0.52 vs bake
  0.43 vs sft 4.33; conv 0.70 vs bake 0.52 vs sft 1.45) — never approaching SFT. The near-step replicates.
  Runs: qa-mix-w05-n1-s1, qa-mix-w05-beta-n1-s0.
- Fig: `results/bake_theorem_qa/_fig_frontier.png` (eval_kl + converse-d′ vs w).

## What this answers (q-fidelity-vs-sharpness-frontier)
DECISIVE-NEGATIVE: no w achieves converse d′ near SFT's while keeping eval_kl near bake's. The convex KL/CE mix
gives a degenerate two-cluster "frontier" ({bake-like ∀ w<1} ∪ {SFT at w=1}), not a Pareto curve. If you want
behavioral equivalence to the prompted teacher (low eval_kl — Bakery's objective), you are capped at the
teacher's discrimination; getting beyond it requires fully dropping the teacher (becoming SFT). Half-baking the
LOSS buys nothing here.

## Counter-arguments / threats to validity
- **Full sweep is alpha-s0 only; the mid-point is confirmed on seed-1 + lw_beta** (above). d′ is single-seed-noisy
  (the non-monotone wiggle across interior w is within that noise — the SIGNAL is "interior ≈ bake, w=1 = jump", not
  the point-to-point order). A full multi-seed sweep would harden the exact transition location.
- **One interpolation family.** This is the convex LOSS mix. KL-at-temperature (soften the teacher) or
  label-smoothed CE might interpolate differently — the eval-time adapter-scaling "half-baking"
  (`model.half_bake_alpha`, base↔bake) is yet another axis, untested here. "No knee" is specific to loss-mixing.

## Implications
The bake↔SFT distinction is not a tunable dial — it is a near-discontinuity at "is the teacher in the loss at
all." This sharpens [[bake-tracks-teacher-sft-sharpens]]: teacher-fidelity and beyond-teacher sharpness are not
on a smooth Pareto front; the KL objective's whole effect is to hold you at the teacher. To beat the teacher
while staying faithful you'd need a BETTER teacher (cf. the size trend: 8B teacher → higher bake ceiling), not a
softer objective.

## Next steps
- DONE: w=0.95 folded; w=0.5 confirmed bake-like on seed-1 + lw_beta → confidence raised to high.
- Optional: a KL-temperature sweep (soften teacher targets) as a SECOND interpolation family — does softening the
  teacher (rather than down-weighting KL) open a knee? (new follow-up).
