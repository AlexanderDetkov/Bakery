---
title: Does baking's cross-component OVER-CONNECTION scale with the affirmation density ("yes-pressure") of the trajectory distribution?
status: open
priority: high
created: 2026-06-10
hypothesis: >
  [[equivalence-world-confirms-cross-over-connection-mechanism]] found baking's cross over-affirmation is
  STARKER in the equivalence world (d2 cross-FA 0.93) than the directed world (0.61), and the difference tracks
  one thing: the equivalence bake distribution is almost all "…are the same kind → Yes" (every within-class pair,
  both orders, is a positive), whereas the directed bake has forward-only positives. Conjecture: baking installs
  an over-permissive (low-precision) closure whose SIZE scales with how much the trajectory distribution affirms
  relatedness. If so, the cross-leak is a controllable property of the bake data, not an intrinsic LoRA limit.
acceptance_criteria: >
  On the SAME directed lw_* worlds, vary the affirmation density of the bake trajectories (e.g. forward-only/
  high-yes vs a balanced set that also exercises true NEGATIVES / rejections at matched count + convergence),
  hold eval_kl matched. POSITIVE if baked d2 cross-FA drops materially as yes-density drops (over-connection is
  data-controllable); NEGATIVE if cross-FA is invariant to affirmation density (intrinsic to copying single-pass
  behavior). Report cross-FA + cross-AUROC by depth, prompted vs baked, per-family (analysis/neg_family_auroc.py).
experiment: >
  Directed lw_alpha (+2-3 siblings), n=1, fast recipe. Arms: (A) current forward-only trajectories; (B) balanced
  trajectories that include teacher-generated rejections of cross/converse probes (needs the teacher to actually
  GENERATE "No" on those — audit first, cf. the teacher-audit note in q-grokking). Matched trajectory count.
links: [[equivalence-world-confirms-cross-over-connection-mechanism]], [[converse-collapse-does-not-survive-bias-immune-instrument]], [[trajectory-type-is-a-binary-coverage-gate]], [[propagation-bounded-by-trajectory-coverage]]
---

## Why now
This is the natural mechanistic follow-up to the confirmed over-connection finding: we know WHAT baking gets
wrong (affirms unrelated pairs) and that the n-curriculum partially fixes it; this asks WHETHER the error is a
controllable function of the bake distribution's affirmation density — the actionable lever for fixing it.
