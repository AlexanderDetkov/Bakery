---
title: Report propagation as AUROC (not raw d′), and run condition-contrasts as PAIRED matched-seed sets
created: 2026-06-08
status: adopted
supersedes: []
---

## Decision
Two standing methodology rules for the d′ propagation instrument (`bake_theorem_qa` / `bake_logic`),
adopted after the n-curriculum results came out noisy:

1. **Lead with AUROC** (and balanced accuracy as a secondary), not the raw per-depth d′, as the headline
   propagation readout. d′ = Φ⁻¹(hit) − Φ⁻¹(FA) amplifies variance near 0/1; AUROC (rank-based,
   threshold-free) and bacc are computed from the SAME forward pass at far lower variance. Both are already
   logged (`dprime.auroc_<state>_d<d>`, `dprime.bacc_<state>_d<d>`); use `analysis/plot_dprime.py --stat auroc`
   and `analysis/plot_traingrok.py --stat auroc`, and `aggregate.py --metric dprime.auroc_baked_d2 --mode final
   --status any`. Keep d′ as a cross-check (it's the bias-immune signal-detection quantity), but do not let
   its scatter drive conclusions. AUROC chance = 0.5; d′ chance = 0.

2. **Run every arm of a CONTRAST as a paired matched-seed set.** Fix `--seed = --model.... model_seed and
   --data.split_seed = k` to the SAME integer `k` across all arms of a contrast (n=1 vs n=2; reg doses;
   bake vs teacher-forced), varying ONLY the condition. Replicate the whole matched set across
   `k ∈ {10, 11, 12}` (seeds start at 10). Report per-set paired differences, then average — not pooled
   unpaired runs. Prefer teacher-forced targets (`data.sample_trajectories=False`) when the goal is a clean
   contrast (deterministic targets + no CoT leak).

## Rationale
- The held-out split in `theorem_qa.py::_build_relations` is a pure function of `(split_seed, depth)`,
  STABLE across the condition, and `model_seed` deterministically seeds LoRA init (runner.py:123). So
  matching `split_seed`+`model_seed` across arms makes them share the EXACT held-out probes and init; the
  condition is then the only difference, and the "which-probes-held-out" + "where-init-started" nuisance
  variance cancels in the contrast.
- Cycles S4c1–c3 reseeded everything (`split_seed = seed`), stacking three nuisance sources into every
  between-arm comparison — the main reason the n=2−n=1 depth-2 effect looked seed-confounded.
- Empirical check (S4c3, 4 seeds, depth-2): switching the readout from d′ to AUROC dropped the n=1
  per-seed spread from sd 0.18 (CV ~26%) to sd 0.02 (CV ~3%) — an ~8× noise reduction on the SAME runs —
  and SHARPENED the curriculum contrast (separation Δ/pooled-sd: 1.18 → 1.60). bacc over-saturated
  (separation 0.38), so AUROC is the right primary. This is the evidence the noise was largely the
  estimator, not the signal. See [[regularization-buys-behavior-preservation-cheaply]] /
  [[baked-propagation-tracks-trained-depth-no-compositional-bonus]].

## Consequences
- Findings should quote AUROC as the headline propagation number, with d′ as a parenthetical cross-check.
- New contrasts MUST use matched `split_seed`+`model_seed` across arms; a contrast that varies the split is
  treated as under-powered. The earlier (reseeded) curriculum + regularization runs are valid but their
  contrasts are unpaired — re-run paired if a tighter CI is needed.
- Cheaper, more reproducible comparisons without spending seeds; complements (does not replace) more seeds.
- Available-but-not-yet-adopted next levers if AUROC + pairing still aren't tight enough: densify the probe
  bank (`make_logic_world.py --qa --true-per-depth 38`) and per-probe bootstrap CIs (see the noise plan).
