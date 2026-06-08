---
title: Baked propagation tracks the TRAINED depth and the TEACHER's reach — it does not generalize one hop further, and the deeper hop does not grok
outcome: positive
confidence: high
created: 2026-06-07
updated: 2026-06-08
question: [[q-n-curriculum-propagation-dynamics]]
metric: dprime
run_ids: [qa-bake-n1-s10, qa-bake-n2-s10, qa-bake-n1-s11, qa-bake-n2-s11, qa-bake-n1-s12, qa-bake-n2-s12, qa-bake-n1-s13, qa-bake-n2-s13]
---

## 4-SEED UPDATE (S4c2, 2026-06-08) — confidence medium→high
Seed-CI added seeds 12,13 (→ 4 seeds, 8 runs). Baked d′ post-saturation (ep≥50), mean ± popsd:
| depth | n=1 (s10,s11,s12,s13) | n=2 | Δ(n2−n1) | prompting ceiling |
|---|---|---|---|---|
| d1 | 0.95 ± 0.32 {0.99,0.64,0.72,1.46} | **1.11 ± 0.04** {1.12,1.07,1.09,1.18} | +0.16 (overlap) | 1.12 |
| d2 | 0.70 ± 0.18 {0.96,0.57,0.79,0.50} | 0.92 ± 0.17 {0.89,1.18,0.69,0.90} | **+0.21 (ranges OVERLAP, ~p≈0.15, NS)** | 0.53 |
| d3 | −0.19 ± 0.19 | −0.15 ± 0.37 | +0.04 (noise) | 0.00 |
- **d3 ceiling now robust at 4 seeds**: baked d3 ≈ 0 for BOTH n, every seed; curriculum does not bootstrap d3.
- **d1**: n=2 recalls reliably (1.11±0.04 ≈ teacher); n=1 noisier (one outlier 1.46).
- **d2 curriculum effect is WEAK and NOT significant at 4 seeds** (+0.21, overlapping ranges). BUT both n
  exceed the prompting ceiling (0.53) on average → baking gives a small depth-2 bump even at n=1 (within
  noise for n=1). Net: the n-curriculum's depth-2 benefit is real-direction but small relative to seed noise.
- **No grokking** reconfirmed: d′ set by ~ep50, flat through ep120–170; eval_kl plateaus ~ep30.

## AUROC re-read (S4c4, 2026-06-08) — most of the "noise" was the d′ estimator
Per [[paired-matched-seed-protocol]], re-read the SAME 8 runs with AUROC (rank-based; no Φ⁻¹ blow-up):
- Noise collapses: n=1 depth-2 per-seed scatter sd 0.18 (CV ~26%) under d′ → **sd 0.02 (CV ~3%)** under AUROC.
- The curriculum contrast SHARPENS: n=1 AUROC d2 = 0.68±0.02, n=2 = 0.75±0.05, Δ=+0.06 with separation
  Δ/pooled-sd **1.60** (vs 1.18 under d′); ranges only just touch (n1 max 0.71, n2 min 0.70). So the n=2≥n=1
  depth-2 effect is real and cleaner than d′ suggested, though still modest.
- d3 sits at **AUROC ≈ 0.50 (chance)** for both n — the teacher ceiling, now unambiguous (the noisy negative
  d′ was estimator artifact). Figure: `results/bake_theorem_qa/_fig_traingrok_n1n2_auroc.png`.
- Caveat unchanged: still single-arm-per-condition (unpaired across seeds) + CoT leak; a paired matched-seed
  re-run would tighten the d2 Δ CI further. bacc over-saturates (separation 0.38) — AUROC is the right primary.

## Insight
On the lw_alpha proof-system instrument (proof-depth = hop, bias-immune d′, 8B), **the prompted teacher
propagates ~1.5 hops zero-shot (d1 d′=1.12, d2 d′=0.53, d3 d′=0)**, and **baking faithfully reproduces
this ceiling**: baked d′ reaches exactly the depth the curriculum trains/distills (d1 for n=1; d1+d2 for
n=2) but does **NOT** compositionally generalize one hop beyond it — **baked d3 stays ≤ 0 in every arm and
both seeds, with no late grokking through epoch 170** (≈6× past the eval_kl plateau and through the
empirical grokking window). The n=2 curriculum lifts held-out d2 above the teacher's zero-shot d2 (it
trains a d2 subset), but neither n nor long training conjures the hop the teacher cannot do.

## Evidence
- **Setup**: `bake_theorem_qa`, Llama-3.1-8B-Instruct (revision default/latest, identical both KL sides —
  gate-checked), objective=bake, **sampled teacher** (canonical), lw_alpha world, lora r/α=16, lr 1e-4,
  bs4 ga1 (eff. batch 4), eval_period 10. Fully reseeded (split_seed=seed). Stopped at epochs 110–170
  (data sufficient; dynamics saturated by ~epoch 50). d′ per depth from ~16 true + 16 false depth-matched
  probes (so ±0.3–0.5 sampling noise per cell).
- **Prompting ceiling — perfectly reproducible across all 4 arms**: prompted d′ at d1..d4 = `[1.12, 0.53,
  0.0, 0.0]` (independent of n and seed, as it must be — same teacher). ⇒ in-context axioms support ~1
  strong hop, a weak 2nd hop, nothing at d≥3.
- **Baked d′, mean over post-plateau evals (epoch ≥ 40)**:
  | arm | d1 | d2 | d3 | d4 | eval_kl_final |
  |---|---|---|---|---|---|
  | n1-s10 | 0.99 | 0.93 | −0.22 | 0.10 | 0.253 |
  | n2-s10 | 1.07 | 0.92 | −0.18 | 0.37 | 0.153 |
  | n1-s11 | 0.64 | 0.55 | −0.43 | −0.07 | 0.252 |
  | n2-s11 | 1.05 | 1.18 | −0.73 | 0.02 | 0.159 |
- **d3 ceiling (decisive)**: baked d3 **max-ever** = 0.00 / 0.00 / −0.15 / −0.61 — never positive, full
  trajectory flat (e.g. n1-s10 d3 over 17 evals to epoch 170: all ∈ [−0.4, 0.0]). prompted d3 = 0. So
  neither prompting nor baking reaches d3, and the ≤2-hop curriculum (n=2) does **not** bootstrap held-out
  d3. **No grokking**: d3 shows no upward transition past the eval_kl plateau (~epoch 30) or through the
  empirical grokking window (≈135–165 from prior cycles) on the furthest run (n1-s10 @ 170).
- **eval_kl ⟂ propagation, reconfirmed + extended**: eval_kl plateaus by ~epoch 30 (n1 ~0.25, n2 ~0.15)
  while depth-d′ levels are set early and then flat — no late deep-hop grokking. (Cf.
  [[size-helps-fidelity-not-the-propagation-gap]].)
- **Curriculum effect on d2 (suggestive, direction-consistent, seed-confounded)**: the clean mechanism is
  visible in seed 11 — n1-s11 d2=0.55 ≈ teacher's 0.53 (n=1 never trains d2, so baking can only distill the
  teacher's weak d2), while n2-s11 d2=1.18 ≫ teacher (n=2 trains a d2 subset → exceeds zero-shot). Seed 10
  shows no gap (n1-s10 d2=0.93 ≈ n2-s10 0.92) because n1-s10's d2 is anomalously high. Mean: n2 d2=1.05 vs
  n1 d2=0.74 (+0.31), but per-seed Δ ∈ {−0.01, +0.63} — does NOT cleanly exclude 0 with 2 seeds.

## Counter-arguments / threats to validity
- **2 seeds is too few for the d2 curriculum claim.** The entire average effect rides on seed 11; seed 10's
  n=1 run discriminates d2 at 0.93 with no d2 training — either a genuine (unstable) weak 1-hop compositional
  bonus or probe noise. UNRESOLVED → seed-CI (seeds 12,13) launched to settle it.
- **Per-depth d′ is noisy** (~16+16 probes): cell-level differences <~0.4 are within noise. The d3 ≤ 0
  result is robust (every arm, every eval), but small d2 gaps are not.
- **Data-quantity confound for n=2**: n=2 trains on MORE relations than n=1 (adds the depth-2 subset), so
  its lower eval_kl and higher d1/d2 partly reflect more training data, not "deeper curriculum" alone. The
  d2-specific lift is the curriculum mechanism, but a matched-trajectory-count control would be cleaner.
- **Runs stopped at epoch 110–170, not 400.** The grokking-null is "through ~170" (past the known window),
  not "through 400". A longer sentinel would harden "no LATE d3 grokking". (Saved checkpoints exist.)
- **Sampled-teacher caveat** ([[sampled-teacher-trajectories-keep-cot]]): held-out d′ under sampling is
  partly recall-of-recited; mitigated here by short answers + the DAG contamination filter (gate-enforced).

## Implications
- Sharpens [[propagation-bounded-by-trajectory-coverage]]: baking installs **max(trained-depth,
  teacher-reach)** and adds **no free compositional hop** — the propagation frontier is exactly where the
  training/teacher signal stops. For the central "baking vs prompting" question: **baking ≈ prompting at the
  depths the teacher reaches, can exceed prompting at a depth it explicitly trains (n=2 → d2), and is hard-
  capped at the teacher's blind spot (d3) regardless of training length.** Knowledge "propagation" via
  single-pass baking is therefore a coverage/teacher-ceiling phenomenon, not an emergent multi-hop one.
- Directly answers the directive's grokking premise for THIS task: the deeper hop does **not** grok with
  longer baking (≥170 ep) — consistent with the corrected reading in [[CORRECTED-picture-robust-metric]]
  that earlier "grokking" was prior-recovery, not new capability.

## Next steps
- [[q-n-curriculum-propagation-dynamics]] stays **active**: seed-CI (seeds 12,13 → 4 seeds) RUNNING to
  settle the d2 curriculum gap and the n1-s10 d2 anomaly (compositional bonus vs noise).
- Enqueue: (a) **matched-trajectory-count** n=1 vs n=2 control (rule out the data-quantity confound);
  (b) a **longer grokking sentinel** (≥400 ep, one arm) to harden the no-late-d3 claim;
  (c) the **trajectory-regularization** axis ([[q-regularization-preserves-behavior]]) — does anchoring hurt
  the (already teacher-bounded) propagation; (d) **teacher-forced** arm (sample_trajectories=False) — does
  injecting ground-truth d2/d3 (beyond the teacher) push the frontier, isolating "teacher ceiling" from
  "objective limit".
