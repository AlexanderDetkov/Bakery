---
title: The "baking affirms the converse / is direction-blind" headline does NOT survive the bias-immune proof-system instrument; baking's real divergence from prompting is over-affirming UNRELATED (cross-component) pairs
outcome: positive          # a correction that sharpens: refutes the converse-collapse, relocates the associative signal
confidence: medium         # clean across 4 seeds + bias-immune metric, but CoT-leak on d2, one world, preempted/under-converged runs
created: 2026-06-08
question: [[q-propagation-deductive-chain]]
metric: dprime
run_ids: [qa-bake-n1-s10, qa-bake-n1-s11, qa-bake-n1-s12, qa-bake-n1-s13, qa-bake-n2-s10, qa-bake-n2-s11, qa-bake-n2-s12, qa-bake-n2-s13]
---

## Insight
Re-reading the EXISTING seed-10–13 adapters' `metrics.json` per negative-family (no re-scoring — the
`dprime` metric already logs `auroc_<state>_d<d>_<negtype>` and `fa_<state>_d<d>_<negtype>`), the
pre-Hilbert headline that **baking affirms the false converse / is direction-blind**
([[baking-is-associative-prompting-is-directional]], [[yes-saturation-is-fact-general-converse-amplification-is-not]])
**does not reproduce** on the controlled proof-system world (lw_alpha) under the bias-immune metric.
Baked converse-AUROC is ≥ prompting at both depths and baked converse false-alarm is far from saturation;
at d1 PROMPTING actually affirms the converse *more* than baking. The signal that DOES survive — baking
diverging from prompting — is on the **cross** family (genuinely unrelated, different-component pairs):
baking spuriously connects unrelated concepts, prompting does not. So baking's "associative" error is
**over-connection across the partition, not symmetrization of direction.**

## Evidence (post-plateau mean ± popsd over 4 seeds; ep≥100 w/ last-2 fallback; analysis/neg_family_auroc.py)
**AUROC(true@d vs negative)** — higher = better discrimination, 0.5 = chance:
| depth | family | prior | prompted | baked n=1 | baked n=2 |
|---|---|---|---|---|---|
| d1 | converse | 0.45 | 0.68 | 0.68±.02 | 0.73±.02 |
| d1 | cross | 0.49 | 0.95 | 0.69±.03 | 0.76±.03 |
| d1 | missing_edge | 0.40 | 0.95 | 0.79±.02 | 0.82±.01 |
| d2 | converse | 0.47 | 0.69 | 0.80±.02 | 0.84±.06 |
| d2 | cross | 0.57 | 0.93 | 0.66±.04 | 0.77±.07 |
| d2 | missing_edge | 0.41 | 0.54 | 0.59±.04 | 0.64±.04 |

**FA = P("Yes" | negative)** — yes-saturation signature (→1.0); lower = better:
| depth | family | prior | prompted | baked n=1 | baked n=2 |
|---|---|---|---|---|---|
| d1 | converse | 0.07 | **0.50** | 0.39±.11 | 0.38±.03 |
| d1 | cross | 0.08 | 0.08 | 0.23±.09 | 0.10±.04 |
| d2 | converse | 0.07 | 0.07 | 0.18±.10 | 0.07 |
| d2 | cross | 0.08 | 0.08 | 0.34±.07 | 0.17±.08 |

- **No converse-collapse:** baked converse-AUROC 0.68 (d1) / 0.80–0.84 (d2), ≥ prompting (0.68 / 0.69);
  baked converse-FA 0.07–0.39 (nowhere near the ~1.0 saturation of the Veld belief-metric runs). At d1,
  prompting affirms the converse MORE (FA 0.50) than baking (0.39). The d′ instrument is built to expose
  yes-saturation (H≈FA ⇒ AUROC≈0.5); it shows the opposite.
- **Cross is where baking loses to prompting:** prompted rejects cross near-perfectly (AUROC 0.93–0.95,
  FA 0.08); baked is markedly worse (AUROC 0.66–0.77, FA 0.23–0.34). Mild (not saturation); the n=2
  curriculum cuts it (cross FA 0.34→0.17, AUROC 0.66→0.77).
- **d3 = chance for every family, both states** (AUROC ~0.5) — the teacher ceiling is family-independent;
  baked d3 FA elevated (0.29–0.64) is the only saturation-like drift, and only where nothing is learned.
- **n=1 d2 ≈ prompting under AUROC (corrects a metric mismatch):** AUROC prompted d2 = 0.71, baked n=1 =
  0.69 (≈, slightly below), baked n=2 = 0.76 (above). The "n=1 exceeds prompting at d2" impression in
  [[baked-propagation-tracks-trained-depth-no-compositional-bonus]] came from comparing baked-d′ (0.70) to
  prompted-d′ (0.53); under the primary low-variance stat (AUROC) the n=1 advantage vanishes — only the
  CURRICULUM (n=2) exceeds prompting at d2.
- Setup: bake_theorem_qa, Llama-3.1-8B, sampled teacher, lw_alpha, lora r/α=16, lr 1e-4, bs4. 8 runs
  (n∈{1,2} × seeds 10–13); read at post-plateau evals (eval_kl ≈0.25 n1 / ≈0.15 n2).

## Counter-arguments / threats to validity
- **CoT leak inflates d2** ([[sampled-teacher-trajectories-keep-cot]]): 29–50 held-out probes are recited
  in the teacher's sampled CoT, so held-out d2 (incl. d2 converse/cross AUROC) is partly recall-of-recited.
  The d1 row is the clean comparison; the clean d2 test is the unfinished teacher-forced arm.
- **Runs preempted (`status=failed`); n2-s12/s13 reached only ep80–90** (under-converged) → the n=2 d2
  numbers are the noisiest (±.06/.07).
- **One world (lw_alpha), single-arm-per-seed (unpaired).** This shows the controlled instrument does not
  reproduce the collapse; it does NOT retroactively explain the Veld/Tellus chains where it was seen
  ([[CORRECTED-picture-robust-metric]] already flagged converse failure as chain-specific).
- **The d1-converse is a weak directionality test here:** both prompting (FA 0.50) and baking (FA 0.39)
  hover near chance on affirming the immediate converse of a stated axiom, so "directional" is not strongly
  true of EITHER at d1; the discrimination signal is mostly at d2.

## Implications
- **Corrects** [[baking-is-associative-prompting-is-directional]] / [[yes-saturation-is-fact-general-converse-amplification-is-not]]
  for this instrument: under a bias-immune metric on a controlled multi-component world, baking is
  approximately directional (rejects the converse ≈ as well as / better than prompting), not direction-blind.
- **Refutes the simple symmetric-closure conjecture** (relational-generalization note §5a): `rst(E)` predicts
  baking AFFIRMS the converse (same component) and REJECTS cross (different component). The data show the
  opposite emphasis — baking mostly rejects the converse and LEAKS on cross. So baking here approximates the
  TRUE relation but noisier, with the noise pointed at **cross-component over-association**, not
  symmetrization. The §5a conjecture is downgraded/relocated accordingly (note revised).
- The surviving, robust sense of "associative shadow" on this instrument is **over-connection across the
  partition** (the `cross` family) — exactly the formulation's "partitioned-domain" negative (§2). That is
  the per-family signature to track going forward, not the converse.

## Next steps
- Re-read the **teacher-forced** arm per-family once it converges (no CoT leak) → the clean d2 converse/cross
  numbers ([[q-teacher-ceiling-vs-objective-limit]]).
- The equivalence-relation world (formulation §3) becomes the sharp test: if baking truly over-connects
  within reachability, an `rst`-world (where cross-component IS the only false family) should make the
  cross-leak the dominant — and only — error. Cheap (components already computed).
- Paired matched-seed re-run ([[paired-matched-seed-protocol]]) to tighten the cross-family gap CI.
