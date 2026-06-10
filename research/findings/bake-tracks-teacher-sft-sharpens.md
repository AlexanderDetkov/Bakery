---
title: Baking ≠ SFT is faithfulness vs sharpness, NOT a reversal curse — on the Hilbert task neither arm fails the converse; baking tracks the (weakly-propagating) teacher, SFT sharpens past it
outcome: positive            # decisive on the hypothesis's DIRECTION; refutes "baking inherits SFT's reversal curse" on this task
confidence: high             # full seed-0 curriculum grid (n1,n2,n3) + leak-free teacher-forced confirmation; seed-1 CIs + grokking-length SFT in flight
created: 2026-06-09
question: [[q-sft-vs-bake-reversal-curse]]
metric: dprime
run_ids: [qa-ssft-n1-s0, qa-sbake-n1-s0, qa-ssft-n2-s0, qa-sbake-n2-s0, qa-ssft-n3-s0, qa-sbake-n3-s0, qa-tf-bake-n1-s0, qa-tf-sft-n1-s0]
---

## Insight
Putting a matched one-hot **SFT** arm (`--train.objective sft`, CE on the SAME cached trajectories) next to
**baking** (`bake`, KL-to-teacher) and **prompting** (read from the d′ metric's `prompted` state) on the
rigorous `theorem_qa`/Hilbert task **refutes the reversal-curse hypothesis in its predicted direction**: SFT
does NOT collapse on the converse — it discriminates the converse *better* than baking. The real bake-vs-SFT
difference is a **faithfulness-vs-sharpness tradeoff**, not a directional failure:
- **Baking is constrained to mimic the prompted teacher** (its definition: `eval_kl` = KL(teacher‖adapter) = 0.47).
  It therefore *inherits the teacher's weak single-pass propagation* — moderate d′.
- **SFT ignores the teacher** and maximizes the hard-label likelihood (`eval_kl` = 4.43, ~10× baking's). It
  *sharpens* the same supervised tokens into more confident yes/no separation → higher probe d′ across the board,
  while landing far from the teacher's distribution.

So "baking's converse problem" (the old Veld `bake_fact` result) does NOT reproduce on the clean Hilbert task,
for EITHER objective — reinforcing [[CORRECTED-picture-robust-metric]] that the Veld converse failure was
chain/metric-specific, not a property of training on forward text.

## Evidence (matched data; d′: 0 = chance, higher = sharper discrimination; held-out = depths 2–6)
Replicates across the full seed-0 curriculum grid (n1, n2, n3); seeds (s1) chained. `prompting` is training-free, identical across arms.
| curriculum | arm | eval_kl | d1 recall | fwd held-out (d2–6) | converse |
|---|---|---|---|---|---|
| — | prompting (base+u, ref) | — | 1.12 | 0.11–0.18 | −0.07 |
| n1-s0 | **baking** (KL→teacher) | **0.47** | 1.12 | 0.52 | **0.89** |
| n1-s0 | **SFT** (one-hot CE) | **4.43** | 1.43 | 1.21 | **1.79** |
| n2-s0 | **baking** (KL→teacher) | **0.29** | 1.65 | 0.43 | **0.78** |
| n2-s0 | **SFT** (one-hot CE) | **4.39** | 2.19 | 1.63 | **2.07** |
| n3-s0 | **baking** (KL→teacher) | **0.21** | 1.65 | 0.50 | **0.79** |
| n3-s0 | **SFT** (one-hot CE) | **4.38** | 1.89 | 1.78 | **2.64** |
| n1-s0 **TF** | **baking** (leak-free) | **0.22** | 0.45 | 0.54 | **0.73** |
| n1-s0 **TF** | **SFT** (leak-free) | **4.37** | 2.04 | 1.44 | **2.21** |
- **Curriculum depth sharpens SFT but not baking:** as the trained depth grows n1→n3, SFT's converse d′ climbs
  1.79→2.07→2.64 and forward 1.21→1.78, while baking stays flat (converse ~0.8, fwd ~0.4–0.5) — baking remains
  pinned to the teacher's (depth-insensitive, weak single-pass) distribution; SFT is free to exploit the extra
  supervision. eval_kl is depth-insensitive for SFT (~4.4 throughout) and falls for baking (0.47→0.21).
- **2-seed CIs (12 runs: 3 depths × 2 seeds × {bake,sft}) are disjoint on every axis:** BAKE eval_kl 0.28±0.09,
  fwd-heldout d′ 0.42±0.08, converse d′ 0.65±0.28; SFT eval_kl 4.44±0.09, fwd 1.65±0.24, converse 2.31±0.33. No CI
  overlap anywhere (~16× on eval_kl). Both arms' converse d′ stay POSITIVE (no curse); the lone near-failure is
  bake n3-s1 (converse 0.08) — still non-negative, the one chain/seed where baking barely separates the converse.
- **Leak-free (teacher-forced, TF) confirms the whole pattern:** on clean canonical targets (no CoT recitation),
  bake eval_kl 0.22 vs SFT 4.37; SFT uniformly sharper. Forward held-out is UNCHANGED/higher clean (bake 0.54≈0.52
  sampled; SFT 1.44>1.21) → **the sampled-CoT leak did NOT manufacture the bake-vs-SFT forward gap.** Starkest
  single contrast: on IDENTICAL teacher-forced data, KL-bake gets d1 trained-recall **0.45** while CE-SFT gets
  **2.04** — the KL objective is regularized onto the teacher's modest confidence; CE drives to certainty. (TF
  baking's low d1 also reflects thinner targets — 1369 vs 3686 supervised tokens — i.e. the token-richness effect.)
- **No reversal curse, either arm:** converse d′ is POSITIVE and large for both (bake 0.89, SFT 1.79) — both
  reject the (non-provable) converse. The converse probes were NOT among the leaked relations (see caveat), so
  this comparison is clean. Prompting is ~chance on the converse here (−0.07).
- **The eval_kl gap is the headline distinction:** baking 0.47 vs SFT 4.43. Baking lands ON the teacher (that IS
  its objective); SFT lands far from it despite identical data. `train_kl→0` for both (both fit the trained tokens).
- **SFT is uniformly sharper** (d1 1.43>1.12, fwd 1.21>0.52, converse 1.79>0.89) — hard CE produces peakier,
  better-separated logits than KL-to-a-soft-teacher.
- Fig: `results/bake_theorem_qa/_fig_bake_vs_sft_n1s0.png` (per-depth d′ + eval_kl/train-loss overlay).

## What this means for "baking vs prompting vs SFT + propagation" (the user's question)
Three distinct behaviors on the SAME facts/probes: **prompting** carries the fact but propagates weakly
single-pass (held-out fwd ~0.1, converse ~0); **baking** reproduces that prompted teacher faithfully (low
eval_kl) and so propagates about as weakly, but cleanly rejects the converse; **SFT** abandons teacher-fidelity
to sharpen discrimination (high eval_kl, highest d′). If the goal is *behavioral equivalence to base+prompt*
(Bakery's stated objective), baking wins by construction; if the goal is *raw probe discrimination*, one-hot SFT
is sharper — but it is NOT mimicking the prompted model.

## Counter-arguments / threats to validity
- **Full curriculum grid (n1,n2,n3) × 2 seeds = 12 runs, CIs disjoint.** The separation survives seed variation
  (see CI bullet); the one bake outlier (n3-s1 converse 0.08) is non-negative. Still ONE chain (lw_alpha) and ONE
  model (8B) — fact-generality across chains/models remains open.
- **Sampled-CoT leak — RESOLVED ([[sampled-teacher-trajectories-keep-cot]]):** the teacher-forced (clean) bake+SFT
  pair (qa-tf-{bake,sft}-n1-s0) reproduces the entire pattern with forward held-out UNCHANGED/higher (bake 0.54,
  SFT 1.44), so the leak did NOT manufacture the bake-vs-SFT gap. Caveat cleared.
- **Higher d′ ≠ better baking.** SFT's high d′ comes WITH eval_kl 4.43 (it is not the prompt-baking objective).
  Don't read "SFT wins" as "SFT bakes better" — it answers a different question (label-fit, not teacher-mimicry).
- **"No curse" is task-specific.** This is the single-inheritance Hilbert DAG with full depth-1 coverage; the
  grokking-length and graph-structure questions may still expose objective-specific limits.

## Implications
The KL-distillation framing does NOT rescue OR worsen a reversal curse here because there is no curse to begin
with on this task — both objectives install the converse. What the framing DOES change is *how far the adapter
strays from the teacher*: baking stays on it (low eval_kl, the whole point), SFT does not. The ~/Invertibility
reversal curse therefore does not transfer to fact-baking on a pretrained 8B in this regime; the interesting
axis is teacher-fidelity, not directionality.

## Next steps (enqueued)
- Finish the matched SFT sweep (qa-ssft-n2-s0, qa-ssft-n3-s0 — chained) for the curriculum-depth × objective grid.
- A **teacher-forced** (`data.sample_trajectories=False`) bake+SFT pair at n1-s0 to remove the CoT leak and get
  clean forward-propagation magnitudes.
- Cross against [[q-grokking-converse-via-longer-training]]: does long-training SFT diverge further from / closer
  to the teacher; does baking's d′ grok toward SFT's sharpness or stay teacher-faithful?
