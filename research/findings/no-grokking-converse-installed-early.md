---
title: No grokking — baking installs the converse EARLY and it stays stable to 10k epochs; training past the eval_kl plateau does not trigger a late transition (and mildly over-trains)
outcome: negative           # DECISIVE-NEGATIVE on the grokking hypothesis; positive corollary = converse healthy from the start
confidence: high            # full 10k-epoch run, 200 evals, d′ instrument; n=1 chain/seed (Hilbert lw_alpha, n1-s0)
created: 2026-06-10
question: [[q-grokking-converse-via-longer-training]]
metric: dprime
run_ids: [qa-sbake-n1-s0-long]
---

## Insight
Training the LoRA bake **100–200× past the point eval_kl plateaus** (10 000 epochs, 200 evals) does NOT make the
converse grok. The eval_kl plateaus essentially at epoch 0 (~0.41–0.48 throughout) and the held-out converse d′
shows **no late phase transition**: it is ALREADY healthy at epoch 50 (0.85), bumps to a mid-training peak ~1.1
(epochs ~3k–5k), then **regresses back to ~0.8** by 10k. Forward held-out d′ behaves the same (0.44→peak 0.72→0.50).
So the premise of the grokking hypothesis — "the converse fails at short training and might generalize late" —
DOES NOT HOLD on the rigorous Hilbert task: baking installs the converse immediately (consistent with
[[bake-tracks-teacher-sft-sharpens]]), so there is nothing broken to rescue. The only late effect is mild
**over-training regression** (peak ~ep 4–5k, then drift down), the opposite of grokking.

## Evidence (qa-sbake-n1-s0-long, Llama-3.1-8B, lw_alpha, sampled bake, 10k ep / 200 evals)
| epoch | eval_kl | fwd held-out d′ (d2–6) | converse d′ |
|---|---|---|---|
| 50    | 0.48 | 0.44 | **0.85** |
| 850   | 0.46 | 0.62 | 1.00 |
| 3250  | 0.41 | 0.65 | 1.04 |
| 4850  | 0.43 | 0.72 | **1.12 (peak)** |
| 6450  | 0.44 | 0.67 | 1.01 |
| 8050  | 0.43 | 0.54 | 0.89 |
| 10000 | 0.44 | 0.51 | **0.82** |
- **eval_kl plateaus at epoch 0** and never improves — the distillation loss is uninformative about the (separate)
  d′ dynamics, re-confirming eval_kl ⟂ propagation ([[propagation-bounded-by-trajectory-coverage]]).
- **Converse d′ > 0 from epoch 50** and stays positive to 10k — NO collapse, NO late jump; a broad mid-training
  hump then regression. Same shape for forward held-out.
- Figs: `results/bake_theorem_qa/_fig_grok_long_FINAL_dprime.png`, `_fig_grok_long_FINAL_bacc.png`.

## What this resolves
[[q-grokking-converse-via-longer-training]] asked whether baking's converse failure is the reversal curse,
learnable late via grokking. Answer: on the Hilbert/d′ instrument there is **no converse failure to begin with**
(it's installed early), and **long training does not produce a late transition** — it mildly over-trains. The
reversal-curse → grokking framing (imported from ~/Invertibility) **does not transfer** to fact-baking a
pretrained 8B in this regime. Pairs with [[bake-tracks-teacher-sft-sharpens]] (matched SFT also has no curse;
the bake-vs-SFT axis is teacher-fidelity, not directionality).

## Counter-arguments / threats to validity
- **n=1 chain × 1 seed** (lw_alpha, n1-s0). The mid-training hump-then-regression could be seed noise; a second
  seed/chain long run would confirm the over-training drift. (The grokking-length SFT mirror qa-ssft-n1-s0-long is
  running — it tests whether the SHARPER objective shows any late dynamics baking doesn't.)
- **Sampled trajectories** (CoT leak on forward held-out); but the converse channel is leak-free, and the leak was
  shown not to inflate forward d′ either ([[bake-tracks-teacher-sft-sharpens]] teacher-forced control).
- **This is the n1 curriculum** (depth-1 axioms only). A weight-decay arm and a converse/path-trajectory arm (E2 in
  the question) were NOT run — the original question proposed them; given the converse is already installed, they are
  lower priority, but a wd>0 long run would close the "regularization-driven grokking" door fully.
- "No grokking" is specific to this d′/proof-depth instrument; a different elicitation could differ.

## Long CE-vs-KL cross-ref (qa-ssft-n1-s0-long, 10k ep — DONE)
Ran the matched grokking-length SFT mirror. Result confirms and extends the finding:
- **Neither objective groks.** SFT's converse d′ is already 1.9 at ep 50 and ends 2.4 at 10k — positive throughout,
  modest monotone-ish rise, NO late phase transition. So "no grokking" is not bake-specific.
- **Divergent long-training dynamics.** KL-bake mildly OVER-TRAINS (conv d′ 0.93→peak 1.23 @ ~ep2.5k→0.85 @ 10k;
  fwd 0.44→0.83→0.51) — teacher-anchoring + extra epochs slightly degrade held-out d′. CE-SFT instead SUSTAINS/
  slightly sharpens (conv 1.9→2.4; fwd 1.35→1.58) with no regression.
- **The fidelity gap is permanent:** SFT eval_kl stays ~3.9→4.5 across 10k ep — it never drifts toward the teacher
  even with 200× the training. Confirms [[bake-tracks-teacher-sft-sharpens]] is a stable property of the objective,
  not a short-training transient.

## Next steps
- Optional: one wd=0.05 long bake to fully rule out regularization-driven late grokking (close E1).
