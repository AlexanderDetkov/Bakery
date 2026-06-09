---
title: Faster baking — constant lr 3e-4 + teacher-cache reaches a converged bake ~7× faster; quantization is memory-only (slower), cosine-from-low-lr and weight-decay don't help
outcome: positive
confidence: medium          # multi-seed convergence + clean speed/mem; but n=1 task, one world, eval_kl-as-convergence-proxy, AUROC guard within seed noise
created: 2026-06-09
question: [[q-make-baking-faster]]
metric: eval_kl
run_ids: [hp-lr1e4-cos, hp-lr3e4-cos, hp-lr5e4-cos, hp-lr3e4-const, hp-lr3e4-const-fine, hp-lr3e4-const-s11, hp-lr3e4-const-s12, hp-lr3e4-const-s13, hp-lr1e4-const-fine, hp-lr1e4-const-s11, hp-lr3e4-const-wd05, qbench-bf16, qbench-4bit, qbench-8bit, qa-cmp-n1-s12-off, qa-cmp-n1-s12-on]
---

## Insight
On bake_theorem_qa (Llama-3.1-8B, lw_alpha, n=1), the lever that makes baking faster is **the learning
rate, not quantization**. A **constant lr 3e-4** reaches the target eval_kl (≤0.26) in **~4× fewer epochs**
than the lr 1e-4 default, with no propagation cost; stacked with the **teacher-logit cache** (~1.8×
wall-clock/epoch, within-seed-noise equivalent) the net is **~7× faster wall-clock to a converged bake**.
Quantization is the opposite of a speedup (memory tool, ~1.6–2× *slower*); cosine-decay-from-low-lr is a
pessimization; weight-decay is a no-op at this scale.

## Evidence

**Convergence speed — multi-seed, constant lr, epochs to eval_kl≤0.26** (fine eval, every 2 ep):
| lr | seeds (ep-to-target) | mean | AUROC d2 (mean) |
|---|---|---|---|
| **3e-4 (winner)** | s10=8, s11=8, s12=4, s13=4 | **6.0 ± 2.0** | ~0.69 |
| 1e-4 (baseline) | s10=22, s11=30 | **26** | ~0.69 |
→ **~4.3× fewer epochs** to target. Winner AUROC d2 (~0.69) ≈ baseline (~0.69) — the speedup does NOT
hurt propagation (per-seed AUROC noise ±0.03–0.05; s13 winner 0.633 is the low end of that band).

**lr × schedule (80 ep, eval@10):**
| arm | ≤0.26 by | plateau eval_kl | AUROC d1 / d2 |
|---|---|---|---|
| lr 1e-4 cosine | **never** | 0.279 | 0.73 / 0.67 |
| lr 3e-4 cosine | ep10 | 0.230 | 0.73 / 0.70 |
| lr 5e-4 cosine | ep10 | 0.219 | 0.74 / 0.68 |
| **lr 3e-4 constant** | ep10 | **0.216** | 0.73 / **0.77** |
→ high lr is the lever; **constant ≥ cosine**; **cosine-from-1e-4 never converges** (lr decays below the
useful range). lr 3e-4 constant has the lowest plateau and best AUROC-d2.

**weight_decay (lr3e4 const, s10): no effect** — wd 0 vs 0.05: ≤0.26 @ep8 vs @ep6, plateau 0.222 vs 0.236,
AUROC d2 0.742 vs 0.707 (all within noise). Not a useful lever here.

**Quantization (Phase A, 8B n=1; per 10-epoch+eval block, ~timing):** bf16 883s (18.0 GB); 4-bit 1385s
(**1.57× SLOWER**, 12.1 GB); 8-bit 1824s (**2.07× slower**, 15.5 GB). AUROC comparable across precisions →
quantization is a **memory** tool (use only when a model won't otherwise fit), never a speedup; a bigger
batch off the freed VRAM didn't help wall-clock either.

**Teacher-cache:** ~1.8× wall-clock/epoch (s12 1.80×, s13 1.75×), empirically within seed-noise of a normal
bake (see [[teacher-logit-cache]]).

**Eval-batching:** the `probe_batch_size=0` OOM is fixed (auto-chunks at DEFAULT_PROBE_CHUNK=8); validated
no-OOM on real 8B + bit-exact on the deterministic test fake (commit 1243a9f). Read-only, so it never
changes the baked adapter.

## Counter-arguments / threats to validity
- **eval_kl is the convergence proxy, not propagation.** The recipe is tuned to reach low eval_kl fast; we
  guard quality with AUROC-per-depth (held flat), but eval_kl⟂propagation in general
  ([[propagation-bounded-by-trajectory-coverage]]) — a "converged" bake here is fidelity-converged, and
  downstream propagation can still lag (train to propagation convergence for science claims).
- **One task/world (lw_alpha, n=1), 8B only.** The optimal lr likely shifts with model size, rank, dataset
  size, and n; 3e-4 is the right *direction* (higher), not a universal constant.
- **Baseline ratio is 2 seeds** (winner is 4). The ~4× is a solid direction; the exact factor is noisy.
- **Cache is not bit-exact** (within-seed-noise only) — fine for throughput, not for bit-reproducible
  headlines; the convergence runs used it, so they inherit that caveat.
- **CoT-leak** on held-out d2 (sampled teacher) is constant across arms, so the *relative* AUROC comparison
  is valid; absolute d2 is partly recall-of-recited.

## Implications
- **Recommended fast recipe for this instrument:** `--train.learning_rate 3e-4 --train.lr_schedule constant
  --train.cache_teacher_logits cpu` (+ `--eval.batch_probes true` now that it's memory-safe). Replaces the
  old lr 1e-4 / no-cache / 80–200-epoch default with a ~10–15-epoch bake at ~7× less wall-clock.
- **Drop quantization for speed** — keep it only as a memory escape hatch for models that don't fit.
- Faster bakes make the multi-seed / multi-world hardening the rest of the program needs much cheaper.

## Next steps
- Re-confirm the lr optimum at a second world / higher n and at a different model size before treating
  3e-4 as default beyond lw_alpha-8B.
- Adopt the recipe for the queued science arms (teacher-forced clean arm, equivalence-relation world) so
  they converge in ~15 ep instead of ~80.
