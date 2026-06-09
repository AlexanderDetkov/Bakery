---
title: Teacher-logit cache — a PRAGMATIC (not bit-exact), default-off training speedup that stores logits
created: 2026-06-08
status: adopted
supersedes: []
---

## Decision
Add an OPT-IN teacher-logit cache (`train.cache_teacher_logits` ∈ {off, cpu, gpu}; default **off**) that
memoizes the teacher's FULL-VOCAB float32 log-prob block per trajectory and reuses it across epochs,
skipping the (expensive, long-prompt) teacher forward after the first epoch (~1.5–2× training throughput).
This intentionally relaxes the structural invariant "Bakery never stores logits" — but ONLY behind a
default-off flag, and WITHOUT opening a top-k path:

- `TeacherLogitCache.put` asserts the stored block's `last-dim == vocab_size`, so the cache can only ever
  hold a FULL-vocab distribution. There is still nowhere to put a truncated one → the no-top-k guarantee
  is preserved structurally.
- The KL stays computed in the ONE place (`supervised_kl_terms`); the cache only changes the SOURCE of the
  teacher `t` (a stored full-vocab log-prob block vs a fresh forward), never the KL/masking/shift.
- Refused for objectives whose teacher changes per epoch (pursue: `needs_per_epoch_trajectories`); refused
  for the adapter-on teacher (`base_with_adapter`); never used at eval.
- `train.cache_teacher_verify_every>0` recomputes the teacher at intervals and HARD-FAILS if the cached
  block deviates beyond `cache_teacher_verify_atol` (default 1e-2) — a guard against gross cache bugs /
  genuine non-determinism, tolerant of padding float-noise.

## Rationale — why PRAGMATIC, not bit-exact (the finding)
The plan's premise was that the teacher is "a deterministic pure function → caching is bit-exact." That is
true MATHEMATICALLY but FALSE at the bit level on real batched forwards, and the reason matters:

- The cache-off baseline recomputes the teacher in a PADDED batch every epoch. Right-padding length (hence
  the float-level forward result) depends on the epoch's shuffle/batchmates, so **the baseline's own teacher
  targets already drift epoch-to-epoch at float (ULP) scale.** Verify-mode with `torch.equal` confirmed
  cached != fresh-recompute. There is therefore NO fixed cached value that equals the per-epoch recompute —
  bit-exact equivalence to the baseline is impossible because the baseline is not bit-stable to begin with.
- On the near-degenerate CPU smoke stub (eval_kl ≈ 2e-5, gradients ≈ noise) this float-noise amplified into
  a large adapter divergence — which is why the end-to-end bit-exact test is `xfail` (documents the finding)
  and the empirical check below uses REAL models with meaningful eval_kl.

So the cache perturbs training by the SAME kind/scale of float-noise the baseline already carries across
epochs. Whether that changes the LEARNED model in any meaningful way is an EMPIRICAL question, decided by
comparing two baked models (cache-on vs cache-off) against the natural seed-to-seed difference between baked
models — NOT by a bit-exactness proof.

What IS proven (tests/test_speedup_equivalence.py): (1) within a FIXED batch, store-then-reuse is bit-exact
(`torch.equal`); (2) default-off is byte-identical to historical code (full suite unchanged); (3) the
full-vocab assert and verify HARD-FAIL fire.

## Consequences
- A practitioner can opt in for ~1.5–2× faster baking, accepting float-noise-scale nondeterminism in the
  teacher target (of the same scale baking already has from batched padding). Default off ⇒ zero impact on
  any existing/committed run or finding.
- Findings produced WITH the cache on must note it; prefer cache-off for a headline number unless the
  empirical-similarity check (below) is cited.
- Not bit-exact ⇒ do NOT use the cache to claim reproducibility of a specific adapter; use it for throughput.

## Empirical similarity check (does the cache learn the SAME baked model?)
Experiment `qa-cmp-n1-s{12,13}-{off,on}` (bake_theorem_qa, Llama-3.1-8B, lw_alpha, n=1, 100 ep, matched
split_seed=model_seed=seed so off vs on differ ONLY by the cache). Compare the cache-on baked model to the
cache-off baked model (same seed) on the headline readouts (eval_kl trajectory, AUROC/d′ per depth) and put
that divergence next to the seed-to-seed (s12 vs s13) divergence between two baked models. Also measures the
realized speedup.

**RESULT (2026-06-08, qa-cmp-n1-s{12,13}-{off,on}, 100 ep, backend=cpu, paired so off↔on differ ONLY by the cache):**

*Similarity* — post-plateau (ep≥60) mean. Raw: s12 off/on eval_kl 0.2229/0.2265, AUROC d1 0.711/0.721,
d2 0.680/0.688; s13 0.2319/0.2309, d1 0.739/0.731, d2 0.711/0.713. The **cache effect is SMALLER than
seed-noise on every metric** — i.e. turning the cache on changes the baked model LESS than changing the seed:

| contrast | Δeval_kl | Δ AUROC d1 | Δ AUROC d2 |
|---|---|---|---|
| cache OFF vs ON, seed 12 | 0.0036 | 0.010 | 0.009 |
| cache OFF vs ON, seed 13 | 0.0010 | 0.008 | 0.002 |
| seed 12 vs 13 (noise floor, off) | 0.0090 | 0.028 | 0.031 |
| seed 12 vs 13 (noise floor, on) | 0.0044 | 0.010 | 0.024 |

*Speedup* — end-to-end wall-clock incl. the every-10 eval (which the cache does NOT touch, so training-only
is faster still): s12 87.7→48.6 s/epoch = **1.80×**; s13 93.2→53.4 s/epoch = **1.75×**.

**CONCLUSION: the cache is not bit-exact but is empirically equivalent — it perturbs the learned baked model
by less than the irreducible seed-to-seed variation, at ~1.8× end-to-end throughput.** Safe to use for
throughput when this check is cited; keep default-off and prefer cache-off for a bit-reproducible headline.
Caveats: n=1, 2 seeds, cpu backend, sampled-teacher (the CoT-leak is constant across off/on so it cancels in
the paired contrast). Combining with eval-batching (read-only, bit-exact on the fake; auto-chunks at
`propagation.DEFAULT_PROBE_CHUNK` rows/forward → memory-safe on 8B, after the OOM fix) should raise the
realized whole-run speedup further by shrinking the eval term that currently dilutes the ratio.
