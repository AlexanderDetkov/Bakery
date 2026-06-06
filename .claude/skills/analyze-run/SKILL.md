---
name: analyze-run
description: Analyze a completed bake from its standardized artifacts (metrics.json/manifest.json) — headline eval_kl (final + best), convergence, plot a curve — and judge it against an open question's acceptance criteria. Reads only results/ JSON; never imports the training stack.
---

# /analyze-run — judge a bake from artifacts only

## Procedure
1. `from analysis.load import load_run; r = load_run("<run_dir>")` (or `python -m analysis.aggregate ...` for a sweep). This reads ONLY `manifest.json`/`config.json`/`metrics.json`.
2. **Headline:** `eval_kl` — report final (`r.final("eval_kl")`) and best (`r.best("eval_kl")`, min, since LOWER is better). This is the held-out distillation objective: did baking converge on UNSEEN contexts?
3. **Dynamics:** scan the `eval_kl` / `train_kl` series for convergence / instability / divergence; note epochs-to-plateau. A train_kl that falls while eval_kl stalls = overfitting the bake contexts.
4. **Plot:** `python -m analysis.plot_curves <run_dir> --metrics eval_kl train_kl` → writes a PNG into the run dir. Report the PATH; do not embed pixels.
5. **Sanity vs manifest:** `status == "completed"`; the SAME `base_checkpoint` revision underlies both sides (paired-comparison invariant); `data_stats.pairing_opted_out` and `n_eval_ctx` are sane; seeds + sampling recorded.
6. **Verdict:** pass / fail / inconclusive vs the question's `acceptance_criteria`, WITH numbers. Name confounds (different base checkpoint, too few trajectories/contexts, eval contexts too close to the bake distribution, KL low but behavior still diverges). These become the finding's counter-arguments.
7. Output a short structured summary to feed **`/record-finding`**.

## MUST NOT
Import `bakery.runner` / `bakery.objectives` / transformers / peft from an analysis turn. Load `.pt` / `.safetensors`. Read the full `log.txt`. Hardcode a path beyond the given `run_dir`.
