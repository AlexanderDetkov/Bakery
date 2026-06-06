---
name: validate-bake
description: Fast read-only methodology pre-flight on a PLANNED bake — inspect the resolved config (--print-config) and confirm it is valid (same base checkpoint both sides, held-out eval, full logits + generated-token masking, concrete seeds/sampling) BEFORE spending GPU. The runtime gate is still the hard backstop.
---

# /validate-bake — methodology pre-flight (config only)

A cheap, read-only check that a planned run is methodologically valid, explained in research terms, before any GPU time. (The runtime validation gate in `bakery/trajectories/base.py` is the hard backstop; this catches problems earlier.)

## Procedure
1. Resolve the config: `python run.py --experiment <name> <overrides> --print-config` (small JSON; no model loaded).
2. Assert, and report PASS/FAIL with the specific violated invariant + the CLAUDE.md section to consult:
   - **Paired comparison** — `model.name` (+ `model.revision`) is a single base checkpoint; both base and baked framings run against it. (Sequential/knowledge baking: `model.adapter_to_load` is set and was baked on this same base.)
   - **Held-out eval** — `generation.eval_num_contexts ≥ 1` and contexts are split disjointly (the gate enforces disjointness at runtime; here just confirm a held-out set is requested).
   - **Full logits** — there is no top-k truncation of the KL distribution anywhere (Bakery never stores logits, so this is structural — just confirm nobody added a truncation knob).
   - **Masking** — generation produces a non-empty continuation (`min_new_tokens ≥ 1`); the supervised span is the generated tokens only (gate-enforced).
   - **Reproducibility** — `seed` is concrete; `temperature`/`top_p`/`top_k`/`max_new_tokens`, `lora_rank`/`lora_alpha`, `learning_rate`/`num_epochs` are all set (no nulls/sentinels).
3. On FAIL, fix the config — do not proceed.

## MUST NOT
Load the model, generate trajectories, read weights/logs, or mutate anything. This is config-only.
