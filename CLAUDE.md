# Bakery — guide for AI editors (authoritative)

Read this before changing anything. It is the source of truth for the methodology and the
invariants — its job is to stop a well-meaning edit from silently invalidating an experiment.

## What this studies

**Prompt Baking**: convert a prompt `u` into LoRA weights `θ_u` so the UNPROMPTED baked model
mimics the PROMPTED base model:

    B(θ, u) = argmin_θ_u  D_KL( P_θ(·|u) ‖ P_θ_u(·) )

Operationally: sample trajectories from the prompted base model over diverse contexts `x0`; compute
token-level KL between the base-with-prompt logits (teacher) and the baked-without-prompt logits
(student) on the GENERATED tokens only; backprop into a LoRA adapter. Both logits come from ONE peft
model by toggling the adapter (`bundle.base()` disables it → teacher; `bundle.baked()` → student), so
trajectories store token ids + masks, never logits. Headline metric: **`eval_kl`** (held-out; lower
is better). Variants (extension seams): pursuit (iterative), knowledge baking (sequential
composition), half-baking (α-scaled adapter), re-prompting.

## Invariants that MUST NEVER break (the 5 validity criteria)

1. **Identical base checkpoint** — both sides of the KL run against the same model id + revision.
2. **Full-vocab logits** — no top-k truncation. (Structural: Bakery never stores logits.)
3. **Generated-token masking** — KL on EXACTLY the generated tokens; the same `y` in both framings;
   pad/eos never in the supervised span.
4. **Held-out eval contexts** — disjoint from the trajectory-generation contexts.
5. **Reproducibility** — seeds + sampling params + all hyperparameters recorded in the manifest.

Plus **determinism**: a given seed reproduces the same context split, trajectories, and adapter init.

## The safety kernel — use it, don't go around it

`bakery/trajectories/base.py` makes faulty data *un-constructable*:
- **`TrajectoryDataset`** (frozen) is the ONLY object training/eval accept; it can ONLY be produced by
  `run_validation_gate(...)` (a private sentinel makes hand construction raise). Never build it directly.
- The gate runs, in order: (E) base-checkpoint + tokenizer identity, (A) mask alignment, (B) train/eval
  context disjointness, (C) completeness/shape, (D) pluggable pairing — then freezes.
- **`assert_mask_alignment`** is the canonical statement of criterion 3, stated on RAW token ids, so it
  fires even if the mask-building logic drifts.
- **`iter_supervised_ids`** (encoding.py) is the single source of truth for the supervised tokens.
- Builders subclass **`DatasetBuilder`** and implement three pure hooks; `build()` is **final** and runs
  the gate; `register_builder` rejects any subclass that overrides `build`.
- The KL is computed in ONE place: `aligned_kl` / `supervised_kl_terms` in `bakery/objectives/base.py`
  (teacher = adapter-disabled, student = adapter-enabled, full vocab, on the gate-validated span).

## How to run

```bash
pip install -e ".[dev,analysis]"                      # one-time
python run.py --list                                  # registered experiments
python run.py --experiment bake_squad --model.lora_rank 16 --seed 0 --run_name bake-truth-r16
python run.py --experiment bake_squad --config configs/bake_squad.yaml
python run.py --sweep configs/sweeps/bake_lr_rank.yaml
python run.py --experiment bake_squad --print-config  # resolve + dump, don't run
```

Unknown `--section.key value` flags override config fields (`--model.lora_rank 16`,
`--generation.num_contexts 200`, `--train.learning_rate 1e-4`). **Unknown keys fail loudly.**

## How to test

```bash
make test          # full suite (pytest) — invariants + end-to-end smoke
make test-fast     # skips the slow model-loading smoke (the loop's gate)
make smoke         # tiny CPU end-to-end bake on a stub model (no GPU, no gating)
```

`tests/test_trajectory_contract.py` proves the gate can't be bypassed and every validator fires;
`tests/test_objective_alignment.py` proves the KL is on the supervised span via the disabled adapter.
**If a test fails, you broke an invariant — fix the code, not the test.** Every new builder/objective
ships its own test — enforced by `tests/test_builder_coverage.py` / `tests/test_objective_coverage.py`.

## Results-directory contract

Each run writes `results/<experiment>/<run_id>/`: `manifest.json` (git SHA, seeds, env, base-checkpoint
+ tokenizer fingerprint, generation provenance, gate `data_stats`, status), `config.json`,
`metrics.json` (parallel lists; headline `eval_kl`), `log.txt`, `data/trajectories.pt` +
`data/prompts/`, `checkpoints/` (PEFT adapters). **Analysis reads ONLY the three JSON files**
(`analysis/load.py`) — never the `.pt` and never the training stack (enforced by
`tests/test_analysis_isolation.py`).

## Add a baking experiment in 5 steps (`/scaffold-new-variant`)

1. **Objective (if new):** `bakery/objectives/<name>.py` — an `Objective` subclass + `@register_objective`,
   reusing `aligned_kl`/`supervised_kl_terms` (never re-roll the KL/masking). Declare its `sampler`.
2. **Builder (if new):** `bakery/trajectories/<name>.py` — a `DatasetBuilder` with the three pure hooks,
   ending (via the final `build()`) in the gate. `@register_builder`. NEVER override `build()`.
3. **Experiment:** `bakery/experiments/<name>.py` — a DataConfig + `register_experiment(...)` (with
   `defaults={...}` if it ships its own config). Auto-discovered — no import-list edit.
4. **Test:** add an alignment/contamination test (coverage meta-tests enforce this). `make test` + `make smoke`.
5. **Run:** `python run.py --experiment <name> ...`.

You write ZERO training/IO/logging/plotting code — the runner inherits all of it.

## Other safe extensions
- **New model:** any HF causal-LM id works through `bakery/models/peft_factory.py` (LoRA on its
  attention projections); set `--model.name`. The stub `hf-internal-testing/tiny-random-LlamaForCausalLM`
  is the CPU smoke model.
- **New metric:** `@register_metric(name)` in `bakery/eval/metrics/<name>.py` returning a `MetricResult`,
  then name it in the experiment's `extra_metrics` or `--eval.metrics`. Flows into metrics.json + analysis.
- **New generation backend:** implement `TrajectoryGenerator` (e.g. vLLM) behind the same interface —
  free, because generation stores ids, not logits.

## Context-rot rules (the project's discipline)
- Python modules: one responsibility per file, ≤ ~250 lines target. No god-files. No committed notebooks.
- Read SMALL artifacts (manifest/metrics/config JSON, the tail of run-log.jsonl), never weights / `.pt` /
  full logs. Plots are PNG files (report the path). `analysis/` never imports the training stack.
- Predictable paths + the registry (`python run.py --list`) over scanning the tree.
- Full details: `research/README.md → Context-budget doctrine`. (Stated once there; referenced here.)

## Autonomous researcher (`/research-loop`)
The repo runs as a long-running independent researcher; its durable brain is `research/` (committed).
The cycle: orient → select an open question → `/validate-bake` + design → `/bake-a-prompt` (background) →
`/analyze-run` → `make test-fast` gate → `/record-finding` → enqueue follow-ups → commit on a
`research/<slug>` branch → reschedule. Supporting skills: `/scaffold-new-variant`, `/lint-research`.

**Hard rules** (also restated as warnings by `.claude/hooks/`): always build data through the gate;
never hand-construct `TrajectoryDataset`; **never weaken the gate, the KL primitive, or `tests/`**
("fix the code, not the test"); never commit to `main`; never delete `results/` or `findings/`. If you
think an invariant is wrong, write a `research/decisions/` note for human review.

**Kill switch:** `touch research/STOP` halts at the next cycle; `rm research/STOP` resumes. A
`max_cycles_per_session` budget in `research/agenda.md` also bounds it. **Compute:** runs on the local GPU(s).

## Footguns
- Comparing the two sides on DIFFERENT base checkpoints (the gate raises BASE MISMATCH — keep it that way).
- A truncated-logit KL instead of the full vocab (don't add a top-k knob to the KL).
- A masking/shift error: the supervised span is the generated tokens; logit at `p-1` predicts token `p`
  (`ASSISTANT_SHIFT`). Get it from `objectives/base.py`; don't re-roll it.
- Eval contexts leaking into the bake distribution (the gate's context-disjointness raises).
- Hand-building `TrajectoryDataset` (it raises) or overriding `DatasetBuilder.build()` (`register_builder`
  rejects it).
- Editing a test to make it pass instead of fixing the code.
