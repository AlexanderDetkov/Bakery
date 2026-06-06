---
name: scaffold-new-variant
description: Scaffold a new baking variant (objective), trajectory builder, or experiment safely, through the validation gate — following CLAUDE.md's "add a baking experiment in 5 steps". Generates the declaration + an invariant test, then runs make test + make smoke. Use when a research question needs a new variant/experiment.
---

# /scaffold-new-variant — add a variant through the gate

The repo writes the training/IO/logging for you. You add only declarations + a test. The gate makes a methodologically-invalid run un-constructable; never go around it.

## What kind of addition?
- **New objective** (pursue / knowledge / a new loss): add `bakery/objectives/<name>.py` with an `Objective` subclass + `@register_objective`. Reuse `aligned_kl` / `supervised_kl_terms` from `objectives/base.py` — do NOT re-roll the KL or the masking. Declare its required `sampler`.
- **New trajectory source**: add a `DatasetBuilder` subclass in `bakery/trajectories/<name>.py` implementing the three pure hooks. Do NOT override `build()` (it is final and runs the gate); `@register_builder` rejects overrides.
- **New experiment**: add `bakery/experiments/<name>.py` — a DataConfig + `register_experiment(builder_name=..., objective=..., defaults={...})`. Auto-discovered (no import-list edit).

## The 5 steps (read CLAUDE.md → "Add a baking experiment in 5 steps" first)
1. Read ONE existing example (`objectives/bake.py`, `trajectories/squad_qa.py`, `experiments/bake_squad.py`).
2. Write the declaration(s). Keep each file one-responsibility, ≤ ~250 lines.
3. Write the test:
   - new objective → add a case to `tests/test_objective_alignment.py` (KL on the supervised span; base via disabled adapter). `tests/test_objective_coverage.py` fails the suite if an objective has no test.
   - new builder → add `tests/test_<name>.py` exercising the gate (mask alignment, context disjointness). `tests/test_builder_coverage.py` enforces this.
4. `make test` (invariants) and `make smoke` (end-to-end on the stub). Both green.
5. Register it in an `open-questions/` plan and run it via `/bake-a-prompt`.

## MUST NOT
Override `build()` / weaken the gate / hand-construct `TrajectoryDataset`. Re-implement the KL or masking (use `aligned_kl`). Write training/IO/logging/plotting code (it is inherited). Add a variant without a test. "Fix the test instead of the code."
