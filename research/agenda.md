# Research agenda

North star for the autonomous loop. **Human-editable.** The loop treats this as read-mostly
priorities; it adds work to `open-questions/`, never edits the goals here without a `decisions/` note.

## Central idea
Prompt Baking converts a prompt `u` into LoRA weights `θ_u` so the UNPROMPTED baked model mimics
the PROMPTED base model:

    B(θ, u) = argmin_θ_u  D_KL( P_θ(·|u) ‖ P_θ_u(·) )

via base-checkpoint trajectory generation + token-level KL distillation into a LoRA adapter.
Headline metric: **`eval_kl`** (the held-out distillation objective; LOWER is better). The payoff is
behavioral equivalence of (baked model, no prompt) vs (base model, prompt `u`) on UNSEEN contexts.

## Validity criteria (every run must satisfy; enforced by the gate + tests)
1. Paired comparison on an IDENTICAL base checkpoint (same model id + revision for both sides).
2. Full-vocab logits in the KL (no top-k truncation) — structural: Bakery never stores logits.
3. Generated-token masking — KL on EXACTLY the generated tokens, pad/eos excluded.
4. Held-out eval contexts, disjoint from the trajectory-generation contexts.
5. Recorded: sampling params (temperature/top_p/top_k/max_new_tokens), seeds, all hyperparameters.

## Current focus (priority order)
1. Single-prompt baking fidelity — does `eval_kl → low` hold on held-out contexts, and at what LoRA rank / lr?
2. Pursuit (iterative re-generation) — does it reach lower achievable `eval_kl` than one-shot baking?
3. Knowledge baking (sequential composition) — does baking `u1` then `u2` retain `u1`?
4. Half-baking / re-prompting — partial-strength baking and its dynamics.

## Guardrails for the loop
Gate-only trajectories (via a registered builder's `build()`); never hand-construct the frozen
`TrajectoryDataset`; never weaken the gate, the KL primitive, or `tests/`; never commit to `main`;
never delete `results/` or `findings/`. If the gate seems wrong, write a `decisions/` note for human
review. Prefer cheap, decisive runs. Record negative/inconclusive results as findings too.

## Compute budget
- `max_cycles_per_session: 6`   (loop halts + writes a wrap-up when reached)
- compute: local GPU default; opt-in vast.ai (`--backend vast` / `vast/remote.py`) for large-model bakes.
- single-run guard: keep exploratory bakes small (a few hundred trajectories, modest epochs, a small
  model e.g. Llama-3.2-1B/3B); reserve 8B+ and long bakes for questions a small run already promised.
