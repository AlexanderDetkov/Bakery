---
name: research-loop
description: Autonomous prompt-baking researcher. Runs ONE research cycle (orient → select an open question → design → launch+monitor a bake → analyze → gate via tests → record a finding → enqueue follow-ups → commit) then schedules the next. Use to start/continue long-running independent baking research.
---

# /research-loop — the autonomous researcher

You run **one cycle**, then schedule the next. All durable state lives in `research/` — re-derive
context by reading files, NOT from chat history. Read `research/README.md` once if unsure of the layout.

Work on a `research/<slug>` branch (create/switch if on `main`/`master` — never commit to the protected branch). Keep each cycle small and decisive.

## Step 0 — Stop / budget check (FIRST)
- If `research/STOP` exists → append a wrap-up to `research/log.md` and **do not reschedule**. End.
- Read `research/agenda.md → Compute budget`. Count cycles run this session (from `log.md`). If `max_cycles_per_session` is reached → wrap up and **do not reschedule**.

## Step 1 — Orient
Read: `research/agenda.md`, `research/index.md`, the tail of `research/log.md`, the last lines of `research/run-log.jsonl`, and the `open-questions/` with `status: open|active`. Recall relevant harness memory.

## Step 2 — Select (or generate) a question
- Pick the highest-`priority`, **actionable** `open` question (one with a concrete `experiment` + `acceptance_criteria`). Set it `active`.
- If none are actionable, synthesize 1–3 new `open-questions/` from `findings/` + the agenda, then pick one.

## Step 3 — Design
Turn it into a concrete bake: a `bake_squad` config with overrides, or a `configs/sweeps/*.yaml`. Run **`/validate-bake`** on the resolved config first. If it needs a NEW variant (pursuit/knowledge/half-bake) or builder, invoke **`/scaffold-new-variant`** and confirm `make test` is green before running. Keep exploratory bakes small (a few hundred trajectories, modest epochs, a small model) per the agenda budget.

## Step 4 — Run
Invoke **`/bake-a-prompt`** with the experiment + overrides + a stable `--run_name`. It launches in the background; you are re-invoked when it finishes. Append a one-line "launched" note to `log.md`.
- **Don't wait forever.** If not re-invoked in a reasonable window, read `results/<exp>/<run_name>/metrics.json` — if missing or its mtime hasn't advanced across several eval periods, kill it, log it, and `park` the question.

## Step 5 — Analyze
Invoke **`/analyze-run`** on the `run_dir`: read `metrics.json`/`manifest.json` only, compute the headline `eval_kl` (final + best), compare to the question's `acceptance_criteria`, note convergence/instability and confounds.

## Step 6 — Gate
Run `make test-fast`. If **red**, do NOT record or commit and do NOT "fix the test": write an `open-questions/` regression item, append to `log.md`, and stop (no reschedule).

## Step 7 — Record
Invoke **`/record-finding`**: write/update `research/findings/<slug>.md` (insight, evidence with run ids + numbers, **counter-arguments**, implications), set the question `resolved` (or keep `active` with a refined plan), refresh `index.md`, append a cycle block to `log.md`. The `run-log.jsonl` row is written by the runner — verify it landed.

## Step 8 — Enqueue follow-ups
Add 1–3 `open-questions/` the finding suggests (ablations, larger N, controls, new prompts/variants).

## Step 9 — Commit & reschedule
- Commit this cycle on the `research/<slug>` branch (`git add research/ <new experiment files>`; never `main`).
- Schedule the next cycle, **never double-schedule**: under the harness `/loop` → `ScheduleWakeup` with `prompt:"/research-loop"`; standalone → `CronCreate` one-shot. If neither is available, log that the loop is paused pending manual re-invocation.

## Hard rules
- Integrity rules live in **CLAUDE.md → Autonomous researcher → Hard rules** (gate-only data; never hand-construct `TrajectoryDataset`; never weaken the gate / KL primitive / tests; never commit to `main`; never delete results/findings). They bind every cycle.
- Record negative/inconclusive results as findings too.
- After 2 consecutive failed designs on one question, `park` it with an escalation note.

## MUST NOT
Read model weights / `.safetensors` / `trajectories.pt` / full `log.txt` / notebooks into context. Block the turn waiting on a run. Trust prior chat over the files.
