# research/ — the autonomous researcher's durable brain

All durable state lives here (committed to git), NOT in chat transcripts — so the loop survives
context resets and is reviewable via `git log -- research/`. The `/research-loop` skill drives it.

## Layout
- `agenda.md` — human-owned north-star: central idea, the 5 validity criteria, compute budget.
- `open-questions/` — the task queue (one `.md` per question) + `_TEMPLATE.md`.
- `findings/` — results, each with evidence + **counter-arguments** + `_TEMPLATE.md`.
- `decisions/` — methodology log (why an invariant/policy is what it is) + `_TEMPLATE.md`.
- `run-log.jsonl` — the committed run ledger; ONE JSON line per run, written by the runner.
- `log.md` — append-only cycle journal (one block per cycle).
- `index.md` — the catalog (kept current; the agent orients from this, not by scanning dirs).
- `STOP` — kill switch (gitignored): `touch research/STOP` halts the loop at the next cycle; `rm` resumes.

## The cycle (one per `/research-loop` invocation)
orient → select an open question → design (+ `/validate-bake`) → `/bake-a-prompt` (background) →
`/analyze-run` → `make test-fast` gate → `/record-finding` → enqueue follow-ups → commit on a
`research/<slug>` branch → reschedule. Supporting skills: `/scaffold-new-variant`, `/lint-research`.

## Context-budget doctrine (read before every cycle)
The whole point of the results contract is that you judge a run from a few KB of JSON, never from
big files. **NEVER** read into context:
- model weights, adapter `.safetensors`, or `results/**/data/trajectories.pt` (raw teacher tokens);
- a full `results/**/log.txt` (stat its mtime / read the last ~20 lines if you must check liveness);
- a `.ipynb` (notebooks are banned from the repo);
- the training stack from an analysis turn (`analysis/` reads only manifest/config/metrics JSON).
**DO** read: `manifest.json` / `metrics.json` / `config.json`, the TAIL of `log.md` / `run-log.jsonl`,
and `research/*.md`. Prefer `--print-config` over reconstructing a config from several YAMLs.

## run-log.jsonl schema (written by the runner)
`run_id, experiment, objective, backend, host, device, model, base_prompt, baked_prompt,
config_digest, data_stats, status (running|completed|failed), started_utc, finished_utc,
best_eval_kl, final_eval_kl, run_dir`.
