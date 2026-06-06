---
name: lint-research
description: Health-check the research/ knowledge base — find orphan pages, stale 'active' questions, broken [[links]], run-log↔finding cross-ref gaps, and findings missing counter-arguments — then fix what's safe and report the rest. Use periodically or when the index feels out of date.
---

# /lint-research — knowledge-base health check

Lightweight, read-mostly integrity pass over `research/` (all small text). Fix mechanical drift; flag judgement calls.

## Checks
1. **Orphans** — `findings/` or `open-questions/` pages not listed in `research/index.md`.
2. **Stale `active`** — questions marked `active` with no recent `log.md` activity AND no in-flight `run-log.jsonl` row (`status:"running"`). Suggest `open` or `parked`.
3. **Broken `[[links]]`** — `[[slug]]` references with no matching page.
4. **Run-log ↔ finding cross-refs** — every `run_ids` cited in a finding exists in `run-log.jsonl`; flag `status:"running"` rows older than a day (likely dead).
5. **Missing counter-arguments** — any finding whose "Counter-arguments / threats to validity" section is empty (a hard requirement).
6. **Settled-but-open** — `open` questions already answered by a finding.

## Output
Append a dated `lint` block to `research/log.md` listing what was fixed and what needs human judgement. Refresh `research/index.md` if you added/repaired entries.

## MUST NOT
Delete pages. Commit to `main`. "Repair" by inventing evidence or counter-arguments. Touch `results/` or the training stack.
