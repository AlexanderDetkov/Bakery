---
name: record-finding
description: Record a baking result into the knowledge base — write/update research/findings/<slug>.md from the template, update the source open question's status, refresh research/index.md, and append a cycle block to research/log.md. Use after analyzing a run.
---

# /record-finding — commit a result to the brain

## Procedure
1. Copy `research/findings/_TEMPLATE.md` → `research/findings/<slug>.md`. Fill the frontmatter: `outcome` (positive|negative|inconclusive), `confidence`, `created`, `question: [[<slug>]]`, `run_ids: [...]`, `metric: eval_kl`.
2. Write the body:
   - **Insight** — one sentence.
   - **Evidence** — numbers + which run(s): final/best `eval_kl`, base-checkpoint revision, n contexts/trajectories, epochs-to-plateau.
   - **Counter-arguments / threats to validity** — **ALWAYS non-empty** (seed luck; too few contexts; eval too close to bake distribution; KL low but behavior diverges; LoRA-rank or temperature confound). This is the discipline that stops the loop fooling itself.
   - **Implications** and **Next steps**.
3. Update the source open question: set `resolved` (or keep `active` with a refined plan if it needs iteration).
4. Refresh `research/index.md` (add the finding; update the question's line).
5. Append a dated cycle block to `research/log.md` (question, run_id, outcome, one-line numbers).
6. Verify the runner wrote the `run-log.jsonl` row (it owns that file — do not hand-edit metrics into it).

## MUST NOT
Invent frontmatter or numbers. Overstate (always report final alongside best, and name the confounds). Skip the counter-arguments section. Edit `run-log.jsonl` headline metrics by hand.
