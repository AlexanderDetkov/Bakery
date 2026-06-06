---
title: <one-line question>
status: open            # open | active | resolved | parked
priority: medium        # high | medium | low
created: YYYY-MM-DD
hypothesis: <specific, testable claim>
acceptance_criteria: <numeric threshold, e.g. "best eval_kl < 0.05 on held-out contexts">
experiment: <experiment + key overrides, e.g. bake_squad --generation.base_prompt data/prompts/truth_u.md --model.lora_rank 16>
links: []               # [[finding-slug]] / [[decision-slug]] / related questions
---

## Question
<what we want to know and why it matters for the central idea>

## Plan
<the concrete bake(s) to run; mark "heavy" if it needs an 8B+ model or a long bake>

## Notes
<scratch; updated as the loop works the question>
