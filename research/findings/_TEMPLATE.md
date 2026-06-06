---
title: <one-line claim the result supports>
outcome: positive       # positive | negative | inconclusive
confidence: medium      # high | medium | low
created: YYYY-MM-DD
question: [[<open-question-slug>]]
metric: eval_kl
run_ids: []             # run ids cited as evidence (rows in run-log.jsonl)
---

## Insight
<one-sentence takeaway>

## Evidence
<numbers + which run(s): final/best eval_kl, base-checkpoint revision, n contexts/trajectories,
epochs-to-plateau, lr/rank>

## Counter-arguments / threats to validity
<ALWAYS non-empty. e.g. seed luck; too few contexts; eval contexts too close to the bake distribution;
eval_kl low but behavior still diverges (objective-vs-payoff gap); LoRA-rank or temperature confound.
The cheap discipline that stops the loop fooling itself.>

## Implications
<what this means for the central idea / which variant to pursue next>

## Next steps
<follow-up open-questions this suggests>
