---
title: A single-token "Final answer:" cue readout is artifactual for CoT propagation; the CoT-vs-no-CoT comparison is inconclusive (this round)
outcome: inconclusive       # the CoT question is NOT answered; the durable output is a confirmed methodological flaw + the no-CoT result stands
confidence: high            # the cue-only ablation cleanly confirms the artifact; the "reasoning" conclusions are what's inconclusive
created: 2026-06-06
question: [[q-propagation-cot-confound]]
metric: propagation
run_ids: [prop-veld-8b-mixed, prop-veld-1b-mixed]
---

## Insight
The CoT-vs-no-CoT readout built this cycle (generate a greedy rationale, then score logP(Yes)−logP(No)
of a single token after a "Final answer (Yes or No):" cue) is **artifactual**: a cue-only ablation (zero
rationale tokens) reproduces almost the entire apparent "CoT deflation", so the cue/scoring format — not
reasoning — drives the effect. The CoT comparison is therefore **inconclusive about reasoning** this round;
the user's central CoT question stays open pending a corrected readout. The cycle-3 NO-CoT finding is unaffected.

## Method + the diagnostic
Reused the cycle-3 Veld adapters (no re-bake): `bakery/eval/cot_probe.py` scores prior/prompted/baked on the
30 Veld probes both no-CoT (immediate forced choice) and "CoT" (greedy ≤100-token rationale → cue → score).
A 3-skeptic adversarial pass flagged the comparison as confounded; a **cue-only ablation** (`--max_cot_tokens 0`,
no rationale at all) was the clean test.

## Evidence (8B, forward-entail pos=Yes mean belief; cue-only = no rationale)
- prior:    no-CoT +2.07 → **cue-only −5.63** → full-CoT −6.91
- prompted: no-CoT +10.42 → **cue-only +1.42** → full-CoT +2.48
- baked:    no-CoT +7.01 → **cue-only −0.51** → full-CoT +2.06
- **Smoking gun:** the hop-0 *prompted* probes ask about a rule stated VERBATIM in context ("is every Plonk a
  Marn?"); no-CoT +14.3 → **cue-only +2.5** (an ~12-nat collapse with ZERO reasoning) → full-CoT +1.5. Reasoning
  cannot "thoughtfully decline" a fact printed two lines above; the cue alone causes the collapse. (3 CoT scores
  were exactly 0.0 — the cue lands where Yes/No are equiprobable.)
- Consequence: no-CoT and CoT live on **different scales** (the shift is large AND state-dependent), so absolute
  no-CoT-vs-CoT belief comparisons are invalid — only the rationale DELTA (full-CoT − cue-only) isolates reasoning.

## What (weakly) survives the artifact
- **Rationale delta (full-CoT − cue-only), controlling for the cue:** baked +2.57, prompted +1.06, prior −1.28
  — i.e. reasoning helps the baked model slightly MORE than prompting, but this is small and n-noisy (3/cell);
  NOT a headline.
- **CoT does NOT repair baking's converse deficit** (the one within-readout-valid CoT comparison): the
  baking-induced converse-affirming deficit (prior−baked) is +4.75 no-CoT and +6.44 under CoT (persists/grows;
  vs-prompted +4.74→+3.15, change not significant). Suggestive that reasoning aloud does not undo baking's
  directional corruption — but it rests on the artifactual readout + n=5 converse, so demoted to suggestive.

## Counter-arguments / threats to validity
- The whole CoT arm is contaminated by the cue artifact; do NOT cite no-CoT-vs-CoT absolute gaps as evidence.
- n=3–5 per (state × probe-type × depth), single greedy rationale (no self-consistency), single chain/seed.
- The original `cot_probe.json` discarded the rationale text (now persisted) — first runs were unauditable.
- Even the "rationale delta" controls only for THIS cue; a different cue/parse could give a different delta.

## Implications
Methodological lesson for the program: to measure CoT effects, do NOT score a forced single token after an
answer cue — **parse the model's actual Yes/No from free generation** (or report the rationale-delta over a
cue-only baseline), and **always run a cue-only ablation** to bound the readout's intrinsic bias. The robust
no-CoT results ([[baking-is-associative-prompting-is-directional]], [[propagation-bounded-by-trajectory-coverage]])
are unaffected — the no-CoT forced choice IS a clean single-pass internal readout. The user's CoT question
(does reasoning aloud rescue baked/1B failures?) is still OPEN and needs the corrected eval.

## Next steps
- Rebuild the CoT readout ([[q-propagation-cot-confound]] stays active): free-generate the answer and parse
  Yes/No from the text (regex/first-token-after-"answer"), with k-sample self-consistency; keep the cue-only
  ablation as the bias baseline; persist rationales (now done). Then re-ask: does CoT rescue the 8B baked
  converse error and the 1B's depth failures?
- Code fix landed this cycle: `cot_probe.py` now supports `--max_cot_tokens 0` (cue-only) and persists rationales.
