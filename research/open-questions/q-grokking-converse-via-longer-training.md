---
title: Is baking's converse failure the REVERSAL CURSE — learnable late via grokking (long training + weight decay) and/or compositional "path" trajectories?
status: resolved        # DECISIVE-NEGATIVE (2026-06-10): no late transition to 10k ep; converse installed EARLY -> [[no-grokking-converse-installed-early]]
priority: high
created: 2026-06-06
hypothesis: >
  Baking's chain-specific converse failure (Veld 8B baked converse-reject 0.00 while forward 0.93;
  [[CORRECTED-picture-robust-metric]]) is the REVERSAL CURSE expressed in LoRA-distillation form. The
  sister project ~/Invertibility found the inverse map IS learnable but only (a) with training far past
  the point the train loss plateaus (GROKKING — and grokking there is transient/regularization-driven)
  and (b) on PATHS = compositions of forward+inverse edges, not isolated forward edges; the curse there
  scales with coverage density, not capacity. Mapped onto baking: eval_kl (the distillation loss) plateaus
  early while the converse generalization may keep moving late — exactly the "eval_kl converges before
  belief does" signature we already saw ([[size-helps-fidelity-not-the-propagation-gap]]) but never pushed
  on (prior bakes were 15-40 epochs = pre-grok). So the converse should become learnable by (i) training
  ~30-80x longer, (ii) adding weight decay (the classic grokking driver), and/or (iii) baking on
  converse/"path" trajectories that exercise the reverse direction (which forward-only trajectories cannot).
acceptance_criteria: >
  Report propagation.converse_acc_baked vs epoch with eval_kl overlaid (analysis/plot_grokking.py),
  on Veld 8B. DECISIVE-POSITIVE if converse_acc_baked rises materially above its short-training ~0 value
  AFTER eval_kl has plateaued, for some arm (wd>0 in E1, or converse/path trajectory type in E2) — i.e.
  the converse groks. DECISIVE-NEGATIVE if converse_acc_baked stays ~0 to 1200 epochs across wd in {0,0.05}
  AND across trajectory types {forward, converse, path, mixed}, despite eval_kl -> ~0 — i.e. a low-rank
  additive adapter trained to copy single-pass behavior cannot represent the converse (a real LoRA/objective
  limit, not undertraining). Either way, characterize WHERE the eval_kl plateau sits relative to any
  converse transition.
experiment: >
  E1 (running): bake_fact Veld 8B, 1200 epochs, eval_period 15, save_every 300, trajectory_category=mixed,
  arms wd=0 (e1a, GPU0) vs wd=0.05 (e1b, GPU1), batch_size 2 grad_accum 2 (8B OOMs at batch 4 on the long
  Veld prompt). E2 (queued): same long recipe, trajectory_category in {forward, converse, path} from
  data/contexts/veld_directional_contexts.json + mixed, matched count. Instrument: propagation per-form
  accuracy (forward/converse/negation x prior/prompted/baked) now logged per eval step.
links: [[CORRECTED-picture-robust-metric]], [[tokenization-artifact-corrected-prompting-is-directional]], [[size-helps-fidelity-not-the-propagation-gap]], [[contrastive-trajectories-do-not-fix-the-converse]], [[trajectory-type-is-a-binary-coverage-gate]]
---

## Why now / what's new vs prior attempts
- [[contrastive-trajectories-do-not-fix-the-converse]] tried converse-eliciting trajectories and they did
  NOT fix the converse — BUT that was (a) under the broken space-only metric and (b) at 15-40 epochs
  (pre-grok). This question re-tests with the tokenization-robust metric AND grokking-length training AND
  an explicit weight-decay arm — the three levers ~/Invertibility says are required.
- The cross-project bridge: ~/Invertibility/.../findings/reversal-curse-scales-with-N-not-capacity
  (reversal collapses 82%->2% with domain size at fixed seen-pairs; capacity doesn't rescue it; pathless
  T=1 never generalizes, path T>=2 does; grokking is transient). Treat that as the toy-model theory; this
  question tests whether it transfers to fact-baking on a pretrained 8B.

## Teacher audit (do before trusting E2)
The path/converse trajectory types only carry rejection signal if the prompted 8B teacher actually
GENERATES converse-rejection on those contexts. Free-generate teacher continuations on
veld_directional_contexts.json (converse + path) and confirm they reject the converse before/while baking
(reconciles the tension between "teacher generates converse-affirmation" in the contrastive finding and
"prompting rejects converse 4/5" in the corrected finding — likely context-phrasing dependent).

## Notes
- If the converse groks, this is the headline positive: baking CAN install logical direction with the right
  training, and the prior "direction-blind" framing was an undertraining artifact.
- If it does not, pair with E4 ([[q-sft-vs-bake-reversal-curse]]) to localize the limit (data vs objective).
