---
title: Does the truthful-assistant prompt bake to low held-out eval_kl?
status: open
priority: high
created: 2026-06-06
hypothesis: KL-distilling the truthful-assistant system prompt (data/prompts/truth_u.md) into a rank-16 LoRA adapter drives eval_kl on held-out SQuAD contexts below 0.05 within ~20 epochs, without the prompt at inference.
acceptance_criteria: best eval_kl < 0.05 on held-out contexts AND eval_kl still decreasing or plateaued (not diverging) by the final epoch.
experiment: bake_squad --generation.base_prompt data/prompts/truth_u.md --generation.baked_prompt data/prompts/empty.md --model.lora_rank 16 --generation.num_contexts 100 --generation.eval_num_contexts 25 --train.num_epochs 20 --seed 0
links: []
---

## Question
The foundational sanity check for the whole programme: can we bake a simple, well-specified
system prompt into a LoRA adapter so the unprompted model reproduces the prompted base model's
next-token distribution on contexts it never saw during baking?

## Plan
Run the experiment above on the local GPU (Llama-3.1-8B-Instruct). If 8B is too heavy for a first
pass, first confirm on a small model (`--model.name meta-llama/Llama-3.2-3B-Instruct`) to derisk,
then run 8B. Watch the eval_kl curve for convergence vs the train_kl curve (overfitting check).

## Notes
First real run after the smoke. Establishes the rank/lr operating point for the focus questions.
