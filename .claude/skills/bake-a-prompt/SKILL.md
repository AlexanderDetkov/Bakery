---
name: bake-a-prompt
description: Launch and monitor ONE prompt-baking run in the background (local GPU or opt-in vast.ai), recording it in research/run-log.jsonl. Use to execute a bake for the research loop or by hand.
---

# /bake-a-prompt — launch + monitor one bake

Execute one baking run without holding the conversation hostage, and ensure it lands in the committed ledger.

## Procedure
1. **Pre-flight:** run **`/validate-bake`** on the resolved config (same base checkpoint both sides, held-out eval, full logits + masking, concrete seeds/sampling). Fix the config if it FAILs; do not spend GPU on an invalid run.
2. **Stable run name:** `RUN_NAME=<variant>-<question-slug>-<short>` (e.g. `bake-truth-r16`). Artifacts land in `results/<experiment>/<RUN_NAME>/`.
3. **Launch in the background:**
   ```
   python run.py --experiment bake_squad \
     --generation.base_prompt data/prompts/<u>.md --generation.baked_prompt data/prompts/empty.md \
     --model.lora_rank 16 --seed 0 --run_name <RUN_NAME>
   ```
   The runner appends a `status:"running"` ledger row at start and flips it to `completed`/`failed` (with headline `eval_kl`) at end. You are re-invoked on completion → hand to **`/analyze-run`**.
4. **Remote (opt-in, big models):** `python vast/remote.py <same args>` rents a GPU, syncs results back, and always destroys the box. The ledger row gets `"backend":"vast"`.
5. **Progress check:** read ONLY `results/<exp>/<RUN_NAME>/metrics.json` (small). Never tail the full log.
6. **On failure to start / death:** say so; retry once or `park` the question. Never silently drop work.

## MUST NOT
Read adapter weights / `.safetensors` / `trajectories.pt` / the full `log.txt`. Run training in the foreground (it blocks the loop). Skip the `/validate-bake` pre-flight.
