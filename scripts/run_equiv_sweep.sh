#!/usr/bin/env bash
# Equivalence-world sweep: 4 eq graphs × n∈{1,2}, fast recipe (lr 3e-4 const + teacher cache),
# mirroring the directed graph-lw_* sweep exactly so the directed-vs-equivalence comparison is clean.
# Runs in waves of 3 on GPUs 0,1,2 (GPU 3 reserved for the finishing grok arm). Sharp test of
# "cross-component over-connection is baking's mechanism": eq worlds make CROSS the only false family.
set -u
cd /workspace/Bakery
export HF_HOME=/workspace/.hf_home
mkdir -p logs

bake () {  # graph n gpu
  local g=$1 n=$2 gpu=$3
  echo "[launch gpu$gpu] eqg-$g-n$n  $(date '+%H:%M:%S')"
  CUDA_VISIBLE_DEVICES=$gpu python run.py --experiment bake_theorem_qa_equiv \
    --data.world_spec data/worlds/$g.json \
    --data.probe_bank data/probes/${g}_qa_probes.json \
    --generation.base_prompt data/prompts/${g}_u.md \
    --data.train_max_depth $n \
    --train.num_epochs 20 --train.eval_period 20 --train.batch_size 4 \
    --train.learning_rate 3e-4 --train.lr_schedule constant --train.cache_teacher_logits cpu \
    --generation.num_contexts 100 --data.split_seed 10 \
    --run_name eqg-$g-n$n > logs/eqg-$g-n$n.log 2>&1 &
}

# Wave 1 — n=1 across alpha/beta/gamma
bake eq_alpha 1 0; bake eq_beta 1 1; bake eq_gamma 1 2; wait
echo "[wave1 done] $(date '+%H:%M:%S')"
# Wave 2 — delta n=1, then start n=2
bake eq_delta 1 0; bake eq_alpha 2 1; bake eq_beta 2 2; wait
echo "[wave2 done] $(date '+%H:%M:%S')"
# Wave 3 — remaining n=2
bake eq_gamma 2 0; bake eq_delta 2 1; wait
echo "[wave3 done] $(date '+%H:%M:%S')"

echo "EQ SWEEP COMPLETE $(date '+%Y-%m-%d %H:%M:%S')"
