#!/usr/bin/env bash
# Extend the LEAK-FREE (teacher-forced) confirmation across eq graphs: eq_{beta,gamma,delta}-n1.
# The gate enforces train/eval disjointness in teacher-forced mode, so it will BUILD only the cells
# with a naturally-disjoint split (eq_alpha-n1 already passed) and refuse the rest — that's expected;
# whichever build give multi-graph leak-free cross-FA numbers. Zero dev; fast recipe. GPUs 0,1,2.
set -u
cd /workspace/Bakery
export HF_HOME=/workspace/.hf_home
mkdir -p logs

bake () {  # graph gpu
  local g=$1 gpu=$2
  echo "[launch gpu$gpu] tf-$g-n1  $(date '+%H:%M:%S')"
  CUDA_VISIBLE_DEVICES=$gpu python run.py --experiment bake_theorem_qa_equiv \
    --data.world_spec data/worlds/$g.json \
    --data.probe_bank data/probes/${g}_qa_probes.json \
    --generation.base_prompt data/prompts/${g}_u.md \
    --data.sample_trajectories false \
    --data.train_max_depth 1 \
    --train.num_epochs 20 --train.eval_period 20 --train.batch_size 4 \
    --train.learning_rate 3e-4 --train.lr_schedule constant --train.cache_teacher_logits cpu \
    --generation.num_contexts 100 --data.split_seed 10 \
    --run_name tf-$g-n1 > logs/tf-$g-n1.log 2>&1 &
}

bake eq_beta 0; bake eq_gamma 1; bake eq_delta 2; wait
echo "TF EQ GRAPHS COMPLETE $(date '+%Y-%m-%d %H:%M:%S')"
