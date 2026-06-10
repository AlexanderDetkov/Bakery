#!/usr/bin/env bash
# Teacher-forced CLEAN arm: removes the sampled-CoT leak (gate ENFORCES context-disjointness when
# sample_trajectories=False) so held-out d2/d3 are pure propagation, not recall-of-recited. De-confounds
# the central cross-over-connection finding for BOTH the directed (lw_alpha) and equivalence (eq_alpha)
# worlds, n in {1,2}. Same fast recipe (lr 3e-4 const + cache). One wave on GPUs 0-3.
set -u
cd /workspace/Bakery
export HF_HOME=/workspace/.hf_home
mkdir -p logs

bake () {  # experiment run_name gpu n
  local exp=$1 rn=$2 gpu=$3 n=$4
  echo "[launch gpu$gpu] $rn  $(date '+%H:%M:%S')"
  CUDA_VISIBLE_DEVICES=$gpu python run.py --experiment $exp \
    --data.sample_trajectories false \
    --data.train_max_depth $n \
    --train.num_epochs 20 --train.eval_period 20 --train.batch_size 4 \
    --train.learning_rate 3e-4 --train.lr_schedule constant --train.cache_teacher_logits cpu \
    --generation.num_contexts 100 --data.split_seed 10 \
    --run_name $rn > logs/$rn.log 2>&1 &
}

bake bake_theorem_qa       tf-lw_alpha-n1 0 1
bake bake_theorem_qa       tf-lw_alpha-n2 1 2
bake bake_theorem_qa_equiv tf-eq_alpha-n1 2 1
bake bake_theorem_qa_equiv tf-eq_alpha-n2 3 2
wait
echo "TEACHERFORCED CLEAN COMPLETE $(date '+%Y-%m-%d %H:%M:%S')"
