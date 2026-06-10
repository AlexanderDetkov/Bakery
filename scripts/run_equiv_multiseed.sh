#!/usr/bin/env bash
# Equivalence multi-seed hardening: eq_alpha × n∈{1,2} × seed∈{11,12} (sampled mode), to lift finding #20
# from 4-graphs-×-1-seed to seed-replicated (matching the directed paired-matched-seed protocol:
# split_seed=model_seed per arm). Fast recipe. GPUs 0,1,3 (GPU 2 busy with the clean teacher-forced run).
set -u
cd /workspace/Bakery
export HF_HOME=/workspace/.hf_home
mkdir -p logs

bake () {  # run_name gpu n seed
  local rn=$1 gpu=$2 n=$3 s=$4
  echo "[launch gpu$gpu] $rn  $(date '+%H:%M:%S')"
  CUDA_VISIBLE_DEVICES=$gpu python run.py --experiment bake_theorem_qa_equiv \
    --data.train_max_depth $n \
    --train.num_epochs 20 --train.eval_period 20 --train.batch_size 4 \
    --train.learning_rate 3e-4 --train.lr_schedule constant --train.cache_teacher_logits cpu \
    --generation.num_contexts 100 --data.split_seed $s --seed $s \
    --run_name $rn > logs/$rn.log 2>&1 &
}

bake eqms-eq_alpha-n1-s11 0 1 11; bake eqms-eq_alpha-n2-s11 1 2 11; bake eqms-eq_alpha-n1-s12 3 1 12; wait
echo "[wave1 done] $(date '+%H:%M:%S')"
bake eqms-eq_alpha-n2-s12 0 2 12; wait
echo "EQ MULTISEED COMPLETE $(date '+%Y-%m-%d %H:%M:%S')"
