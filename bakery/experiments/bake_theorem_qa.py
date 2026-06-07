"""Knowledge-propagation via teacher-forced QA: train depth-1..n declarative answers (FULL axiom
coverage) and measure HELD-OUT propagation by PROOF DEPTH with the bias-immune d′ readout.

Uses the gated `theorem_qa` builder. Every axiom is trained as "Yes, every X is a Y." (coverage by
construction + a hard gate criterion); the closure ("only these rules") is carried by "No, not..."
negatives; depths 2..n add a balanced sampled subset and the rest + depths > n are held out — the
propagation test. The TRAINING question matches the EVAL question (`bakery.logic.phrasing`), so the
baked yes/no decision transfers and the depth-1 recall baseline is finally cleared (the old
declarative-completion runs left it at d′≈0 — a format mismatch, not a propagation result).

Soft-bake vs one-hot-SFT is one flag: `--train.objective {bake|sft}`. Curriculum depth is
`--data.train_max_depth` (n; sweep `configs/sweeps/theorem_qa_n_arm.yaml` over {1,2,3}). Swap the
world via the two data.* paths. Defaults to the cheap 1B for iteration; override
`--model.name meta-llama/Llama-3.1-8B-Instruct` for the headline. NOTE the expected bake<SFT
asymmetry (the prompted teacher itself only propagates ~1 hop, so bake is teacher-ceiling-limited
while SFT's ground-truth labels can teach deeper) is a finding, not a bug.
"""

import bakery.trajectories.theorem_qa as theorem_qa  # noqa: F401 — registers the builder
from bakery.registry import register_experiment

register_experiment(
    "bake_theorem_qa",
    builder_name="theorem_qa",
    data_config_cls=theorem_qa.TheoremQADataConfig,
    objective="bake",
    extra_metrics=("eval_kl", "dprime", "behavior_drift"),
    defaults={
        "model": {"name": "meta-llama/Llama-3.2-1B-Instruct", "lora_rank": 16, "lora_alpha": 16},
        "generation": {
            "base_prompt": "data/prompts/lw_alpha_u.md",
            "baked_prompt": "data/prompts/empty.md",
            # bake samples y from the prompted teacher (canonical baking); short answers keep the QA
            # sampling cheap and reduce the chance a sampled answer rambles into a held-out relation.
            "max_new_tokens": 32,
            "trajectories_per_context": 4,   # samples per relation question (teacher estimate density)
            "do_sample": True,
        },
        "train": {"num_epochs": 100, "batch_size": 8, "learning_rate": 1e-4,
                  "eval_period": 2, "save_every": 20, "save_adapters": True},
        "eval": {"metrics": ["eval_kl", "dprime"]},
        # Regularization OFF by default (num_train_contexts=0). Turn on with e.g.
        # --regularization.num_train_contexts 64 to anchor general behavior to the base on SQuAD.
        "regularization": {"num_train_contexts": 0, "eval_num_contexts": 16,
                           "source": "squad", "context_split": "validation",
                           "max_new_tokens": 40, "do_sample": False, "seed": 0},
        "data": {
            "world_spec": "data/worlds/lw_alpha.json",
            "probe_bank": "data/probes/lw_alpha_qa_probes.json",
            "train_max_depth": 1,
            "per_depth_train_cap": 16,
            "split_seed": 0,
        },
    },
    description="Teacher-forced QA baking of a logic world's axioms (full coverage) + a depth-1..n "
                "curriculum; held-out propagation by proof depth via a bias-immune d′. bake vs SFT = "
                "--train.objective; curriculum depth n = --data.train_max_depth (sweep {1,2,3}).",
)
