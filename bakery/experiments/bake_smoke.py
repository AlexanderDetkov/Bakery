"""Smoke experiment: the full pipeline on a tiny stub model + synthetic contexts, on CPU.

Self-contained via `defaults` so `python run.py --experiment bake_smoke` just runs (no GPU,
no gating, no dataset download). This is what `make smoke` and the slow end-to-end test use to
verify the whole gate -> generate -> bake -> eval -> results path.
"""

import bakery.trajectories.squad_qa as squad_qa  # noqa: F401 — registers SquadQABuilder
from bakery.registry import register_experiment

register_experiment(
    "bake_smoke",
    builder_name="squad_qa",
    data_config_cls=squad_qa.SquadQADataConfig,
    objective="bake",
    extra_metrics=("eval_kl",),
    defaults={
        "model.name": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "model.dtype": "float32",
        "model.device": "cpu",
        "model.lora_rank": 4,
        "model.lora_alpha": 8,
        "generation.context_dataset": "synthetic",
        "generation.num_contexts": 4,
        "generation.eval_num_contexts": 2,
        "generation.trajectories_per_context": 2,
        "generation.max_new_tokens": 8,
        "generation.min_new_tokens": 1,
        "generation.batch_size": 4,
        "generation.cache_enabled": False,
        "train.num_epochs": 2,
        "train.batch_size": 4,
        "train.eval_period": 1,
        "train.learning_rate": 1e-3,
    },
    description="Tiny CPU end-to-end smoke (stub model + synthetic contexts). Verifies the pipeline.",
)
