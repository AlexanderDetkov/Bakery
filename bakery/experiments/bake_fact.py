"""Knowledge-propagation experiment: bake a single new FACT u and measure how far it propagates.

Declaration only. Uses the gated `fact_propagation` builder (trajectories generated from
categorized elicitation contexts) and the `bake` objective; reports `eval_kl` (bake fidelity)
plus `propagation` (the research payload: per-hop forced-choice belief shift for prior vs
prompted vs baked, all on ONE checkpoint). Defaults to the cheap Llama-3.2-1B-Instruct for fast
iteration; override --model.name meta-llama/Llama-3.1-8B-Instruct for the strong-prior headline.
"""

import bakery.trajectories.fact_propagation as fact_propagation  # noqa: F401 — registers the builder
from bakery.registry import register_experiment

register_experiment(
    "bake_fact",
    builder_name="fact_propagation",
    data_config_cls=fact_propagation.FactPropagationDataConfig,
    objective="bake",
    extra_metrics=("eval_kl", "propagation"),
    defaults={
        "model": {"name": "meta-llama/Llama-3.2-1B-Instruct", "lora_rank": 16, "lora_alpha": 16},
        "generation": {
            "base_prompt": "data/prompts/tsunami_u.md",
            "baked_prompt": "data/prompts/empty.md",
            "num_contexts": 30,
            "trajectories_per_context": 4,
            "eval_num_contexts": 8,
            "max_new_tokens": 160,
            "temperature": 1.0,
        },
        "train": {"num_epochs": 15, "batch_size": 8, "learning_rate": 1e-4, "eval_period": 1},
        "eval": {"metrics": ["eval_kl", "propagation"]},
        "data": {
            "context_bank": "data/contexts/tsunami_contexts.json",
            "context_category": "mixed",
            "probe_bank": "data/probes/tsunami_probes.json",
        },
    },
    description="Bake one new FACT into a LoRA and measure n-hop knowledge propagation (prior vs "
                "prompted vs baked) with a no-CoT forced-choice belief readout on held-out probes.",
)
