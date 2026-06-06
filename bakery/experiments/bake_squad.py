"""Baseline experiment: bake a system prompt u into a LoRA adapter over SQuAD contexts.

Declaration only — zero training/IO/logging code. Uses the gated `squad_qa` builder and the
`bake` objective; the runner does the rest. Defaults target Llama-3.1-8B-Instruct (the paper
model); override anything from the CLI/YAML.
"""

import bakery.trajectories.squad_qa as squad_qa  # noqa: F401 — registers SquadQABuilder
from bakery.registry import register_experiment

register_experiment(
    "bake_squad",
    builder_name="squad_qa",
    data_config_cls=squad_qa.SquadQADataConfig,
    objective="bake",
    extra_metrics=("eval_kl",),
    description="Bake a system prompt into a LoRA adapter by KL-distilling from the prompted "
                "Llama-3.1-8B base over held-out SQuAD question contexts.",
)
