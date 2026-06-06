"""Knowledge-propagation as a Hilbert-style proof system: bake the atomic AXIOMS of a logic world
and measure how far the fact propagates by PROOF DEPTH, with a bias-immune signal-detection readout.

Declaration only. Uses the gated `fact_propagation` builder with a `world_spec` (a DAG of definite
implications) + `source_control="atomic"`, so the baking support = the atomic axioms and any
composed/cross/reverse relation a probe tests is genuinely held out (the DAG contamination guard
enforces it). Reports `eval_kl` (bake fidelity), `propagation` (per-depth belief shift / coverage),
and the headline `dprime` (per-depth d′ over depth-matched true/false probes — propagation that is
not just a Yes/No bias). Defaults to the cheap 1B for iteration; override
`--model.name meta-llama/Llama-3.1-8B-Instruct` for the headline. Soft-bake vs one-hot-SFT is a
single flag: `--train.objective {bake|sft}`. Swap the world via the four data.* paths
(data/worlds|prompts|contexts|probes/lw_{alpha,beta,gamma,delta}*).
"""

import bakery.trajectories.fact_propagation as fact_propagation  # noqa: F401 — registers the builder
from bakery.registry import register_experiment

register_experiment(
    "bake_logic",
    builder_name="fact_propagation",
    data_config_cls=fact_propagation.FactPropagationDataConfig,
    objective="bake",
    extra_metrics=("eval_kl", "propagation", "dprime"),
    defaults={
        "model": {"name": "meta-llama/Llama-3.2-1B-Instruct", "lora_rank": 16, "lora_alpha": 16},
        "generation": {
            "base_prompt": "data/prompts/lw_alpha_u.md",
            "baked_prompt": "data/prompts/empty.md",
            "num_contexts": 30,
            "trajectories_per_context": 4,
            "eval_num_contexts": 8,
            "max_new_tokens": 64,
            "temperature": 1.0,
        },
        "train": {"num_epochs": 15, "batch_size": 8, "learning_rate": 1e-4, "eval_period": 1},
        "eval": {"metrics": ["eval_kl", "propagation", "dprime"]},
        "data": {
            "context_bank": "data/contexts/lw_alpha_contexts.json",
            "context_category": "atomic",
            "probe_bank": "data/probes/lw_alpha_probes.json",
            "world_spec": "data/worlds/lw_alpha.json",
            "source_control": "atomic",
        },
    },
    description="Bake the atomic axioms of a propositional logic world (DAG) and measure n-hop "
                "knowledge propagation by PROOF DEPTH: prior/prompted/soft-bake/one-hot-SFT, with a "
                "bias-immune d′ readout over depth-matched true/false held-out probes.",
)
