"""Typed run configuration with CLI / YAML overrides and sweep expansion.

Dataclasses are the source of truth. A run is configured by, in increasing priority:
the experiment's registered defaults -> an optional YAML file -> dotted CLI overrides
(``--model.lora_rank 16 --generation.num_contexts 200``). Unknown keys raise loudly,
so a typo can never silently no-op (the failure mode that let the reference repos'
launch scripts drift).
"""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass, field, fields
from itertools import product
from typing import Any, Optional

import yaml


# ======================================================================================
# Schema
# ======================================================================================

@dataclass
class ModelConfig:
    name: str = "meta-llama/Llama-3.1-8B-Instruct"
    revision: Optional[str] = None                  # pin the HF commit (reproducibility)
    dtype: str = "bfloat16"
    device: str = "cuda"
    lora_rank: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    adapter_to_load: Optional[str] = None           # prior adapter dir (sequential / knowledge baking)
    half_bake_alpha: float = 1.0                    # scale applied to the adapter at eval (half-baking)
    # SPEEDUP (opt-in). None = today's behavior (transformers resolves its own default attention impl).
    # Setting "sdpa"/"eager"/"flash_attention_2" changes ULP-level numerics → opt-in only.
    attn_implementation: Optional[str] = None


@dataclass
class GenerationConfig:
    base_prompt: str = "data/prompts/truth_u.md"    # the prompt u (system text), baked into weights
    baked_prompt: str = "data/prompts/empty.md"     # prompt the baked model sees (usually empty)
    context_dataset: str = "squad"                  # source of x0 contexts ("squad" | "synthetic")
    context_split: str = "train"
    num_contexts: int = 100
    trajectories_per_context: int = 8
    eval_num_contexts: int = 25                     # DISJOINT eval contexts (gate-enforced)
    max_new_tokens: int = 256
    min_new_tokens: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    batch_size: int = 16                            # generation batch size
    backend: str = "hf"                             # trajectory-generation backend ("hf" | future "vllm")
    cache_dir: str = "trajectory_cache"
    cache_enabled: bool = True
    on_the_fly: bool = False                        # regenerate each epoch w/o disk (pursuit)


@dataclass
class TrainConfig:
    objective: str = "bake"                         # registered Objective name (bake | pursue | ...)
    num_epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-4
    optimizer: str = "AdamW"
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    grad_accum: int = 1
    kl_reg: float = 0.0                             # weight on optional KL-to-base-prompt regularizer
    eval_period: int = 1                            # epochs between evals
    save_every: int = 0                             # 0 = save best + final only
    save_adapters: bool = True
    # SPEEDUP (opt-in; all default to today's exact path = recompute the teacher every step).
    # The teacher (frozen base + fixed prompt + fixed trajectory tokens) yields identical logits every
    # epoch, so they can be memoized. "off" = no cache (current behavior). gpu|cpu|memmap = where the
    # cached full-vocab float32 teacher log-probs live. Refused for objectives whose teacher changes per
    # epoch (pursue). See research/decisions/teacher-logit-cache.md.
    cache_teacher_logits: str = "off"               # off | gpu | cpu | memmap
    cache_teacher_verify_every: int = 0             # >0: recompute teacher every k steps, assert ~equal
    cache_teacher_verify_atol: float = 1e-2         # verify tolerance on log-probs (catches gross cache
                                                    # bugs while tolerating padding float-noise; bit-exact
                                                    # is unachievable on real batched/padded forwards)
    cache_teacher_dtype: str = "float32"            # matches the live path's float32 log_softmax
    # Invariant-safe alternative (stores NO logits): reuse the shared system+prompt KV across teacher
    # forwards. Off = current behavior. See Optimization B.
    cache_teacher_prefix_kv: bool = False


@dataclass
class EvalConfig:
    metrics: tuple = ("eval_kl",)                   # names resolved in the metric registry
    benchmarks: tuple = ()
    eval_with_base_prompt: bool = True              # also report baked-model-WITH-prompt (re-prompting)
    # SPEEDUP (opt-in). False = today's exact unbatched per-probe scoring loop. True = batch the probe
    # logprob forwards (same full-vocab float32 math; bit-exact in fp32, tol-equivalent + flip-free in
    # bf16). probe_batch_size 0 = all rows in one forward; >0 caps rows/forward for memory.
    batch_probes: bool = False
    probe_batch_size: int = 0


@dataclass
class RegularizationConfig:
    """Anchor the baked model to the ORIGINAL base model on irrelevant questions (anti-degeneration).

    Mixes "anchor" trajectories into training: teacher = base no-prompt adapter-OFF, student = baked
    no-prompt adapter-ON, supervised on tokens the BASE model generated (no prompt). The existing KL
    primitive then drives the adapter toward identity on those inputs. `num_train_contexts` is the
    strength knob (0 = OFF). `behavior_drift` measures held-out drift on a disjoint window of the same
    pool. Greedy reference (`do_sample=False`) keeps that metric comparable across eval periods.
    """
    num_train_contexts: int = 0      # strength: # anchor trajectories mixed into training (0 = OFF)
    eval_num_contexts: int = 16      # held-out anchors for the behavior_drift metric
    source: str = "squad"            # irrelevant-question source ("squad" | "synthetic")
    context_split: str = "validation"
    max_new_tokens: int = 40         # short base completions keep anchoring cheap
    do_sample: bool = False          # greedy => fixed base reference => stable drift across epochs
    seed: int = 0                    # context selection (+ sampling, if do_sample)


@dataclass
class RunConfig:
    experiment: str
    seed: int = 0
    model_seed: Optional[int] = None                # defaults to `seed`
    gen_seed: Optional[int] = None                  # defaults to `seed`
    output_root: str = "results"
    run_name: Optional[str] = None
    backend: str = "local"                          # where the run executed (recorded in the ledger)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    data: dict = field(default_factory=dict)        # experiment-specific; validated against DataConfig


# The dotted-override sections. `data` is handled separately (validated against the
# experiment's own DataConfig). Adding a section here is the only wiring a new
# top-level config group needs.
_SECTIONS = {
    "model": ModelConfig,
    "generation": GenerationConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
    "regularization": RegularizationConfig,
}


def config_to_dict(cfg: RunConfig) -> dict:
    return asdict(cfg)


# ======================================================================================
# Coercion + dotted overrides
# ======================================================================================

def _strip_optional(typ):
    if typing.get_origin(typ) is typing.Union:
        args = [a for a in typing.get_args(typ) if a is not type(None)]
        return args[0] if args else str
    return typ


def _is_tuple_type(typ) -> bool:
    return typ is tuple or typing.get_origin(typ) is tuple


def _coerce(value: Any, typ) -> Any:
    """Coerce a CLI string (or YAML scalar/list) to the field's declared type."""
    if not isinstance(value, str):
        if _is_tuple_type(typ):
            return tuple(value) if isinstance(value, (list, tuple)) else (value,)
        return value
    typ = _strip_optional(typ)
    if typ is bool:
        return value.strip().lower() in ("1", "true", "yes", "y", "t")
    if typ is int:
        return int(value)
    if typ is float:
        return float(value)
    if _is_tuple_type(typ):
        return tuple(s.strip() for s in value.split(",") if s.strip())
    return value


def _hints(dc) -> dict:
    return typing.get_type_hints(type(dc))


def _assert_field(dc_cls, name, dotted_key) -> None:
    valid = {f.name for f in fields(dc_cls)}
    if name not in valid:
        raise KeyError(
            f"Unknown config key {dotted_key!r}. Valid keys: {sorted(valid)}"
        )


def _set_dotted(run: RunConfig, dotted_key: str, value: Any, data_config_cls) -> None:
    parts = dotted_key.split(".")

    if len(parts) == 2 and parts[0] in _SECTIONS:
        section = getattr(run, parts[0])
        _assert_field(_SECTIONS[parts[0]], parts[1], dotted_key)
        setattr(section, parts[1], _coerce(value, _hints(section)[parts[1]]))
    elif len(parts) == 2 and parts[0] == "data":
        valid = {f.name for f in fields(data_config_cls)}
        if parts[1] not in valid:
            raise KeyError(
                f"Unknown data key {dotted_key!r}. Valid data keys for this experiment: "
                f"{sorted(valid)}"
            )
        typ = typing.get_type_hints(data_config_cls)[parts[1]]
        run.data[parts[1]] = _coerce(value, typ)
    elif (len(parts) == 1 and parts[0] in {f.name for f in fields(RunConfig)}
          and parts[0] not in _SECTIONS and parts[0] != "data"):
        typ = typing.get_type_hints(RunConfig)[parts[0]]
        setattr(run, parts[0], _coerce(value, typ))
    else:
        raise KeyError(f"Unknown config key {dotted_key!r}.")


def _flatten(d: dict) -> dict:
    """Flatten a nested config dict into dotted keys (one level: section.key)."""
    out: dict = {}
    sections = set(_SECTIONS) | {"data"}
    for k, v in d.items():
        if isinstance(v, dict) and k in sections:
            for kk, vv in v.items():
                out[f"{k}.{kk}"] = vv
        else:
            out[k] = v
    return out


def cli_list_to_overrides(tokens: list) -> dict:
    """Turn ['--generation.num_contexts', '200', '--train.save_adapters'] into a dotted dict.

    A `--flag` with no following value is treated as boolean true.
    """
    overrides: dict = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ValueError(f"Expected an option starting with '--', got {tok!r}.")
        key = tok[2:]
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            overrides[key] = tokens[i + 1]
            i += 2
        else:
            overrides[key] = "true"
            i += 1
    return overrides


# ======================================================================================
# Loading
# ======================================================================================

def load_config(experiment: str, cli_overrides: Optional[dict] = None,
                yaml_path: Optional[str] = None) -> RunConfig:
    """Resolve a RunConfig: experiment defaults -> YAML -> dotted CLI overrides."""
    import bakery.experiments  # noqa: F401 — ensure experiments are auto-discovered/registered
    from bakery.registry import get_experiment

    spec = get_experiment(experiment)
    data_config_cls = spec.data_config_cls

    run = RunConfig(experiment=experiment, data=asdict(data_config_cls()))
    if spec.objective:
        run.train.objective = spec.objective

    # Experiment-shipped defaults (lowest priority after the dataclass defaults).
    for k, v in _flatten(spec.defaults or {}).items():
        _set_dotted(run, k, v, data_config_cls)

    if yaml_path is not None:
        with open(yaml_path) as f:
            raw = yaml.safe_load(f) or {}
        raw.pop("experiment", None)
        for k, v in _flatten(raw).items():
            _set_dotted(run, k, v, data_config_cls)

    if cli_overrides:
        for k, v in cli_overrides.items():
            _set_dotted(run, k, v, data_config_cls)

    return run


def build_data_config(run: RunConfig, data_config_cls):
    """Instantiate the experiment's typed DataConfig from `run.data`."""
    return data_config_cls(**run.data)


# ======================================================================================
# Sweeps
# ======================================================================================

def expand_sweep(sweep: dict) -> list:
    """Expand a sweep spec into a list of (experiment, overrides_dict).

    Sweep YAML shape:
        experiment: bake_squad
        base:  { train: { num_epochs: 30 } }            # optional, nested or dotted
        grid:  { model.lora_rank: [8, 16, 32], seed: [0, 1] }
    """
    experiment = sweep["experiment"]
    base = _flatten(sweep.get("base", {}) or {})
    grid = sweep.get("grid", {}) or {}

    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]

    points = []
    for combo in product(*value_lists) if keys else [()]:
        overrides = dict(base)
        for k, v in zip(keys, combo):
            overrides[k] = v
        points.append((experiment, overrides))
    return points
