"""Experiment registry.

An experiment binds a name to: a registered DATA BUILDER (which produces the
validated, frozen TrajectoryDataset via the gate), the experiment's typed
DataConfig, the default OBJECTIVE (bake / pursue / ...), and the names of any
extra metrics. The runner looks an experiment up by name and never branches on
experiment type beyond that.

Adding an experiment = drop a module in `bakery/experiments/` that calls
`register_experiment(...)`. The package auto-imports it (no import-list edit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class BakingExperimentSpec:
    name: str
    builder_name: str              # registered DatasetBuilder (trajectories registry)
    data_config_cls: type          # the experiment's typed DataConfig dataclass
    objective: str = "bake"        # default registered Objective (overridable via --train.objective)
    defaults: dict = field(default_factory=dict)  # config defaults this experiment ships (dotted keys)
    eval_fn: Optional[Callable] = None
    extra_metrics: tuple = ()
    description: str = ""


_REGISTRY: dict[str, BakingExperimentSpec] = {}


def register_experiment(name, *, builder_name, data_config_cls, objective="bake",
                        defaults=None, eval_fn=None, extra_metrics=(), description=""):
    if name in _REGISTRY:
        raise ValueError(f"Experiment {name!r} already registered.")
    spec = BakingExperimentSpec(
        name=name,
        builder_name=builder_name,
        data_config_cls=data_config_cls,
        objective=objective,
        defaults=dict(defaults or {}),
        eval_fn=eval_fn,
        extra_metrics=tuple(extra_metrics),
        description=description,
    )
    _REGISTRY[name] = spec
    return spec


def get_experiment(name) -> BakingExperimentSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown experiment {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_experiments() -> list:
    return sorted(_REGISTRY)
