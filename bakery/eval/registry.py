"""Metric registry + typed result object.

Add a metric with `@register_metric(name)` returning a `MetricResult` (or a flat dict), then
name it in the experiment's `extra_metrics` or the run's `eval.metrics`. It flows into
metrics.json, the log, and analysis automatically — no runner edit, no hand-enumerated keys.
`MetricResult` carries scalars, CIs, and matrices uniformly via `to_metrics()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class EvalContext:
    bundle: object        # ModelBundle
    data: object          # frozen TrajectoryDataset
    run_cfg: object
    device: str


@dataclass
class MetricResult:
    name: str
    value: Optional[float] = None
    stderr: Optional[float] = None
    ci95: Optional[tuple] = None
    matrix: Optional[list] = None
    labels: Optional[list] = None
    extra: dict = field(default_factory=dict)

    def to_metrics(self) -> dict:
        out: dict = {}
        if self.value is not None:
            out[self.name] = self.value
        if self.stderr is not None:
            out[f"{self.name}.stderr"] = self.stderr
        if self.ci95 is not None:
            out[f"{self.name}.ci95_lo"], out[f"{self.name}.ci95_hi"] = self.ci95
        if self.matrix is not None:
            out[f"{self.name}.matrix"] = self.matrix
        if self.labels is not None:
            out[f"{self.name}.labels"] = self.labels
        for k, v in self.extra.items():
            out[f"{self.name}.{k}"] = v
        return out


_METRICS: dict[str, Callable] = {}


def register_metric(name):
    def deco(fn):
        if name in _METRICS:
            raise ValueError(f"Duplicate metric {name!r}.")
        _METRICS[name] = fn
        return fn
    return deco


def get_metric(name):
    if name not in _METRICS:
        raise KeyError(f"Unknown metric {name!r}. Known: {sorted(_METRICS)}")
    return _METRICS[name]


def list_metrics() -> list:
    return sorted(_METRICS)


def run_metrics(names, ctx: EvalContext) -> dict:
    """Compute the named metrics and return a flat {key: value} dict for metrics.json."""
    out: dict = {}
    for name in names:
        res = get_metric(name)(ctx)
        if res is None:
            continue
        if isinstance(res, MetricResult):
            out.update(res.to_metrics())
        elif isinstance(res, dict):
            out.update(res)
        else:
            raise TypeError(f"Metric {name!r} must return MetricResult or dict, got {type(res)}.")
    return out
