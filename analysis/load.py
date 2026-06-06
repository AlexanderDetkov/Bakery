"""Load a run from its standardized artifacts ONLY (manifest/config/metrics JSON).

This is the single reader the agent/analysis use. It imports nothing from the training
stack, so an analysis turn never drags torch/transformers/peft (or their import-time side
effects) into scope. `eval_kl` is the headline metric — LOWER is better.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Run:
    run_dir: Path
    manifest: dict
    config: dict
    metrics: dict

    @property
    def status(self) -> str:
        return self.manifest.get("status")

    @property
    def experiment(self) -> str:
        return self.manifest.get("experiment")

    def final(self, key: str):
        vals = self.metrics.get(key, [])
        return vals[-1] if vals else None

    def best(self, key: str, mode: str = "min"):
        vals = [v for v in self.metrics.get(key, []) if isinstance(v, (int, float))]
        if not vals:
            return None
        return min(vals) if mode == "min" else max(vals)


def load_run(run_dir) -> Run:
    p = Path(run_dir)

    def _read(name):
        f = p / name
        return json.loads(f.read_text()) if f.exists() else {}

    return Run(
        run_dir=p,
        manifest=_read("manifest.json"),
        config=_read("config.json"),
        metrics=_read("metrics.json"),
    )
