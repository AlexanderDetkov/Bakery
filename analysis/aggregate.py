"""Aggregate completed runs under a results/ tree into a table (group-by config knobs).

Reads only artifacts. Example:
    python -m analysis.aggregate results/bake_squad --group-by model.lora_rank train.learning_rate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.load import load_run


def iter_runs(root):
    for manifest in sorted(Path(root).rglob("manifest.json")):
        yield load_run(manifest.parent)


def _dig(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def aggregate(root, group_by=(), metric="eval_kl", mode="best") -> list:
    rows = []
    for run in iter_runs(root):
        if run.status != "completed":
            continue
        row = {"run_id": run.manifest.get("run_id")}
        for g in group_by:
            row[g] = _dig(run.config, g)
        row[f"{mode}_{metric}"] = run.best(metric) if mode == "best" else run.final(metric)
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--group-by", nargs="*", default=[])
    ap.add_argument("--metric", default="eval_kl")
    ap.add_argument("--mode", choices=["best", "final"], default="best")
    a = ap.parse_args(argv)
    print(json.dumps(aggregate(a.root, a.group_by, a.metric, a.mode), indent=2, default=str))


if __name__ == "__main__":
    main()
