"""Aggregate runs under a results/ tree into a table (group-by config knobs).

Reads only artifacts. Examples:
    python -m analysis.aggregate results/bake_squad --group-by model.lora_rank train.learning_rate
    # low-variance propagation readout across arms (note: --mode final; status any includes stopped runs):
    python -m analysis.aggregate results/bake_theorem_qa --group-by data.train_max_depth seed \
        --metric dprime.auroc_baked_d2,dprime.prop_distance_bacc_baked --mode final --status any
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


def aggregate(root, group_by=(), metric="eval_kl", mode="best", status="completed") -> list:
    """One row per run. `metric` may be a comma-list → one column each. `status`: 'completed' (default)
    or 'any' (include stopped/running runs whose metrics.json is still valid — e.g. early-stopped bakes).
    For higher-is-better readouts (auroc/bacc/dprime) use --mode final; 'best' uses min (eval_kl convention).
    """
    metrics = [m.strip() for m in metric.split(",") if m.strip()]
    rows = []
    for run in iter_runs(root):
        if status != "any" and run.status != "completed":
            continue
        row = {"run_id": run.manifest.get("run_id"), "status": run.status}
        for g in group_by:
            row[g] = _dig(run.config, g)
        for m in metrics:
            row[f"{mode}_{m}"] = run.best(m) if mode == "best" else run.final(m)
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--group-by", nargs="*", default=[])
    ap.add_argument("--metric", default="eval_kl", help="metric key, or comma-list of keys")
    ap.add_argument("--mode", choices=["best", "final"], default="best")
    ap.add_argument("--status", choices=["completed", "any"], default="completed",
                    help="'any' includes stopped/running runs with a valid metrics.json")
    a = ap.parse_args(argv)
    print(json.dumps(aggregate(a.root, a.group_by, a.metric, a.mode, a.status), indent=2, default=str))


if __name__ == "__main__":
    main()
