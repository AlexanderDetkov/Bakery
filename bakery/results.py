"""The results-directory contract.

Every run writes:
    results/<experiment>/<run_id>/
        manifest.json     reproducibility envelope (git SHA, seeds, env, base checkpoint,
                          tokenizer fingerprint, generation provenance, gate stats, status)
        config.json       the fully-resolved RunConfig
        metrics.json      dict of parallel metric lists keyed by metric name
        log.txt           human-readable training log
        data/trajectories.pt, data/prompts/{base_u,baked,template}.txt
        checkpoints/...   PEFT adapter directories

Analysis reads ONLY manifest.json / config.json / metrics.json — never the .pt or
the training stack. The committed run-ledger (research/run-log.jsonl) is the durable
record that survives the gitignored results/ tree.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from bakery.config import config_to_dict

SCHEMA_VERSION = 1


def _git_info() -> dict:
    def _run(args):
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()

    try:
        return {
            "sha": _run(["git", "rev-parse", "HEAD"]),
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_run(["git", "status", "--porcelain"])),
        }
    except Exception:
        return {"sha": None, "branch": None, "dirty": None}


def _env_info(device: str) -> dict:
    def _ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": _ver("transformers"),
        "peft": _ver("peft"),
        "cuda": torch.version.cuda,
        "device": device,
    }


def _maybe_asdict(obj):
    return asdict(obj) if is_dataclass(obj) else obj


def make_run_id(run_cfg) -> str:
    if run_cfg.run_name:
        return run_cfg.run_name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{run_cfg.experiment}-{ts}-s{run_cfg.seed}-{uuid.uuid4().hex[:6]}"


def prepare_run_dir(run_cfg, run_id: str) -> Path:
    run_dir = Path(run_cfg.output_root) / run_cfg.experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config_to_dict(run_cfg), indent=2))
    return run_dir


def write_manifest(run_dir: Path, *, run_cfg, run_id, data, num_trainable_params, status,
                   metrics_schema, sweep_id=None) -> None:
    """Assemble the reproducibility envelope from the frozen TrajectoryDataset + config."""
    m = run_cfg.model
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": run_cfg.experiment,
        "run_id": run_id,
        "sweep_id": sweep_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_info(),
        "env": _env_info(m.device),
        "seed": run_cfg.seed,                 # top-level run seed (criterion 5: reproducibility)
        "seeds": {
            "data": data.seeds.data_seed,
            "model": data.seeds.model_seed,
            "generation": data.seeds.gen_seed,
        },
        # --- identity of the paired comparison (the trustworthiness core) ---
        "base_checkpoint": _maybe_asdict(data.base_checkpoint_id),
        "tokenizer": _maybe_asdict(data.tokenizer_fingerprint),
        "generation": _maybe_asdict(data.spec),
        # --- objective / adapter lineage ---
        "objective": run_cfg.train.objective,
        "lora": {
            "rank": m.lora_rank, "alpha": m.lora_alpha, "dropout": m.lora_dropout,
            "target_modules": list(m.target_modules),
        },
        "adapter_to_load": m.adapter_to_load,
        "half_bake_alpha": m.half_bake_alpha,
        "num_trainable_params": int(num_trainable_params),
        # --- proof every validator ran ---
        "data_stats": data.stats,
        "builder": data.builder_name,
        "status": status,
        "metrics_schema": list(metrics_schema),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def update_manifest_status(run_dir: Path, status: str) -> None:
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = status
    path.write_text(json.dumps(manifest, indent=2, default=str))


def save_data_artifacts(run_dir: Path, data) -> None:
    """Persist trajectories (token ids + masks; NO logits) and the exact prompt texts."""
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_trajectories": list(data.train_trajectories),
            "eval_trajectories": list(data.eval_trajectories),
            "spec": data.spec,
        },
        data_dir / "trajectories.pt",
    )
    prompts = getattr(data, "prompts", {}) or {}
    if prompts:
        pdir = data_dir / "prompts"
        pdir.mkdir(parents=True, exist_ok=True)
        for name, text in prompts.items():
            if text is not None:
                (pdir / f"{name}.txt").write_text(text)


def write_metrics(run_dir: Path, history: dict) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(history, indent=2, default=str))


# ======================================================================================
# The committed run-ledger (research/run-log.jsonl)
#
# One append-only JSON line per run that the autonomous research loop reads to know
# "what has been baked and how it went" without scanning every manifest.json. It lives
# at the git toplevel (NOT under the gitignored results/) so the knowledge survives.
# ======================================================================================

def run_log_path(output_root: str | None = None) -> Path:
    """Resolve the committed run-ledger path at the repo's git toplevel.

    Honors the ``BAKERY_RUN_LOG`` env var (tests point it at a temp file so they never
    touch the committed ledger). Falls back to ``<output_root>/..`` / cwd outside git.
    """
    override = os.environ.get("BAKERY_RUN_LOG")
    if override:
        return Path(override)
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
        ).decode().strip()
        root = Path(top)
    except Exception:
        root = Path(output_root).resolve().parent if output_root else Path.cwd()
    return root / "research" / "run-log.jsonl"


def append_run_log(entry: dict, *, output_root: str | None = None) -> None:
    path = run_log_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def update_run_log(run_id: str, *, output_root: str | None = None, **fields) -> None:
    """Merge ``fields`` into the most-recent ledger line whose run_id matches.

    No-op if the ledger or the run_id is absent (so a failed start never crashes a run).
    """
    path = run_log_path(output_root)
    if not path.exists():
        return
    lines = path.read_text().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if not lines[i].strip():
            continue
        try:
            row = json.loads(lines[i])
        except json.JSONDecodeError:
            continue
        if row.get("run_id") == run_id:
            row.update(fields)
            lines[i] = json.dumps(row, default=str)
            path.write_text("\n".join(lines) + "\n")
            return


def headline_from_metrics(metrics: dict) -> dict:
    """The ledger's at-a-glance numbers. eval_kl is the headline; LOWER is better."""
    out = {}
    kl = [v for v in metrics.get("eval_kl", []) if isinstance(v, (int, float))]
    if kl:
        out["final_eval_kl"] = kl[-1]
        out["best_eval_kl"] = min(kl)
    gap = [v for v in metrics.get("held_out_eval_gap", []) if isinstance(v, (int, float))]
    if gap:
        out["final_held_out_eval_gap"] = gap[-1]
    return out
