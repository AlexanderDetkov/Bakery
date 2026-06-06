"""End-to-end smoke: a tiny CPU bake produces complete, contract-conformant artifacts.

Marked `slow` (loads a tiny stub model from HF the first time). This is the verification that
the whole pipeline — gate -> generate -> bake -> eval -> results — works as one piece.
"""

import json

import pytest

from bakery.config import load_config
from bakery.runner import run


@pytest.mark.slow
def test_smoke_end_to_end(tmp_path):
    cfg = load_config("bake_smoke")
    cfg.output_root = str(tmp_path / "results")
    cfg.run_name = "smoke-test"
    run_dir = run(cfg)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["objective"] == "bake"
    assert manifest["data_stats"]["n_eval_ctx"] >= 1
    assert manifest["base_checkpoint"]["model_name"].endswith("tiny-random-LlamaForCausalLM")

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "eval_kl" in metrics and len(metrics["eval_kl"]) >= 1
    assert "train_kl" in metrics and len(metrics["train_kl"]) >= 1

    config = json.loads((run_dir / "config.json").read_text())
    assert config["train"]["objective"] == "bake"

    assert (run_dir / "checkpoints" / "final").exists()
    assert (run_dir / "data" / "trajectories.pt").exists()
