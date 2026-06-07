"""Coverage + pure-logic tests for the squad_qa builder (network-free).

The full generate -> gate path for squad_qa is exercised by the slow end-to-end smoke
(test_runner_smoke). Here we test the parts that need no model: the generation spec and the
train/eval context split.
"""

import random

import pytest

from bakery.config import RunConfig
import bakery.trajectories.squad_qa as squad_qa
from bakery.trajectories.squad_qa import SquadQABuilder, SquadQADataConfig, _load_contexts


def test_generation_spec_sampler_and_distinct_shas():
    b = SquadQABuilder()
    b.gen_seed = 0
    cfg = RunConfig(experiment="x")
    cfg.generation.base_prompt = "Be a truthful assistant."   # inline text (load_prompt falls back)
    cfg.generation.baked_prompt = ""
    spec = b.build_generation_spec(cfg)
    assert spec.sampler == "base_disable_adapter"
    assert spec.base_prompt_sha256 != spec.baked_prompt_sha256
    assert spec.dataset_id.startswith("squad:")


def test_contexts_split_unique():
    g = RunConfig(experiment="x").generation
    g.context_dataset = "synthetic"
    pool = _load_contexts(g, SquadQADataConfig(context_max_chars=600), n_needed=6, rng=random.Random(0))
    assert len(pool) == 6
    assert len(set(pool)) == 6                       # unique contexts (no truncation collisions)


def test_contexts_truncated():
    g = RunConfig(experiment="x").generation
    g.context_dataset = "synthetic"
    pool = _load_contexts(g, SquadQADataConfig(context_max_chars=12), n_needed=6, rng=random.Random(0))
    assert all(len(c) <= 12 for c in pool)           # truncation respected


def test_contexts_dedup_after_truncation(monkeypatch):
    # Two raw questions that differ only after the truncation boundary must not become identical
    # train/eval contexts under different x0_ids.
    monkeypatch.setattr(squad_qa, "SYNTHETIC_QUESTIONS", [
        "AAAAAAAAAA-one is here",
        "AAAAAAAAAA-two is here",
    ])
    g = RunConfig(experiment="x").generation
    g.context_dataset = "synthetic"
    with pytest.raises(ValueError, match="Need 2"):
        _load_contexts(g, SquadQADataConfig(context_max_chars=10), n_needed=2, rng=random.Random(0))

    pool = _load_contexts(g, SquadQADataConfig(context_max_chars=600), n_needed=2, rng=random.Random(0))
    assert len(pool) == 2
