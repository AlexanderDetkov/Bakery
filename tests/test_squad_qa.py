"""Coverage + pure-logic tests for the squad_qa builder (network-free).

The full generate -> gate path for squad_qa is exercised by the slow end-to-end smoke
(test_runner_smoke). Here we test the parts that need no model: the generation spec and the
train/eval context split.
"""

import random

from bakery.config import RunConfig
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
