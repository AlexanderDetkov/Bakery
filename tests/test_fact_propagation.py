"""Coverage + pure-logic tests for the fact_propagation builder (network-free).

The full generate -> gate path is exercised by the end-to-end smoke / real bakes. Here we test
the parts that need no model: category filtering of the context bank, the train/eval split, and
the generation spec (sampler + that it carries the probe_bank for the propagation metric).
"""

import random

import pytest

from bakery.config import RunConfig
from bakery.trajectories.fact_propagation import (
    FactPropagationBuilder,
    FactPropagationDataConfig,
    _load_contexts,
    _source_filter_sha,
)
from bakery.logic.world import Rule, make_world

BANK = "data/contexts/tsunami_contexts.json"


def test_category_filter_and_unique_split():
    cfg = FactPropagationDataConfig(context_bank=BANK, context_category="neutral")
    pool = _load_contexts(cfg, n_needed=10, rng=random.Random(0))
    assert len(pool) == 10
    assert len(set(pool)) == 10                      # unique, no truncation collisions


def test_mixed_has_more_than_a_single_category():
    rng = random.Random(0)
    mixed = _load_contexts(FactPropagationDataConfig(context_bank=BANK, context_category="mixed"),
                           n_needed=40, rng=rng)
    assert len(mixed) == 40                          # mixed pools all categories


def test_truncation_respected():
    cfg = FactPropagationDataConfig(context_bank=BANK, context_category="mixed", context_max_chars=15)
    pool = _load_contexts(cfg, n_needed=10, rng=random.Random(0))
    assert all(len(c) <= 15 for c in pool)


def test_too_few_contexts_raises():
    cfg = FactPropagationDataConfig(context_bank=BANK, context_category="restate")
    with pytest.raises(ValueError):
        _load_contexts(cfg, n_needed=10_000, rng=random.Random(0))


def test_unknown_category_raises():
    cfg = FactPropagationDataConfig(context_bank=BANK, context_category="does-not-exist")
    with pytest.raises(ValueError):
        _load_contexts(cfg, n_needed=1, rng=random.Random(0))


def test_generation_spec_sampler_and_carries_probe_bank():
    b = FactPropagationBuilder()
    b.gen_seed = 0
    cfg = RunConfig(experiment="x")                  # run.data == {} -> dataconfig defaults apply
    cfg.generation.base_prompt = "A new fact about the world."   # inline text
    cfg.generation.baked_prompt = ""
    spec = b.build_generation_spec(cfg)
    assert spec.sampler == "base_disable_adapter"
    assert spec.base_prompt_sha256 != spec.baked_prompt_sha256
    assert spec.extra["probe_bank"].endswith(".json")
    assert spec.extra["context_category"] in spec.dataset_id


def test_source_filter_cache_key_tracks_filter_controls():
    # Atomic filtering changes the realized train/eval trajectories after generation; stale caches must
    # not be reused when those controls change.
    cfg = FactPropagationDataConfig(source_control="atomic", oversample=2)
    w1 = make_world("w", ["A", "B"], [Rule(("A",), "B")])
    w2 = make_world("w", ["A", "B", "C"], [Rule(("A",), "B"), Rule(("B",), "C")])

    base = _source_filter_sha(cfg, chain=[], world=w1)
    assert base != _source_filter_sha(FactPropagationDataConfig(source_control="atomic", oversample=3),
                                      chain=[], world=w1)
    assert base != _source_filter_sha(cfg, chain=[], world=w2)
    assert base != _source_filter_sha(FactPropagationDataConfig(source_control="free", oversample=2),
                                      chain=[], world=w1)
    assert base != _source_filter_sha(cfg, chain=["A", "B"], world=None)
