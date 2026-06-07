"""Regression tests for the 2026-06-07 code-review fixes (P0/P1/P2b).

P0  the base model must NOT be cached/shared across runs (only the tokenizer is) — else an
    in-process sweep inherits an earlier run's mutated/trained weights.
P1  the trajectory cache key must include the generating checkpoint, and the cache must store +
    round-trip that provenance, so the "same checkpoint" invariant holds for cached data.
P2b context dedup must happen AFTER truncation, so two contexts that collide under
    context_max_chars cannot land in both train and eval.
"""

import json
import random

import pytest

from bakery.models import peft_factory
from bakery.trajectories.base import CheckpointId
from bakery.trajectories.fact_propagation import FactPropagationDataConfig, _load_contexts
from bakery.trajectories.generator import (
    CacheIdentity,
    checkpoint_key,
    load_trajectories_jsonl,
    save_trajectories_jsonl,
)
from tests import _invariant_kit as kit


# --- P0: model not cached, tokenizer cached -------------------------------------------------

def test_base_model_loader_is_not_cached():
    # the tokenizer loader is memoized (read-only, safe to share); the MODEL loader is NOT, so
    # every build_bundle gets a fresh, unmutated base (no cross-run adapter/weight leakage).
    assert hasattr(peft_factory._load_tokenizer, "cache_info")
    assert not hasattr(peft_factory._load_base_model, "cache_info")


# --- P1: checkpoint provenance in the cache key + stored/round-tripped ----------------------

_BASE = dict(tokenizer_id="m", base_prompt_sha="a", baked_prompt_sha="b", template_sha=None,
             contexts_sha="c", sampling_sha="s", backend="hf")


def test_cache_key_depends_on_checkpoint():
    k1 = CacheIdentity(**_BASE, checkpoint="m@r1:bfloat16:None|adapter=").key()
    k2 = CacheIdentity(**_BASE, checkpoint="m@r2:bfloat16:None|adapter=").key()
    assert k1 != k2


def test_checkpoint_key_sensitive_to_revision_dtype_adapter():
    a = CheckpointId(model_name="m", revision="r1", dtype="bfloat16")
    assert checkpoint_key(a) != checkpoint_key(CheckpointId(model_name="m", revision="r2", dtype="bfloat16"))
    assert checkpoint_key(a) != checkpoint_key(CheckpointId(model_name="m", revision="r1", dtype="float16"))
    assert checkpoint_key(a) != checkpoint_key(a, adapter_to_load="prior_adapter")


def test_cache_stores_and_roundtrips_generation_checkpoint(tmp_path):
    p = tmp_path / "c.jsonl"
    meta = {"generation_checkpoint": {"model_name": "m", "revision": None,
                                      "dtype": "float32", "weights_sha256": None}}
    save_trajectories_jsonl(p, [kit.make_traj(0, (10, 11))], [kit.make_traj(1, (12,))], meta=meta)
    train, eval_, m2 = load_trajectories_jsonl(p)
    assert len(train) == 1 and len(eval_) == 1                       # rows still load
    assert CheckpointId(**m2["generation_checkpoint"]).dtype == "float32"   # provenance recovered


def test_cache_load_tolerates_missing_meta(tmp_path):
    # a file without a meta row (legacy) loads with meta == {}
    p = tmp_path / "legacy.jsonl"
    p.write_text(json.dumps({"split": "train", "base_input_ids": [5, 10], "base_sup_mask": [0, 1],
                             "baked_input_ids": [7, 10], "baked_sup_mask": [0, 1],
                             "x0_id": 0, "num_supervised": 1}) + "\n")
    train, eval_, meta = load_trajectories_jsonl(p)
    assert len(train) == 1 and meta == {}


# --- P2b: truncate before dedup -------------------------------------------------------------

def _bank(tmp_path, texts):
    p = tmp_path / "ctx.json"
    p.write_text(json.dumps({"contexts": [{"category": "mixed", "text": t} for t in texts]}))
    return str(p)


def test_contexts_dedup_after_truncation(tmp_path):
    # two contexts identical in their first 10 chars, differing only after
    texts = ["AAAAAAAAAA-one is here", "AAAAAAAAAA-two is here"]
    cfg = FactPropagationDataConfig(context_bank=_bank(tmp_path, texts),
                                    context_category="mixed", context_max_chars=10)
    # truncation collapses both to "AAAAAAAAAA" -> dedup to ONE -> requesting 2 is impossible
    with pytest.raises(ValueError, match="Need 2"):
        _load_contexts(cfg, 2, random.Random(0))
    # control: no truncation -> the two stay distinct
    cfg2 = FactPropagationDataConfig(context_bank=_bank(tmp_path, texts),
                                     context_category="mixed", context_max_chars=600)
    assert len(_load_contexts(cfg2, 2, random.Random(0))) == 2
