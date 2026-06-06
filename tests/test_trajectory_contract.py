"""The gate cannot be bypassed, and every validator fires.

If a test here fails, you broke an invariant — fix the code, not the test.
"""

import pytest

from bakery.seeding import SeedBundle
from bakery.trajectories.base import (
    CheckpointId,
    DatasetBuilder,
    TokenizerFingerprint,
    TrajectoryDataset,
    register_builder,
)
from bakery.trajectories.encoding import FramedTrajectory, iter_supervised_ids
from tests import _invariant_kit as kit


# --- the gate is the ONLY producer ----------------------------------------------------

def test_hand_construction_raises():
    with pytest.raises(RuntimeError):
        TrajectoryDataset(
            spec=kit.make_spec(), train_trajectories=(), eval_trajectories=(),
            tokenizer_fingerprint=kit.DEFAULT_TOK, base_checkpoint_id=kit.DEFAULT_CKPT,
            builder_name="x", config_snapshot={}, seeds=SeedBundle(0, 0, 0),
            prompts={}, stats={},
        )


def test_register_builder_rejects_build_override():
    class Bad(DatasetBuilder):
        name = "bad_override"

        def fingerprints(self, cfg, *, tokenizer, base_model):
            ...

        def build_generation_spec(self, cfg):
            ...

        def build_trajectories(self, cfg, *, tokenizer, base_model):
            ...

        def build(self, cfg, **kw):   # noqa: D401 — illegal override
            return "bypassed"

    with pytest.raises(TypeError):
        register_builder(Bad)


# --- a valid dataset constructs and reports sane stats --------------------------------

def test_valid_dataset_builds():
    data = kit.valid_dataset()
    assert isinstance(data, TrajectoryDataset)
    assert data.stats["n_train_traj"] == 2
    assert data.stats["n_eval_traj"] == 1
    assert data.stats["n_train_ctx"] == 2
    assert data.stats["n_eval_ctx"] == 1
    assert data.stats["pairing_opted_out"] is True
    assert data.stats["sampler"] == "base_disable_adapter"


def test_iter_supervised_ids_aligned():
    t = kit.make_traj(0, (10, 11, 12))
    base, baked = iter_supervised_ids(t)
    assert base == [10, 11, 12] == baked


# --- A. mask alignment -----------------------------------------------------------------

def test_mask_alignment_token_mismatch():
    t = FramedTrajectory(
        base_input_ids=(5, 10, 11), base_sup_mask=(False, True, True),
        baked_input_ids=(7, 10, 99), baked_sup_mask=(False, True, True),
        x0_id=0, num_supervised=2,
    )
    with pytest.raises(AssertionError, match="differ between base and baked"):
        kit.gate([t], [kit.make_traj(1, (12,))])


def test_mask_alignment_count_mismatch():
    t = FramedTrajectory(
        base_input_ids=(5, 10, 11), base_sup_mask=(False, True, True),
        baked_input_ids=(7, 10), baked_sup_mask=(False, True),
        x0_id=0, num_supervised=2,
    )
    with pytest.raises(AssertionError):
        kit.gate([t], [kit.make_traj(1, (12,))])


def test_mask_alignment_empty_span():
    t = FramedTrajectory(
        base_input_ids=(5, 6), base_sup_mask=(False, False),
        baked_input_ids=(7, 8), baked_sup_mask=(False, False),
        x0_id=0, num_supervised=0,
    )
    with pytest.raises(AssertionError):
        kit.gate([t], [kit.make_traj(1, (12,))])


def test_mask_alignment_pad_in_span():
    # pad_id == 0; put it inside the supervised span.
    t = kit.make_traj(0, (10, 0, 12))
    with pytest.raises(AssertionError, match="pad token"):
        kit.gate([t], [kit.make_traj(1, (13,))])


def test_mask_alignment_span_starts_at_zero():
    # supervised span begins at position 0 -> no token precedes it (shift undefined).
    t = FramedTrajectory(
        base_input_ids=(10, 11), base_sup_mask=(True, True),
        baked_input_ids=(10, 11), baked_sup_mask=(True, True),
        x0_id=0, num_supervised=2,
    )
    with pytest.raises(AssertionError, match="position 0"):
        kit.gate([t], [kit.make_traj(1, (13,))])


# --- B. context disjointness -----------------------------------------------------------

def test_context_contamination():
    train = [kit.make_traj(0, (10, 11))]
    eval_ = [kit.make_traj(0, (12, 13))]   # SAME x0_id in train and eval
    with pytest.raises(AssertionError, match="CONTEXT CONTAMINATION"):
        kit.gate(train, eval_)


# --- C. completeness / shape -----------------------------------------------------------

def test_token_out_of_range():
    t = kit.make_traj(0, (10, 999))        # 999 >= vocab_size 64
    with pytest.raises(AssertionError, match="out of range"):
        kit.gate([t], [kit.make_traj(1, (11,))])


# --- E. identity of the paired comparison ----------------------------------------------

def test_base_checkpoint_mismatch():
    other = CheckpointId(model_name="DIFFERENT", revision=None, dtype="float32")
    with pytest.raises(AssertionError, match="BASE MISMATCH"):
        kit.gate([kit.make_traj(0, (10,))], [kit.make_traj(1, (11,))], gen_ckpt=other)


def test_tokenizer_mismatch():
    other = TokenizerFingerprint(
        name="OTHER", revision=None, vocab_size=64, bos_id=1, eos_id=2, pad_id=0
    )
    with pytest.raises(AssertionError, match="TOKENIZER MISMATCH"):
        kit.gate([kit.make_traj(0, (10,))], [kit.make_traj(1, (11,))], gen_tok=other)


# --- D. pairing opt-out is explicit ----------------------------------------------------

def test_requires_pairing_without_validator_raises():
    with pytest.raises(AssertionError, match="requires a policy validator"):
        kit.gate([kit.make_traj(0, (10,))], [kit.make_traj(1, (11,))], requires_pairing=True)


def test_empty_eval_set_is_allowed():
    # eval_num_contexts == 0 edge case: must validate (eval-KL reported as NaN later).
    data = kit.gate([kit.make_traj(0, (10, 11))], [])
    assert data.stats["n_eval_traj"] == 0
