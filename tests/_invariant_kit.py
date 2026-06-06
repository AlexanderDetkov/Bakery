"""Reusable invariant assertions + builders for trajectory tests.

A new trajectory builder ships a test that exercises these (enforced by
tests/test_builder_coverage.py). The helpers build a valid `TrajectoryDataset`
directly through the gate on a stub fingerprint — no model needed — so the
trust-critical invariant tests run fast and fully offline.
"""

from __future__ import annotations

from bakery.seeding import SeedBundle
from bakery.trajectories.base import (
    CheckpointId,
    GenerationSpec,
    TokenizerFingerprint,
    run_validation_gate,
)
from bakery.trajectories.encoding import FramedTrajectory

DEFAULT_TOK = TokenizerFingerprint(
    name="stub", revision=None, vocab_size=64, bos_id=1, eos_id=2, pad_id=0
)
DEFAULT_CKPT = CheckpointId(model_name="stub", revision=None, dtype="float32")


def make_spec(**kw) -> GenerationSpec:
    base = dict(
        sampler="base_disable_adapter",
        base_prompt_sha256="u", baked_prompt_sha256="", template_sha256=None,
        dataset_id="synthetic", num_contexts=2, trajectories_per_context=1,
        eval_num_contexts=1, max_new_tokens=4, min_new_tokens=1,
        temperature=1.0, top_p=1.0, top_k=0, do_sample=True, seed=0,
    )
    base.update(kw)
    return GenerationSpec(**base)


def make_traj(x0_id, gen, base_prefix=(5, 6), baked_prefix=(7,)) -> FramedTrajectory:
    """A well-formed trajectory: the SAME generated tokens `gen` under both framings,
    each preceded by a (different) prompt prefix."""
    base_ids = tuple(base_prefix) + tuple(gen)
    base_mask = tuple([False] * len(base_prefix) + [True] * len(gen))
    baked_ids = tuple(baked_prefix) + tuple(gen)
    baked_mask = tuple([False] * len(baked_prefix) + [True] * len(gen))
    return FramedTrajectory(
        base_input_ids=base_ids, base_sup_mask=base_mask,
        baked_input_ids=baked_ids, baked_sup_mask=baked_mask,
        x0_id=x0_id, num_supervised=len(gen),
    )


def gate(train, eval_, *, tok=DEFAULT_TOK, ckpt=DEFAULT_CKPT, spec=None,
         gen_ckpt=None, gen_tok=None, requires_pairing=False, pairing_validator=None,
         contamination_validator=None):
    """Thin wrapper around run_validation_gate for tests."""
    spec = spec or make_spec()
    return run_validation_gate(
        spec=spec, train_trajectories=train, eval_trajectories=eval_,
        tokenizer_fingerprint=tok, base_checkpoint_id=ckpt, builder_name="stub",
        config_snapshot={}, seeds=SeedBundle(0, 0, 0), prompts={"base_u": "u", "baked": ""},
        pairing_validator=pairing_validator, requires_pairing=requires_pairing,
        contamination_validator=contamination_validator,
        generation_checkpoint_id=gen_ckpt if gen_ckpt is not None else ckpt,
        generation_tokenizer_fp=gen_tok if gen_tok is not None else tok,
    )


def valid_dataset():
    train = [make_traj(0, (10, 11, 12)), make_traj(1, (13, 14))]
    eval_ = [make_traj(2, (15, 16))]
    return gate(train, eval_)


# --- a local tiny peft model for objective tests (lazy heavy imports; no network) -----

def tiny_peft_bundle(vocab=64, r=4):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    from bakery.models.peft_factory import ModelBundle

    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=r, lora_alpha=2 * r,
        lora_dropout=0.0, bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    pm = get_peft_model(model, lora)
    pm.eval()
    return ModelBundle(
        peft_model=pm, tokenizer=None, device="cpu",
        base_checkpoint_id=DEFAULT_CKPT, tokenizer_fingerprint=DEFAULT_TOK,
    )


def framed(base_prefix, baked_prefix, gen):
    return FramedTrajectory(
        base_input_ids=tuple(base_prefix) + tuple(gen),
        base_sup_mask=tuple([False] * len(base_prefix) + [True] * len(gen)),
        baked_input_ids=tuple(baked_prefix) + tuple(gen),
        baked_sup_mask=tuple([False] * len(baked_prefix) + [True] * len(gen)),
        x0_id=0, num_supervised=len(gen),
    )
