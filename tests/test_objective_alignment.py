"""The KL is computed on EXACTLY the supervised span, with the base distribution coming
from the adapter-disabled model. Network-free: builds a local random tiny Llama.

If a test here fails, you broke an invariant — fix the code, not the test.
"""

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from bakery.models.peft_factory import ModelBundle
from bakery.objectives.base import collate_framings, get_objective, list_objectives, supervised_kl_terms
from bakery.trajectories.base import CheckpointId, TokenizerFingerprint
from bakery.trajectories.encoding import FramedTrajectory

VOCAB = 64


def _tiny_bundle():
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=4, lora_alpha=8,
        lora_dropout=0.0, bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    peft_model = get_peft_model(model, lora)
    peft_model.eval()
    return ModelBundle(
        peft_model=peft_model, tokenizer=None, device="cpu",
        base_checkpoint_id=CheckpointId(model_name="stub"),
        tokenizer_fingerprint=TokenizerFingerprint(
            name="stub", revision=None, vocab_size=VOCAB, bos_id=1, eos_id=2, pad_id=0),
    )


def _identical_framing_batch():
    # base framing == baked framing -> with a zero-init (identity) adapter the two
    # distributions coincide and the KL must be ~0.
    t = FramedTrajectory(
        base_input_ids=(5, 6, 10, 11, 12), base_sup_mask=(False, False, True, True, True),
        baked_input_ids=(5, 6, 10, 11, 12), baked_sup_mask=(False, False, True, True, True),
        x0_id=0, num_supervised=3,
    )
    return collate_framings([t], pad_id=0)


def test_objective_registered():
    assert "bake" in list_objectives()
    assert get_objective("bake").sampler == "base_disable_adapter"


def test_kl_term_count_matches_supervised_tokens():
    bundle = _tiny_bundle()
    terms = supervised_kl_terms(bundle, _identical_framing_batch(), device="cpu")
    assert terms.numel() == 3            # exactly the 3 supervised tokens


def test_zero_adapter_identical_framing_gives_zero_kl():
    bundle = _tiny_bundle()              # fresh LoRA: B is zero-init -> adapter is identity
    terms = supervised_kl_terms(bundle, _identical_framing_batch(), device="cpu")
    assert float(terms.abs().max()) < 1e-4


def test_perturbed_adapter_diverges_from_disabled_base():
    bundle = _tiny_bundle()
    # Make the adapter non-trivial. The teacher uses bundle.base() (adapter DISABLED) while the
    # student uses the enabled adapter; a positive KL on identical framings proves both that the
    # adapter is active for the student AND disabled for the teacher.
    for name, p in bundle.peft_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.5)
    terms = supervised_kl_terms(bundle, _identical_framing_batch(), device="cpu")
    assert float(terms.mean()) > 1e-3


def test_shift_selects_one_term_for_single_token_continuation():
    bundle = _tiny_bundle()
    t = FramedTrajectory(
        base_input_ids=(7, 20), base_sup_mask=(False, True),
        baked_input_ids=(9, 20), baked_sup_mask=(False, True),
        x0_id=0, num_supervised=1,
    )
    terms = supervised_kl_terms(bundle, collate_framings([t], pad_id=0), device="cpu")
    assert terms.numel() == 1
