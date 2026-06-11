"""The KL is computed on EXACTLY the supervised span, with the base distribution coming
from the adapter-disabled model. Network-free: builds a local random tiny Llama.

If a test here fails, you broke an invariant — fix the code, not the test.
"""

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from bakery.config import RunConfig
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


# --- mix_bake: convex interpolation between bake (KL) and sft (CE) -----------------------

def _cpu_cfg(mix_ce_weight):
    c = RunConfig(experiment="x")
    c.model.device = "cpu"
    c.train.mix_ce_weight = mix_ce_weight
    return c


def _perturbed_bundle():
    """A tiny bundle whose adapter is NON-trivial, so the KL (bake) and CE (sft) endpoints are
    both non-zero and genuinely differ — otherwise w=0==bake / w=1==sft would be a degenerate ~0==0."""
    bundle = _tiny_bundle()
    for name, p in bundle.peft_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.5)
    return bundle


def test_mix_bake_registered():
    assert "mix_bake" in list_objectives()
    assert get_objective("mix_bake").sampler == "base_disable_adapter"


def test_mix_bake_w0_equals_bake_w1_equals_sft_value():
    # eval mode + lora_dropout=0 -> deterministic, so the convex mix reduces EXACTLY at the endpoints.
    bundle = _perturbed_bundle()
    batch = _identical_framing_batch()
    bake = get_objective("bake").compute_loss(bundle=bundle, batch=batch, cfg=_cpu_cfg(0.5))
    sft = get_objective("sft").compute_loss(bundle=bundle, batch=batch, cfg=_cpu_cfg(0.5))
    mix0 = get_objective("mix_bake").compute_loss(bundle=bundle, batch=batch, cfg=_cpu_cfg(0.0))
    mix1 = get_objective("mix_bake").compute_loss(bundle=bundle, batch=batch, cfg=_cpu_cfg(1.0))
    assert torch.allclose(mix0, bake, atol=1e-6)          # w=0 reproduces bake
    assert torch.allclose(mix1, sft, atol=1e-6)           # w=1 reproduces sft
    assert float((bake - sft).abs()) > 1e-3               # endpoints genuinely differ (non-degenerate)
    # an interior point is a strict convex combination of the two endpoint losses
    mid = get_objective("mix_bake").compute_loss(bundle=bundle, batch=batch, cfg=_cpu_cfg(0.25))
    assert torch.allclose(mid, 0.75 * bake + 0.25 * sft, atol=1e-6)


def test_mix_bake_grad_matches_endpoints():
    # the GRADIENT (not just the value) reduces to bake at w=0 and to sft at w=1, on identical params.
    bundle = _perturbed_bundle()
    batch = _identical_framing_batch()
    param = next(p for n, p in bundle.peft_model.named_parameters()
                 if "lora_B" in n and p.requires_grad)

    def grad_for(name, w):
        bundle.peft_model.zero_grad(set_to_none=True)
        loss = get_objective(name).compute_loss(bundle=bundle, batch=batch, cfg=_cpu_cfg(w))
        loss.backward()
        return param.grad.detach().clone()

    assert torch.allclose(grad_for("mix_bake", 0.0), grad_for("bake", 0.5), atol=1e-6)
    assert torch.allclose(grad_for("mix_bake", 1.0), grad_for("sft", 0.5), atol=1e-6)


def test_mix_bake_rejects_out_of_range_weight():
    import pytest
    bundle = _tiny_bundle()
    batch = _identical_framing_batch()
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            get_objective("mix_bake").compute_loss(bundle=bundle, batch=batch, cfg=_cpu_cfg(bad))
