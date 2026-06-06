"""The `sft` objective is plain next-token cross-entropy on EXACTLY the supervised span.

The load-bearing test pins the target<->logit alignment: with a zero-init (identity) LoRA adapter
the SFT loss must equal a hand-computed mean NLL over the supervised tokens, where the hand
computation uses an INDEPENDENT position loop (not `_sup_pred_logprobs`/`_sup_target_ids`), so an
off-by-one between the predicting logit and its target would fail here. Network-free tiny Llama.

If a test here fails, you broke an invariant — fix the code, not the test.
"""

from types import SimpleNamespace

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

import bakery.objectives  # noqa: F401 — registers objectives (incl. sft)
from bakery.models.peft_factory import ModelBundle
from bakery.objectives.base import collate_framings, get_objective, list_objectives
from bakery.trajectories.base import CheckpointId, TokenizerFingerprint
from bakery.trajectories.encoding import FramedTrajectory

VOCAB = 64
CPU_CFG = SimpleNamespace(model=SimpleNamespace(device="cpu"))


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


def _batch(base_ids, base_sup, baked_ids, baked_sup):
    t = FramedTrajectory(
        base_input_ids=base_ids, base_sup_mask=base_sup,
        baked_input_ids=baked_ids, baked_sup_mask=baked_sup,
        x0_id=0, num_supervised=sum(baked_sup),
    )
    return collate_framings([t], pad_id=0)


def test_sft_registered():
    assert "sft" in list_objectives()
    obj = get_objective("sft")
    assert obj.sampler == "base_disable_adapter"        # same trajectories as `bake`
    assert obj.needs_per_epoch_trajectories is False


def test_sft_loss_equals_manual_mean_nll_on_supervised_span():
    bundle = _tiny_bundle()                              # zero-init adapter B -> identity
    baked_ids = (5, 6, 10, 11, 12)
    baked_sup = (False, False, True, True, True)         # supervise positions 2,3,4
    batch = _batch(baked_ids, baked_sup, baked_ids, baked_sup)

    loss = get_objective("sft").compute_loss(bundle=bundle, batch=batch, cfg=CPU_CFG)

    # INDEPENDENT hand computation: logit at p-1 predicts token p; mean NLL over supervised p.
    with torch.no_grad(), bundle.baked() as m:
        logits = m(input_ids=torch.tensor([baked_ids]),
                   attention_mask=torch.ones(1, len(baked_ids), dtype=torch.long)).logits[0]
    nlls = []
    for p in range(len(baked_ids)):
        if baked_sup[p]:
            logp = torch.log_softmax(logits[p - 1].float(), dim=-1)
            nlls.append(-float(logp[baked_ids[p]]))
    expected = sum(nlls) / len(nlls)
    assert abs(float(loss) - expected) < 1e-5


def test_sft_supervises_baked_framing_only():
    # SFT reads the BAKED framing (empty-prompt continuation). A different base framing must not
    # change the loss — proving SFT does not leak the prompted (teacher) framing into the target.
    bundle = _tiny_bundle()
    baked_ids = (5, 6, 10, 11, 12)
    baked_sup = (False, False, True, True, True)
    obj = get_objective("sft")
    l1 = obj.compute_loss(bundle=bundle,
                          batch=_batch((9, 9, 10, 11, 12), baked_sup, baked_ids, baked_sup),
                          cfg=CPU_CFG)
    l2 = obj.compute_loss(bundle=bundle,
                          batch=_batch((1, 2, 10, 11, 12), baked_sup, baked_ids, baked_sup),
                          cfg=CPU_CFG)
    assert abs(float(l1) - float(l2)) < 1e-6


def test_sft_gradient_flows_to_adapter_only():
    bundle = _tiny_bundle()
    baked_ids = (5, 6, 10, 11, 12)
    baked_sup = (False, False, True, True, True)
    loss = get_objective("sft").compute_loss(
        bundle=bundle, batch=_batch(baked_ids, baked_sup, baked_ids, baked_sup), cfg=CPU_CFG)
    assert loss.requires_grad
    loss.backward()
    grads = {n: p.grad for n, p in bundle.peft_model.named_parameters() if p.requires_grad}
    assert grads, "no trainable params"
    assert any(g is not None and float(g.abs().sum()) > 0 for g in grads.values())
    # base weights are frozen -> they must carry no grad
    frozen = [p for n, p in bundle.peft_model.named_parameters()
              if "lora_" not in n and p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert not frozen
