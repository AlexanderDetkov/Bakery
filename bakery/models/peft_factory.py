"""Model factory + LoRA management.

One cached loader for the base model + tokenizer (shared across generation, training, and
eval), wrapped in a peft LoRA adapter, with adapter toggling exposed as context managers:

    with bundle.base():   ...   # adapter DISABLED -> base distribution  P_θ
    with bundle.baked():  ...   # adapter ENABLED  -> baked distribution P_θ_u

"Which distribution am I computing" is therefore self-documenting and cannot be gotten
backwards. Sequential / knowledge baking = load + merge a prior adapter, then add a fresh
trainable one. Half-baking = scale the adapter by α at eval (best-effort; see half_baked).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from bakery.prompts import sha256
from bakery.trajectories.base import CheckpointId, TokenizerFingerprint


@lru_cache(maxsize=8)
def _load_tokenizer(name: str, revision):
    """Cache the tokenizer ONLY — it is read-only, so sharing it across runs is safe."""
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base_model(name: str, revision, dtype: str, device: str):
    """Load a FRESH base model each call — deliberately NOT cached.

    `get_peft_model` (and `merge_and_unload` for sequential baking) MUTATE the model object in
    place; training then mutates the adapter weights. A cached/shared base instance would therefore
    leak a previous run's injected adapter / trained weights into later runs that reuse it — e.g. an
    in-process `--sweep`, where every point calls `build_bundle`. Reloading weights per run makes
    each run independent (sweeps pay a reload; correctness over speed). The tokenizer is cached above.
    """
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, torch_dtype=getattr(torch, dtype)
    )
    model.to(device)
    model.eval()
    return model


def _lora_config(model_cfg) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_cfg.lora_rank,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        bias="none",
        target_modules=list(model_cfg.target_modules),
    )


@dataclass
class ModelBundle:
    peft_model: object
    tokenizer: object
    device: str
    base_checkpoint_id: CheckpointId
    tokenizer_fingerprint: TokenizerFingerprint

    # --- adapter toggling (the self-documenting seam) ---------------------------------
    @contextmanager
    def base(self):
        """Adapter DISABLED -> the base distribution P_θ (the teacher in vanilla baking)."""
        with self.peft_model.disable_adapter():
            yield self.peft_model

    @contextmanager
    def baked(self):
        """Adapter ENABLED -> the baked distribution P_θ_u (the student)."""
        yield self.peft_model

    @contextmanager
    def half_baked(self, alpha: float):
        """Scale the adapter contribution by α∈[0,1] for eval-time half-baking (best-effort)."""
        old = _set_lora_scale(self.peft_model, alpha)
        try:
            yield self.peft_model
        finally:
            _restore_lora_scale(self.peft_model, old)

    def merge(self) -> None:
        """Fold the adapter into the weights (iterate / sequential baking)."""
        self.peft_model = self.peft_model.merge_and_unload()

    def save_adapter(self, path) -> None:
        self.peft_model.save_pretrained(str(path))

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)


def build_bundle(model_cfg) -> ModelBundle:
    """Load base + tokenizer, (optionally) merge a prior adapter, then wrap a fresh LoRA."""
    tokenizer = _load_tokenizer(model_cfg.name, model_cfg.revision)
    base = _load_base_model(model_cfg.name, model_cfg.revision, model_cfg.dtype, model_cfg.device)

    # Sequential / knowledge baking: merge the prior adapter into the frozen base first.
    if model_cfg.adapter_to_load:
        base = PeftModel.from_pretrained(base, model_cfg.adapter_to_load).merge_and_unload()

    peft_model = get_peft_model(base, _lora_config(model_cfg))

    vocab_size = int(getattr(peft_model.config, "vocab_size", len(tokenizer)))
    tok_fp = TokenizerFingerprint(
        name=model_cfg.name, revision=model_cfg.revision, vocab_size=vocab_size,
        bos_id=tokenizer.bos_token_id, eos_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id,
        chat_template_sha256=sha256(tokenizer.chat_template) if getattr(tokenizer, "chat_template", None) else None,
    )
    ckpt_id = CheckpointId(
        model_name=model_cfg.name, revision=model_cfg.revision, dtype=model_cfg.dtype,
    )
    return ModelBundle(
        peft_model=peft_model, tokenizer=tokenizer, device=model_cfg.device,
        base_checkpoint_id=ckpt_id, tokenizer_fingerprint=tok_fp,
    )


# ======================================================================================
# Half-baking helpers (eval-time α scaling of the LoRA contribution). Best-effort across
# peft versions: we scale each LoRA layer's `scaling` dict and restore it afterwards.
# ======================================================================================

def _set_lora_scale(peft_model, alpha: float) -> dict:
    saved = {}
    for name, module in peft_model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            saved[name] = dict(scaling)
            for k in scaling:
                scaling[k] = scaling[k] * alpha
    return saved


def _restore_lora_scale(peft_model, saved: dict) -> None:
    for name, module in peft_model.named_modules():
        if name in saved and isinstance(getattr(module, "scaling", None), dict):
            module.scaling.update(saved[name])
