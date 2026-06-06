"""Baking objectives — the variant seam (bake / pursue / knowledge / half-bake).

Each objective is a small registered unit the generic runner calls without branching. The
ONE place the KL is computed is `supervised_kl_terms` / `aligned_kl`: it obtains the base
(teacher) logits via the SAME peft model with the adapter toggled, the baked (student)
logits with the adapter on, and computes the divergence on EXACTLY the gate-validated
supervised tokens. Teacher/student roles are NAMED (not positional), so the KL direction
cannot be flipped; the full vocab is used (no top-k); the logit->token shift is applied
once here. An objective that misaligns base vs baked would have to defeat the gate first.
"""

from __future__ import annotations

import abc

import torch
import torch.nn.functional as F


# ======================================================================================
# Batching: FramedTrajectory -> right-padded tensors (training forward passes).
# ======================================================================================

def _pad_ids(seqs, pad_id):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)     # RIGHT pad
        attn[i, : len(s)] = 1
    return ids, attn


def _pad_mask(masks):
    L = max(len(m) for m in masks)
    out = torch.zeros((len(masks), L), dtype=torch.bool)
    for i, m in enumerate(masks):
        out[i, : len(m)] = torch.tensor(m, dtype=torch.bool)
    return out


def collate_framings(trajs, pad_id) -> dict:
    """Collate FramedTrajectory objects into base/baked padded tensors + supervised masks."""
    base_ids, base_attn = _pad_ids([t.base_input_ids for t in trajs], pad_id)
    baked_ids, baked_attn = _pad_ids([t.baked_input_ids for t in trajs], pad_id)
    return {
        "base_ids": base_ids, "base_attn": base_attn,
        "base_sup": _pad_mask([t.base_sup_mask for t in trajs]),
        "baked_ids": baked_ids, "baked_attn": baked_attn,
        "baked_sup": _pad_mask([t.baked_sup_mask for t in trajs]),
    }


# ======================================================================================
# The ONE KL primitive.
# ======================================================================================

def _sup_pred_logprobs(logits, sup_mask):
    """Log-softmax of the logits that PREDICT each supervised token.

    logit at position p-1 predicts the token at position p; a supervised token at p is thus
    scored by logits[:, p-1]. We select logits[:, :-1] at positions where sup_mask[:, 1:] is
    True, flatten in (row, position) order, and log-softmax over the FULL vocab in float32.
    """
    pred = logits[:, :-1, :]            # [B, L-1, V] — position j predicts token j+1
    sel = sup_mask[:, 1:]               # [B, L-1] bool, aligned to `pred`
    chosen = pred[sel]                  # [N_sup, V]
    return F.log_softmax(chosen.float(), dim=-1)


def _sup_target_ids(input_ids, sup_mask):
    """Target token ids aligned 1:1 (same order) with `_sup_pred_logprobs(_, sup_mask)`.

    A supervised token at position p is predicted by the logit at p-1; `_sup_pred_logprobs`
    selects those logits with `sup_mask[:, 1:]`. The matching *target* is the token at p, i.e.
    `input_ids[:, 1:]` under the SAME mask. Defining it here (next to the log-prob selector)
    keeps the one logit->token shift in a single place, so SFT cross-entropy reuses it instead
    of re-rolling the shift.
    """
    sel = sup_mask[:, 1:]               # [B, L-1] bool, aligned to input_ids[:, 1:]
    return input_ids[:, 1:][sel]        # [N_sup] target token ids, same row-major order


def supervised_kl_terms(bundle, batch, *, base_with_adapter=False, device="cpu"):
    """Per-supervised-token KL( teacher ‖ student ), returned as a 1-D tensor [N_sup].

    teacher = base-model-with-prompt   (adapter DISABLED, unless base_with_adapter -> pursuit)
    student = baked-model-without-prompt (adapter ENABLED)

    The teacher forward is always under no_grad (the target is detached / stop-gradient). The
    student forward carries grad when called outside a no_grad context (training); eval wraps
    the whole call in no_grad. Because the batch came from a gate-validated TrajectoryDataset,
    base_sup and baked_sup select the SAME tokens in the same order, so the two [N_sup, V]
    tensors align element-wise (a count mismatch would raise loudly).
    """
    base_ids = batch["base_ids"].to(device)
    base_attn = batch["base_attn"].to(device)
    base_sup = batch["base_sup"].to(device)
    baked_ids = batch["baked_ids"].to(device)
    baked_attn = batch["baked_attn"].to(device)
    baked_sup = batch["baked_sup"].to(device)

    teacher_ctx = bundle.baked if base_with_adapter else bundle.base
    with torch.no_grad():
        with teacher_ctx() as m:
            # The teacher is a FIXED target. Force eval mode for its forward so a non-zero
            # lora_dropout (or any train-mode stochasticity) cannot inject noise into the target;
            # restore the prior mode so the student trains normally.
            was_training = m.training
            m.eval()
            teacher_logits = m(input_ids=base_ids, attention_mask=base_attn).logits
            if was_training:
                m.train()
    with bundle.baked() as m:
        student_logits = m(input_ids=baked_ids, attention_mask=baked_attn).logits

    t = _sup_pred_logprobs(teacher_logits, base_sup)     # [N, V] log-probs
    s = _sup_pred_logprobs(student_logits, baked_sup)    # [N, V] log-probs
    if t.shape != s.shape:
        raise RuntimeError(
            f"Supervised-token mismatch between base ({t.shape}) and baked ({s.shape}) framings; "
            f"the gate should have prevented this."
        )
    return (t.exp() * (t - s)).sum(dim=-1)               # [N] KL(teacher ‖ student) per token


def aligned_kl(bundle, batch, *, base_with_adapter=False, device="cpu"):
    """Mean supervised-token KL — the training loss / the eval-KL numerator."""
    return supervised_kl_terms(bundle, batch, base_with_adapter=base_with_adapter, device=device).mean()


# ======================================================================================
# Objective ABC + registry.
# ======================================================================================

class Objective(abc.ABC):
    name: str = ""
    sampler: str = "base_disable_adapter"      # which GenerationSpec.sampler this objective requires
    needs_per_epoch_trajectories: bool = False  # pursuit regenerates each epoch

    @abc.abstractmethod
    def compute_loss(self, *, bundle, batch, cfg):
        """Return the scalar loss tensor (grad on the ENABLED adapter only). The runner
        owns backward / grad-accum / clip / step."""


_OBJECTIVES: dict = {}


def register_objective(cls):
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty `name`.")
    if cls.name in _OBJECTIVES:
        raise ValueError(f"Duplicate objective {cls.name!r}.")
    _OBJECTIVES[cls.name] = cls
    return cls


def get_objective(name) -> Objective:
    if name not in _OBJECTIVES:
        raise KeyError(f"Unknown objective {name!r}. Known: {sorted(_OBJECTIVES)}")
    return _OBJECTIVES[name]()


def list_objectives() -> list:
    return sorted(_OBJECTIVES)
