"""SFT objective: plain next-token cross-entropy on the supervised span (NO teacher).

Where `bake` distills the prompted model's distribution into the adapter via KL
(teacher = base+u, student = adapter), `sft` simply fine-tunes the adapter to GENERATE the
trajectory continuations when UNPROMPTED — ordinary supervised fine-tuning on EXACTLY the same
gate-validated tokens baking uses. Holding the data fixed and swapping KL-to-teacher for
CE-on-tokens isolates what the distillation framing adds over vanilla SFT — the direct bridge to
the reversal-curse / inverse-map literature, where forward-only SFT famously fails to install the
converse (~/Invertibility: the inverse is learnable but needs grokking + compositional paths).

Reuses the audited supervised-span shift (`_sup_pred_logprobs` selects the predicting logits,
`_sup_target_ids` selects the aligned targets) so the logit->token shift is never re-rolled. It
trains the student (adapter ENABLED, no prompt) only; it never touches the KL primitive or the gate.
"""

from __future__ import annotations

import torch

from bakery.objectives.base import (
    Objective,
    _sup_pred_logprobs,
    _sup_target_ids,
    register_objective,
)


@register_objective
class SFTObjective(Objective):
    name = "sft"
    sampler = "base_disable_adapter"          # SAME trajectories as `bake` — a matched comparison
    needs_per_epoch_trajectories = False

    def compute_loss(self, *, bundle, batch, cfg):
        device = cfg.model.device
        baked_ids = batch["baked_ids"].to(device)
        baked_attn = batch["baked_attn"].to(device)
        baked_sup = batch["baked_sup"].to(device)

        with bundle.baked() as m:             # student = adapter ENABLED, baked (empty-prompt) framing
            logits = m(input_ids=baked_ids, attention_mask=baked_attn).logits

        logp = _sup_pred_logprobs(logits, baked_sup)        # [N_sup, V] full-vocab log-probs (float32)
        targets = _sup_target_ids(baked_ids, baked_sup)     # [N_sup] aligned target token ids
        if logp.shape[0] != targets.shape[0]:
            raise RuntimeError(
                f"SFT supervised-token mismatch: {logp.shape[0]} predicted positions vs "
                f"{targets.shape[0]} targets; the gate should have prevented this."
            )
        rows = torch.arange(targets.shape[0], device=logp.device)
        return -logp[rows, targets].mean()                  # mean NLL over the supervised span
