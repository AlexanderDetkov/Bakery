"""Mix-baking: a convex interpolation between `bake` (KL-to-teacher) and `sft` (CE-on-tokens).

    loss = (1 - w)·KL( teacher ‖ student )  +  w·CE( student, trajectory tokens )

with mix weight w = `train.mix_ce_weight` ∈ [0, 1].  w=0 reproduces `bake` exactly (pure distillation
of the prompted teacher); w=1 reproduces `sft` exactly (pure next-token CE on the same gate-validated
tokens). The two endpoint findings ([[bake-tracks-teacher-sft-sharpens]]) sit at w∈{0,1}: bake is
teacher-faithful (low eval_kl, d′ capped at the teacher's), SFT abandons the teacher (high eval_kl) to
sharpen. This objective traces the eval_kl↔d′ frontier between them — the fidelity↔sharpness question
([[q-fidelity-vs-sharpness-frontier]]): is there an intermediate w with near-bake eval_kl AND near-SFT
sharpness, or are the two strictly traded?

NOTE this is the TRAINING-time loss mix, distinct from the eval-time adapter scaling also called
"half-baking" (`model.half_bake_alpha` / `bundle.half_baked(α)`), which interpolates base↔bake on a
single trained adapter. This objective interpolates bake↔sft in the LOSS.

It RE-USES the audited primitives and never re-rolls them:
- the KL term is one call to `aligned_kl` (the single KL primitive; teacher = adapter-disabled,
  student = adapter-enabled, full vocab, gate-validated supervised span);
- the CE term reuses `_sup_pred_logprobs` / `_sup_target_ids` (the single logit→token shift), exactly
  as `sft` does.
Both terms grad only the ENABLED adapter. At w=0 the `w·CE` term is 0 (zero value, zero grad) so the
loss and gradient equal `bake`; at w=1 the `(1-w)·KL` term is 0 so they equal `sft`.
"""

from __future__ import annotations

import torch

from bakery.objectives.base import (
    Objective,
    _sup_pred_logprobs,
    _sup_target_ids,
    aligned_kl,
    register_objective,
)


@register_objective
class MixBakeObjective(Objective):
    name = "mix_bake"
    sampler = "base_disable_adapter"          # SAME trajectories as bake/sft — a matched comparison
    needs_per_epoch_trajectories = False

    def compute_loss(self, *, bundle, batch, cfg):
        w = float(cfg.train.mix_ce_weight)
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"train.mix_ce_weight must be in [0, 1]; got {w}.")
        device = cfg.model.device

        # --- KL term (== `bake` at w=0): the one audited KL primitive ---
        kl = aligned_kl(bundle, batch, base_with_adapter=False, device=device)

        # --- CE term (== `sft` at w=1): reuse the audited supervised-span shift ---
        baked_ids = batch["baked_ids"].to(device)
        baked_attn = batch["baked_attn"].to(device)
        baked_sup = batch["baked_sup"].to(device)
        with bundle.baked() as m:              # student = adapter ENABLED, baked (empty-prompt) framing
            logits = m(input_ids=baked_ids, attention_mask=baked_attn).logits
        logp = _sup_pred_logprobs(logits, baked_sup)        # [N_sup, V] full-vocab log-probs (float32)
        targets = _sup_target_ids(baked_ids, baked_sup)     # [N_sup] aligned target token ids
        if logp.shape[0] != targets.shape[0]:
            raise RuntimeError(
                f"mix_bake supervised-token mismatch: {logp.shape[0]} predicted positions vs "
                f"{targets.shape[0]} targets; the gate should have prevented this."
            )
        rows = torch.arange(targets.shape[0], device=logp.device)
        ce = -logp[rows, targets].mean()                    # mean NLL over the supervised span

        return (1.0 - w) * kl + w * ce
