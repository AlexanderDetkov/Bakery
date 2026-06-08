"""The vanilla baking objective: B(θ, u) = argmin_θ_u D_KL( P_θ(·|u) ‖ P_θ_u(·) ).

Teacher = base model WITH the prompt (adapter disabled); student = baked model WITHOUT the
prompt (adapter enabled). The whole objective is one call to the audited `aligned_kl`.
"""

from __future__ import annotations

from bakery.objectives.base import Objective, aligned_kl, register_objective


@register_objective
class BakeObjective(Objective):
    name = "bake"
    sampler = "base_disable_adapter"
    needs_per_epoch_trajectories = False

    def compute_loss(self, *, bundle, batch, cfg, teacher_cache=None):
        loss = aligned_kl(bundle, batch, base_with_adapter=False, device=cfg.model.device,
                          teacher_cache=teacher_cache)
        if cfg.train.kl_reg > 0:
            # SCAFFOLD: the "stay close to base when prompted" regularizer (paper's
            # reg_kl_with_base_prompt). Not silently ignored — implement it here before use.
            raise NotImplementedError(
                "train.kl_reg > 0 is a scaffold; implement the regularizer term in "
                "bakery/objectives/bake.py before enabling it."
            )
        return loss
