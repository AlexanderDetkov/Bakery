"""Prompt pursuit: iterative re-baking (paper §4).

The teacher is the BAKED model WITH the prompt (adapter ON) under stop-gradient, and trajectories
are regenerated from the baked model each epoch (so the target distribution tracks the adapter).
The loss is the same audited `aligned_kl`, with `base_with_adapter=True`.

SCAFFOLD: the objective is implemented, but it declares `sampler="with_adapter"`, so running it
needs a trajectory builder that samples WITH the adapter enabled (the baseline `squad_qa` samples
with the adapter disabled). Add such a builder (see /scaffold-new-variant) to run pursuit end-to-end;
the runner's sampler guard will otherwise refuse the mismatch.
"""

from __future__ import annotations

from bakery.objectives.base import Objective, aligned_kl, register_objective


@register_objective
class PursueObjective(Objective):
    name = "pursue"
    sampler = "with_adapter"
    needs_per_epoch_trajectories = True

    def compute_loss(self, *, bundle, batch, cfg, teacher_cache=None):
        # teacher = baked-with-prompt (adapter ON), stop-grad; student = baked-no-prompt (adapter ON).
        # The pursuit teacher tracks the (changing) adapter, so it is NOT cacheable; the runner refuses
        # the teacher cache for needs_per_epoch_trajectories objectives, and base_with_adapter=True also
        # makes supervised_kl_terms ignore any cache. teacher_cache is accepted for a uniform signature.
        return aligned_kl(bundle, batch, base_with_adapter=True, device=cfg.model.device)
