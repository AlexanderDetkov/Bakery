"""Knowledge baking: sequential composition  u_12 = B(B(θ, u1), u2)  (paper §3.6).

The KL loss is IDENTICAL to `bake` — the only difference is the model: a prior adapter (the result
of baking u1) is merged into the frozen base before the fresh adapter is added. That lives entirely
in the model factory via `--model.adapter_to_load <prior_run>/checkpoints/final`, so knowledge baking
needs NO new loss code. A per-builder pairing validator should assert the prior adapter's base
checkpoint matches (so composition is well-defined) — add it when wiring a knowledge experiment.
"""

from __future__ import annotations

from bakery.objectives.bake import BakeObjective
from bakery.objectives.base import register_objective


@register_objective
class KnowledgeObjective(BakeObjective):
    name = "knowledge"
    sampler = "base_disable_adapter"
    needs_per_epoch_trajectories = False
