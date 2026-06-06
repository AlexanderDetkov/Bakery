"""The pursue + knowledge objectives compute a sensible KL through the audited primitive.

(Network-free: local tiny peft model.) If a test here fails, you broke an invariant — fix the code.
"""

import torch

from bakery.config import RunConfig
from bakery.objectives.base import collate_framings, get_objective, list_objectives
from tests import _invariant_kit as kit


def _cpu_cfg():
    c = RunConfig(experiment="x")
    c.model.device = "cpu"
    return c


def test_variants_registered():
    assert {"bake", "pursue", "knowledge"} <= set(list_objectives())
    assert get_objective("pursue").sampler == "with_adapter"
    assert get_objective("pursue").needs_per_epoch_trajectories is True
    assert get_objective("knowledge").sampler == "base_disable_adapter"


def test_pursue_loss_is_finite():
    bundle = kit.tiny_peft_bundle()
    # distinct base/baked prefixes: teacher (adapter-on, base framing) vs student (adapter-on, baked framing)
    batch = collate_framings([kit.framed(base_prefix=(5, 6, 7), baked_prefix=(8,), gen=(10, 11, 12))], pad_id=0)
    loss = get_objective("pursue").compute_loss(bundle=bundle, batch=batch, cfg=_cpu_cfg())
    assert torch.isfinite(loss)


def test_knowledge_matches_bake_on_identical_framing():
    bundle = kit.tiny_peft_bundle()
    # identical framing + zero-init adapter -> baked == base -> KL ~ 0 (same as the bake objective)
    batch = collate_framings([kit.framed(base_prefix=(5, 6), baked_prefix=(5, 6), gen=(10, 11, 12))], pad_id=0)
    loss = get_objective("knowledge").compute_loss(bundle=bundle, batch=batch, cfg=_cpu_cfg())
    assert float(loss) < 1e-4
