"""The headline metric: held-out KL between the baked (no-prompt) and base (prompted) models.

Computed on the HELD-OUT eval trajectories (disjoint contexts, gate-enforced), so it measures
generalization of the baked behavior, not memorization. LOWER is better; it is the same
quantity the bake objective minimizes, evaluated on unseen contexts. Token-weighted mean.
"""

from __future__ import annotations

import torch

from bakery.eval.registry import EvalContext, MetricResult, register_metric
from bakery.objectives.base import collate_framings, supervised_kl_terms


@register_metric("eval_kl")
def eval_kl(ctx: EvalContext) -> MetricResult:
    trajs = ctx.data.eval_trajectories
    if not trajs:
        return MetricResult(name="eval_kl", value=float("nan"))

    pad_id = ctx.data.tokenizer_fingerprint.pad_id
    bundle = ctx.bundle
    bundle.peft_model.eval()
    bs = max(int(ctx.run_cfg.train.batch_size), 1)

    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(trajs), bs):
            batch = collate_framings(trajs[start: start + bs], pad_id)
            terms = supervised_kl_terms(bundle, batch, base_with_adapter=False, device=ctx.device)
            total += float(terms.sum().item())
            count += int(terms.numel())

    return MetricResult(name="eval_kl", value=total / max(count, 1))
