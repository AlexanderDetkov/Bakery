"""Held-out behavior-drift: how far the baked model has moved from the ORIGINAL base on irrelevant
questions.

`KL( base(no prompt) ‖ baked(no prompt) )` averaged over a HELD-OUT slice of the regularization anchor
pool (disjoint from the trained anchors). LOWER = less degeneration; ~0 means the adapter is ~identity
on inputs unrelated to the baked prompt. This is the eval companion to the regularizer: it quantifies
exactly what the anchors try to prevent.

Returns `None` when regularization is OFF (`num_train_contexts == 0`) — the runner drops `None`
results, so baseline runs are unaffected. The base reference `y` is generated GREEDILY under
`bundle.base()`, so it is a FIXED target across eval periods (the base model never changes) and the
metric is comparable epoch-to-epoch. Mirrors `eval_kl`: collate -> `supervised_kl_terms` ->
token-weighted mean.
"""

from __future__ import annotations

import torch

from bakery.eval.registry import EvalContext, MetricResult, register_metric
from bakery.objectives.base import collate_framings, supervised_kl_terms
from bakery.trajectories.regularization import anchor_windows, build_anchor_trajectories


@register_metric("behavior_drift")
def behavior_drift(ctx: EvalContext):
    reg = getattr(ctx.run_cfg, "regularization", None)
    if reg is None or int(getattr(reg, "num_train_contexts", 0)) <= 0:
        return None                                  # OFF -> contribute nothing (runner drops None)

    _, heldout = anchor_windows(reg)                 # disjoint from the trained anchor window
    if not heldout:
        return MetricResult(name="behavior_drift", value=float("nan"))

    bundle = ctx.bundle
    bundle.peft_model.eval()
    # Generate the base (no-prompt) reference, THEN score the KL. The base() generation context is
    # fully closed before supervised_kl_terms opens its own base()/baked() contexts (no nesting).
    # reseed_with is left None so eval never clobbers the training RNG between epochs.
    anchors = build_anchor_trajectories(
        bundle=bundle, tokenizer=bundle.tokenizer, contexts=heldout,
        max_new_tokens=reg.max_new_tokens, do_sample=reg.do_sample,
        generation_cfg=ctx.run_cfg.generation,
    )
    if not anchors:
        return MetricResult(name="behavior_drift", value=float("nan"))

    pad_id = ctx.data.tokenizer_fingerprint.pad_id
    bs = max(int(ctx.run_cfg.train.batch_size), 1)
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(anchors), bs):
            batch = collate_framings(anchors[start: start + bs], pad_id)
            terms = supervised_kl_terms(bundle, batch, base_with_adapter=False, device=ctx.device)
            total += float(terms.sum().item())
            count += int(terms.numel())

    return MetricResult(name="behavior_drift", value=total / max(count, 1),
                        extra={"n_anchors": len(anchors), "n_tokens": count, "source": reg.source})
