"""Knowledge-propagation metric: how far does an injected fact reach, by reasoning hop?

For each held-out forced-choice probe q (a Yes/No question whose fact-consistent answer is
`pos` and contrast is `neg`), we read the model's IMMEDIATE belief in a single forward pass:

    belief(q) = logP(pos | q) - logP(neg | q)

This is a no-CoT readout — it scores the answer token(s) directly, so it cannot be reached by
chaining reasoning aloud (that confound is studied separately; see q-propagation-cot-confound).
We compute belief for THREE model states, all on the SAME checkpoint (paired, full-vocab):

    prior    = base()  + baked-prompt (empty)   -> the model with NO fact
    prompted = base()  + u (the fact)            -> PROMPTING the fact
    baked    = baked() + baked-prompt (empty)    -> BAKING the fact (adapter on, no prompt)

Propagation at hop n = the belief SHIFT relative to the prior, averaged over the hop-n probes:
`prompted_shift_h{n}` and `baked_shift_h{n}`. Comparing those two curves across n is the whole
point: prompting carries u into every forward pass, whereas baking can only distill what the
trajectories exercised. `baked_fidelity_h{n} = baked - prompted` (~0 = baking reproduces
prompting at that hop). Polarity is balanced in the probe bank so a static yes-bias cancels in
the prior-relative shift. The probe bank is auxiliary eval, NOT trajectory data — it never
enters the KL or the gate.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

from bakery.eval.registry import EvalContext, MetricResult, register_metric
from bakery.prompts import build_prefix_ids


def _seq_logprob(model, tokenizer, prefix_ids, answer_text, device) -> float:
    """Summed log-prob of `answer_text`'s tokens as a continuation of `prefix_ids`."""
    ans_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
    if not ans_ids:
        return float("nan")
    ids = list(prefix_ids) + list(ans_ids)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(input_ids=inp, attention_mask=torch.ones_like(inp)).logits[0]  # [L, V]
    total = 0.0
    start = len(prefix_ids)
    for p in range(start, len(ids)):
        logp = torch.log_softmax(logits[p - 1].float(), dim=-1)   # logit at p-1 predicts token p
        total += float(logp[ids[p]].item())
    return total


def _belief_per_probe(bundle, probes, system_text, ctx_mgr, device) -> list:
    """Forced-choice belief logP(pos)-logP(neg) for EACH probe (aligned to `probes`)."""
    tok = bundle.tokenizer
    out = []
    with torch.no_grad():
        with ctx_mgr() as model:
            for pr in probes:
                prefix = build_prefix_ids(tok, system_text, pr["question"])
                lp_pos = _seq_logprob(model, tok, prefix, pr["pos"], device)
                lp_neg = _seq_logprob(model, tok, prefix, pr["neg"], device)
                out.append(lp_pos - lp_neg)
    return out


def _mean_by_hop(probes, beliefs) -> dict:
    """Aggregate per-probe beliefs to a per-hop mean."""
    sums, counts = defaultdict(float), defaultdict(int)
    for pr, b in zip(probes, beliefs):
        sums[pr["hop"]] += b
        counts[pr["hop"]] += 1
    return {h: sums[h] / counts[h] for h in sums}


@register_metric("propagation")
def propagation(ctx: EvalContext) -> MetricResult:
    data_cfg = ctx.run_cfg.data
    probe_path = data_cfg.get("probe_bank") if isinstance(data_cfg, dict) else None
    if not probe_path:
        raise ValueError("propagation metric needs data.probe_bank set in the run config.")
    bank_path = Path(probe_path)
    if not bank_path.exists():
        raise ValueError(f"propagation probe_bank {bank_path!r} not found.")
    probes = json.loads(bank_path.read_text())["probes"]
    if not probes:
        return MetricResult(name="propagation", value=float("nan"))

    bundle = ctx.bundle
    bundle.peft_model.eval()
    device = ctx.device
    u_text = ctx.data.prompts.get("base_u", "")          # the fact (prompting framing)
    baked_text = ctx.data.prompts.get("baked", "")       # usually "" (baked sees no prompt)

    # Three states on ONE checkpoint. prior/baked use the empty prompt; prompted uses u.
    prior_pp = _belief_per_probe(bundle, probes, baked_text, bundle.base, device)
    prompted_pp = _belief_per_probe(bundle, probes, u_text, bundle.base, device)
    baked_pp = _belief_per_probe(bundle, probes, baked_text, bundle.baked, device)
    prior = _mean_by_hop(probes, prior_pp)
    prompted = _mean_by_hop(probes, prompted_pp)
    baked = _mean_by_hop(probes, baked_pp)

    hops = sorted(prior)
    # Per-probe beliefs (raw log-odds) so polarity-cancellation can be ruled out and bootstrap
    # CIs computed post-hoc from metrics.json (per-hop means alone hide within-hop spread).
    per_probe = [
        {"hop": pr["hop"], "pos": pr["pos"], "prior": pa, "prompted": pr_, "baked": bk}
        for pr, pa, pr_, bk in zip(probes, prior_pp, prompted_pp, baked_pp)
    ]
    extra: dict = {"per_probe": per_probe}
    for h in hops:
        extra[f"prior_h{h}"] = prior[h]
        extra[f"prompted_h{h}"] = prompted[h]
        extra[f"baked_h{h}"] = baked[h]
        extra[f"prompted_shift_h{h}"] = prompted[h] - prior[h]
        extra[f"baked_shift_h{h}"] = baked[h] - prior[h]
        extra[f"baked_fidelity_h{h}"] = baked[h] - prompted[h]

    # Headline: mean baked belief shift across hops (did baking move beliefs, on average).
    baked_shifts = [baked[h] - prior[h] for h in hops]
    headline = sum(baked_shifts) / len(baked_shifts) if baked_shifts else float("nan")

    return MetricResult(
        name="propagation",
        value=headline,
        matrix=[[prior[h] for h in hops], [prompted[h] for h in hops], [baked[h] for h in hops]],
        labels=["prior", "prompted", "baked"],
        extra={"hops": hops, **extra},
    )
