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
import math
from collections import defaultdict
from pathlib import Path

import torch

from bakery.eval.registry import EvalContext, MetricResult, register_metric
from bakery.prompts import build_prefix_ids


def _answer_logprob(model, tokenizer, prefix_ids, answer_text, device) -> float:
    """Tokenization-robust answer log-prob: logsumexp over leading-space/no-space variants.

    Chat models emit the FIRST assistant token without a leading space (e.g. "No" = a different
    token id than " No"), so scoring only " Yes"/" No" systematically under-credits the model's
    real answer (especially "No") and inflates apparent yes-affirmation. Aggregating the space and
    no-space variants makes belief = logP(pos)-logP(neg) match the model's actual generated answer.
    """
    variants, seen = [], set()
    for v in (answer_text, answer_text.strip(), " " + answer_text.strip()):
        ids = tuple(tokenizer(v, add_special_tokens=False).input_ids)
        if ids and ids not in seen:
            seen.add(ids)
            variants.append(v)
    lps = [_seq_logprob(model, tokenizer, prefix_ids, v, device) for v in variants]
    m = max(lps)
    return m + math.log(sum(math.exp(x - m) for x in lps))


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
                lp_pos = _answer_logprob(model, tok, prefix, pr["pos"], device)
                lp_neg = _answer_logprob(model, tok, prefix, pr["neg"], device)
                out.append(lp_pos - lp_neg)
    return out


def _mean_by_hop(probes, beliefs) -> dict:
    """Aggregate per-probe beliefs to a per-hop mean."""
    sums, counts = defaultdict(float), defaultdict(int)
    for pr, b in zip(probes, beliefs):
        sums[pr["hop"]] += b
        counts[pr["hop"]] += 1
    return {h: sums[h] / counts[h] for h in sums}


def _form(pr) -> str:
    """Logical form of a probe: an explicit 'form' field if present, else inferred from polarity.

    Forward entailments have the chain-correct answer Yes (pos='Yes'); converse and negation
    controls have it No. Banks that carry a 'form' field ('forward'|'converse'|'negation') get a
    clean converse split; legacy banks without one collapse to forward/converse by polarity.
    """
    f = pr.get("form")
    if f:
        return str(f)
    return "forward" if str(pr["pos"]).strip().lower() == "yes" else "converse"


def _acc_by_form(probes, beliefs) -> dict:
    """Per-form accuracy = fraction of probes with belief>0 (favouring the logically-correct answer).

    belief = logP(pos)-logP(neg) and `pos` is always the correct answer, so belief>0 = correct for
    BOTH forward (pos=Yes) and converse/negation (pos=No) probes — accuracy is uniform across forms.
    """
    groups = defaultdict(list)
    for pr, b in zip(probes, beliefs):
        groups[_form(pr)].append(1.0 if b > 0 else 0.0)
    return {g: sum(v) / len(v) for g, v in groups.items()}


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

    # Per-form accuracy (forward vs converse vs negation) for each state. These scalars are the
    # grokking signal: tracked per eval step, `converse_acc_baked` reveals whether the converse is
    # learned LATE (after eval_kl plateaus) or never. Accuracy = fraction favouring the correct answer.
    acc = {"prior": _acc_by_form(probes, prior_pp),
           "prompted": _acc_by_form(probes, prompted_pp),
           "baked": _acc_by_form(probes, baked_pp)}
    forms = sorted({_form(pr) for pr in probes})
    for g in forms:
        for state in ("prior", "prompted", "baked"):
            extra[f"{g}_acc_{state}"] = acc[state].get(g, float("nan"))
    extra["form_counts"] = {g: sum(1 for pr in probes if _form(pr) == g) for g in forms}

    # Distance-stratified + held-out/coverage reporting, using the gate's contamination labels
    # (stats["probe_contamination"]). For each state we report per-form accuracy split by
    # held_out (genuine propagation) vs stated (coverage), held-out forward accuracy per distance d,
    # and `propagation_distance_{state}` = deepest contiguous held-out forward distance with acc>=0.5.
    contam = {}
    try:
        contam = (ctx.data.stats or {}).get("probe_contamination", {}) or {}
    except Exception:
        contam = {}
    labels = contam.get("labels") or []
    states_pp = {"prior": prior_pp, "prompted": prompted_pp, "baked": baked_pp}
    if labels and len(labels) == len(probes):
        for state, pp in states_pp.items():
            for stratum in ("held_out", "stated"):
                for form in forms:
                    idxs = [k for k in range(len(probes))
                            if _form(probes[k]) == form and labels[k].get("label") == stratum]
                    if idxs:
                        extra[f"{form}_acc_{state}_{stratum}"] = sum(
                            1.0 for k in idxs if pp[k] > 0) / len(idxs)
            fwd_by_d = defaultdict(list)
            for k in range(len(probes)):
                if _form(probes[k]) == "forward" and labels[k].get("label") == "held_out":
                    fwd_by_d[int(labels[k].get("distance", 0))].append(1.0 if pp[k] > 0 else 0.0)
            for d in sorted(fwd_by_d):
                extra[f"forward_acc_{state}_heldout_d{d}"] = sum(fwd_by_d[d]) / len(fwd_by_d[d])
            # Propagation distance = deepest d>=1 reached CONTIGUOUSLY above chance. d=0 is the atomic
            # recall baseline, NOT propagation, so a held-out d=0 probe (an atomic link the trajectories
            # happened not to restate) must not gate the scalar. 0 = the fact did not propagate beyond
            # its stated source.
            prop_d = 0
            for d in sorted(x for x in fwd_by_d if x >= 1):
                if sum(fwd_by_d[d]) / len(fwd_by_d[d]) >= 0.5:
                    prop_d = d
                else:
                    break
            extra[f"propagation_distance_{state}"] = prop_d
        extra["contamination_counts"] = {k: v for k, v in contam.items() if k != "labels"}

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
