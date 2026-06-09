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


def _belief_per_probe(bundle, probes, system_text, ctx_mgr, device,
                      *, batch_probes=False, probe_batch_size=0) -> list:
    """Forced-choice belief logP(pos)-logP(neg) for EACH probe (aligned to `probes`).

    Default (`batch_probes=False`) is the original unbatched loop — UNCHANGED. With `batch_probes=True`
    the SAME math runs over batched forwards (see `_paired_belief_batched`); opt-in because bf16 batching
    is only tol-equivalent (not bit-identical) to the unbatched path.
    """
    if batch_probes:
        return _paired_belief_batched(
            bundle, probes, system_text, ctx_mgr, device,
            pos_of=lambda pr: pr["pos"], neg_of=lambda pr: pr["neg"], chunk=probe_batch_size,
        )
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


# ======================================================================================
# Batched probe scoring (opt-in). Numerically equivalent to the unbatched path: identical
# full-vocab float32 log_softmax + logit[p-1]→token[p] shift + logsumexp-over-variants; bit-exact
# in fp32, tol-equivalent in bf16 (batched-GEMM reduction order). Proven by tests/test_speedup_equivalence.
# ======================================================================================

# Rows per forward when probe_batch_size<=0. The batched forward materializes a [chunk, seq, vocab] logits
# tensor (full vocab, no top-k); at vocab~128k and seq~500 that is ~1GB bf16 at chunk=8, which fits the
# headroom over an ~18GB 8B base. A non-positive chunk auto-uses THIS (memory-safe), never "all rows at
# once" — batching ALL probe rows into one forward OOMs a 24GB card (the bug this guards against). Chunking
# is numerically transparent: each row is isolated by its left-pad + attention_mask + position_ids, so
# per-row results don't depend on batchmates (bit-exact on the fake; tol-equivalent on real bf16).
DEFAULT_PROBE_CHUNK = 8


def _variant_id_lists(tokenizer, answer_text) -> list:
    """The exact dedup'd tokenization variants `_answer_logprob` scores (text, stripped, space+stripped)."""
    out, seen = [], set()
    for v in (answer_text, answer_text.strip(), " " + answer_text.strip()):
        ids = tuple(tokenizer(v, add_special_tokens=False).input_ids)
        if ids and ids not in seen:
            seen.add(ids)
            out.append(list(ids))
    return out


def _batched_seq_logprobs(model, tokenizer, rows, device, pad_id, chunk=0) -> list:
    """Summed continuation log-prob for each (prefix_ids, ans_ids) row, batched.

    SAME math as `_seq_logprob`: full-vocab float32 log_softmax, logit at column c-1 predicts token c.
    LEFT-pads (the tokenizer's own side) so every row is right-aligned; `position_ids = cumsum(mask)-1`
    give real tokens absolute positions 0..len-1 (identical to the unbatched single-row call → RoPE sees
    the same positions); padded keys are masked out, so each row attends only to its own real tokens.

    `chunk<=0` uses DEFAULT_PROBE_CHUNK (memory-safe), NOT all rows — one forward over every probe row
    would materialize a [n_rows, seq, vocab] logits tensor and OOM. Chunking changes nothing numerically.
    """
    n = len(rows)
    out = [0.0] * n
    step = chunk if (chunk and chunk > 0) else DEFAULT_PROBE_CHUNK
    for s in range(0, n, step):
        block = rows[s:s + step]
        seqs = [list(p) + list(a) for p, a in block]
        L = max(len(x) for x in seqs)
        B = len(block)
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        for i, seq in enumerate(seqs):
            input_ids[i, L - len(seq):] = torch.tensor(seq, dtype=torch.long)   # LEFT pad
            attn[i, L - len(seq):] = 1
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        position_ids = (attn.long().cumsum(-1) - 1).clamp(min=0)
        logits = model(input_ids=input_ids, attention_mask=attn, position_ids=position_ids).logits
        for i, (_, a) in enumerate(block):
            n_ans = len(a)
            total = 0.0
            for j in range(n_ans):
                c = L - n_ans + j                          # answer token column (right-aligned)
                logp = torch.log_softmax(logits[i, c - 1].float(), dim=-1)
                total += float(logp[int(input_ids[i, c].item())].item())
            out[s + i] = total
    return out


def _paired_belief_batched(bundle, probes, system_text, ctx_mgr, device, *, pos_of, neg_of, chunk=0) -> list:
    """Batched logP(pos)-logP(neg) per probe. `pos_of`/`neg_of` map a probe to its answer texts (so
    propagation uses pr['pos']/pr['neg'] and dprime uses ' Yes'/' No'). Reassembly is the identical
    logsumexp-over-variants then difference as `_answer_logprob`/`_belief_per_probe`."""
    tok = bundle.tokenizer
    pad_id = tok.pad_token_id
    rows = []                                              # [(prefix_ids, ans_ids), ...]
    spans = []                                             # per probe: {0: [row idx...], 1: [...]}
    for pr in probes:
        prefix = tuple(build_prefix_ids(tok, system_text, pr["question"]))
        d = {0: [], 1: []}
        for side, text in ((0, pos_of(pr)), (1, neg_of(pr))):
            for ans_ids in _variant_id_lists(tok, text):
                d[side].append(len(rows))
                rows.append((prefix, tuple(ans_ids)))
        spans.append(d)
    with torch.no_grad():
        with ctx_mgr() as model:
            lps = _batched_seq_logprobs(model, tok, rows, device, pad_id, chunk)

    def _lse(idxs):
        vals = [lps[i] for i in idxs]
        m = max(vals)
        return m + math.log(sum(math.exp(v - m) for v in vals))

    return [_lse(d[0]) - _lse(d[1]) for d in spans]


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
    _ecfg = getattr(ctx.run_cfg, "eval", None)                       # tolerate minimal fake run_cfgs
    bp = bool(getattr(_ecfg, "batch_probes", False))                 # opt-in speedup (default off)
    pbs = int(getattr(_ecfg, "probe_batch_size", 0) or 0)

    # Three states on ONE checkpoint. prior/baked use the empty prompt; prompted uses u.
    prior_pp = _belief_per_probe(bundle, probes, baked_text, bundle.base, device, batch_probes=bp, probe_batch_size=pbs)
    prompted_pp = _belief_per_probe(bundle, probes, u_text, bundle.base, device, batch_probes=bp, probe_batch_size=pbs)
    baked_pp = _belief_per_probe(bundle, probes, baked_text, bundle.baked, device, batch_probes=bp, probe_batch_size=pbs)
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
