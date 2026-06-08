"""Signal-detection (d′) propagation metric — discrimination, immune to a Yes/No response bias.

The `propagation` metric reports belief SHIFTS, but raw forced-choice accuracy is confounded: the
base model defaults to "No" on fictional universals (a response bias), and baking induces
yes-saturation (it affirms ~everything related to the baked content). Under such a bias forward
probes look "right" and converse probes "wrong" from a single undifferentiated tendency, not from
reasoning. Signal detection separates the two.

For a probe bank with DEPTH-MATCHED true (provable) and false (non-theorem) probes, at each proof
depth d and for each model state (prior / prompted / baked), we read belief = logP(correct)−logP(wrong)
in ONE forward pass (reusing propagation's belief machinery) and call a "Yes" response `belief > τ`
(τ=0). Then:

    hit rate H(d)        = P(Yes | provable,     depth d)
    false-alarm FA(d)    = P(Yes | non-provable, depth d)
    d′(d) = Φ⁻¹(H) − Φ⁻¹(FA)     criterion c(d) = −(Φ⁻¹(H)+Φ⁻¹(FA))/2     bacc = (H + 1−FA)/2

A pure yes-bias gives H≈FA ⇒ d′≈0 regardless of how much it says "Yes" — yes-saturation is exposed.
A log-linear (k+0.5)/(n+1) correction keeps Φ⁻¹ finite at degenerate rates. A threshold-free AUROC
companion (rank of true vs false beliefs) is reported so the headline doesn't hinge on τ. We also
split the false pool by `neg_type` (converse / cross / missing_edge) to see WHICH negative fails.
`prop_distance_dprime_{state}` = deepest contiguous d≥1 with d′(d) ≥ 1.0.

The probe bank is auxiliary eval — it never enters the KL or the gate.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from bakery.eval.metrics.propagation import _answer_logprob, _paired_belief_batched   # reuse the belief / logit-shift primitive
from bakery.eval.registry import EvalContext, MetricResult, register_metric
from bakery.prompts import build_prefix_ids

TAU = 0.0                 # a "Yes" response is belief > TAU
THRESH_DPRIME = 1.0       # propagation distance = deepest contiguous d>=1 with d' >= this
THRESH_BACC = 0.7
_NEG_TYPES = ("converse", "cross", "missing_edge")

# Acklam's rational approximation of the inverse normal CDF Φ⁻¹ (≈1e-9 accuracy) — avoids a scipy
# dependency. Valid on the OPEN interval (0,1); the log-linear correction guarantees we stay inside.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)


def _phi_inv(p: float) -> float:
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((_C[0]*q+_C[1])*q+_C[2])*q+_C[3])*q+_C[4])*q+_C[5]) / \
               ((((_D[0]*q+_D[1])*q+_D[2])*q+_D[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((_A[0]*r+_A[1])*r+_A[2])*r+_A[3])*r+_A[4])*r+_A[5])*q / \
               (((((_B[0]*r+_B[1])*r+_B[2])*r+_B[3])*r+_B[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((_C[0]*q+_C[1])*q+_C[2])*q+_C[3])*q+_C[4])*q+_C[5]) / \
           ((((_D[0]*q+_D[1])*q+_D[2])*q+_D[3])*q+1)


def _rate(k: int, n: int) -> float:
    """Log-linear corrected rate (k+0.5)/(n+1) — keeps Φ⁻¹ finite when k∈{0,n}."""
    return (k + 0.5) / (n + 1)


def _auroc(pos, neg) -> float:
    """Threshold-free separability = P(belief(true) > belief(false)), ties = 0.5. NaN if a pool empty."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(pos) * len(neg))


def _yesness_per_probe(bundle, probes, system_text, ctx_mgr, device,
                       *, batch_probes=False, probe_batch_size=0) -> list:
    """Decision axis yesness = logP(' Yes') − logP(' No') per probe (NOT correctness-signed).

    This is the signal-detection variable: a "Yes" RESPONSE is yesness > τ, independent of whether
    Yes is the correct answer — so hit rate (yes on provable) and false-alarm (yes on non-provable)
    are comparable across forward and negative probes. Reuses propagation's tokenization-robust
    `_answer_logprob` (and hence the verified logit→token shift).

    Default (`batch_probes=False`) is the original unbatched loop — UNCHANGED. `batch_probes=True`
    runs the SAME math batched via propagation's `_paired_belief_batched` (opt-in; bf16 tol-equivalent).
    """
    if batch_probes:
        return _paired_belief_batched(
            bundle, probes, system_text, ctx_mgr, device,
            pos_of=lambda pr: " Yes", neg_of=lambda pr: " No", chunk=probe_batch_size,
        )
    tok = bundle.tokenizer
    out = []
    with torch.no_grad():
        with ctx_mgr() as model:
            for pr in probes:
                prefix = build_prefix_ids(tok, system_text, pr["question"])
                lp_yes = _answer_logprob(model, tok, prefix, " Yes", device)
                lp_no = _answer_logprob(model, tok, prefix, " No", device)
                out.append(lp_yes - lp_no)
    return out


def _depth(pr) -> int:
    """The depth a probe is binned/paired at. `match_depth` is the explicit pairing depth for both
    true and false probes; fall back to `proof_depth` (theorems) then `hop` for older banks. A
    non-theorem has `proof_depth=None` (no proof) but still a `match_depth` (its control depth)."""
    for key in ("match_depth", "proof_depth", "hop"):
        v = pr.get(key)
        if v is not None:
            return int(v)
    return 1


def _provable(pr) -> bool:
    if "provable" in pr:
        return bool(pr["provable"])
    return str(pr.get("form", "forward")) == "forward"


@register_metric("dprime")
def dprime(ctx: EvalContext) -> MetricResult:
    data_cfg = ctx.run_cfg.data
    probe_path = data_cfg.get("probe_bank") if isinstance(data_cfg, dict) else None
    if not probe_path or not Path(probe_path).exists():
        return MetricResult(name="dprime", value=float("nan"))
    probes = json.loads(Path(probe_path).read_text()).get("probes", [])
    if not probes:
        return MetricResult(name="dprime", value=float("nan"))

    # Held-out filter: a curriculum builder records the relations it TRAINED on in
    # stats["pairing"]["trained_relations"]. Exclude those from the depth>=2 cells so the reported d′
    # is genuine HELD-OUT propagation; depth-1 is kept whole as the (trained) recall baseline. Absent
    # for other experiments -> empty set -> no exclusion (behaviour byte-identical).
    try:
        trained_rel = (ctx.data.stats or {}).get("pairing", {}).get("trained_relations") or []
    except Exception:
        trained_rel = []
    trained = {(r[0], r[1]) for r in trained_rel}

    bundle = ctx.bundle
    bundle.peft_model.eval()
    device = ctx.device
    u_text = ctx.data.prompts.get("base_u", "")
    baked_text = ctx.data.prompts.get("baked", "")
    _ecfg = getattr(ctx.run_cfg, "eval", None)                       # tolerate minimal fake run_cfgs
    bp = bool(getattr(_ecfg, "batch_probes", False))                 # opt-in speedup (default off)
    pbs = int(getattr(_ecfg, "probe_batch_size", 0) or 0)

    pp = {
        "prior": _yesness_per_probe(bundle, probes, baked_text, bundle.base, device, batch_probes=bp, probe_batch_size=pbs),
        "prompted": _yesness_per_probe(bundle, probes, u_text, bundle.base, device, batch_probes=bp, probe_batch_size=pbs),
        "baked": _yesness_per_probe(bundle, probes, baked_text, bundle.baked, device, batch_probes=bp, probe_batch_size=pbs),
    }

    depths = sorted({_depth(pr) for pr in probes})
    idx_true = defaultdict(list)            # depth -> probe indices (provable)
    idx_false = defaultdict(lambda: defaultdict(list))   # depth -> neg_type|'all' -> indices
    for k, pr in enumerate(probes):
        d = _depth(pr)
        if d >= 2 and (pr.get("subj"), pr.get("obj")) in trained:
            continue                            # trained at depth>=2 -> not held out -> exclude
        if _provable(pr):
            idx_true[d].append(k)
        else:
            idx_false[d]["all"].append(k)
            idx_false[d][pr.get("neg_type") or "other"].append(k)

    extra: dict = {"depths": depths, "neg_types": list(_NEG_TYPES), "tau": TAU,
                   "correction": "loglinear_0.5", "dprime_skipped": {}}

    def _cell(vals, ti, fi):
        """(d', hit, fa, criterion, bacc, auroc) for a true/false index pair; d'/c NaN if a pool empty."""
        n_t, n_f = len(ti), len(fi)
        if n_t == 0 or n_f == 0:
            return (float("nan"),) * 5 + (_auroc([vals[i] for i in ti], [vals[i] for i in fi]),)
        hits = sum(1 for i in ti if vals[i] > TAU)
        fas = sum(1 for i in fi if vals[i] > TAU)
        H, FA = _rate(hits, n_t), _rate(fas, n_f)
        zh, zf = _phi_inv(H), _phi_inv(FA)
        auroc = _auroc([vals[i] for i in ti], [vals[i] for i in fi])
        return (zh - zf, H, FA, -(zh + zf) / 2.0, (H + 1 - FA) / 2.0, auroc)

    for state, vals in pp.items():
        for d in depths:
            ti, fi = idx_true[d], idx_false[d]["all"]
            dp, H, FA, crit, bacc, auroc = _cell(vals, ti, fi)
            extra[f"dprime_{state}_d{d}"] = dp
            extra[f"hit_{state}_d{d}"] = H
            extra[f"fa_{state}_d{d}"] = FA
            extra[f"crit_{state}_d{d}"] = crit
            extra[f"bacc_{state}_d{d}"] = bacc
            extra[f"auroc_{state}_d{d}"] = auroc
            extra[f"n_true_{state}_d{d}"] = len(ti)
            extra[f"n_false_{state}_d{d}"] = len(fi)
            if (len(ti) == 0 or len(fi) == 0) and state == "baked":
                extra["dprime_skipped"][f"d{d}_baked"] = "no_positives" if not ti else "no_negatives"
            for t in _NEG_TYPES:
                fit = idx_false[d][t]
                dpt, _, fat, _, _, aut = _cell(vals, ti, fit)
                extra[f"dprime_{state}_d{d}_{t}"] = dpt
                extra[f"fa_{state}_d{d}_{t}"] = fat
                extra[f"auroc_{state}_d{d}_{t}"] = aut

        extra[f"prop_distance_dprime_{state}"] = _prop_distance(
            extra, state, depths, "dprime", THRESH_DPRIME)
        extra[f"prop_distance_bacc_{state}"] = _prop_distance(
            extra, state, depths, "bacc", THRESH_BACC)

    # Headline = mean baked d′ over HELD-OUT depths (>=2) when a curriculum split is present — that is
    # the propagation we care about; depth-1 is the trained recall baseline, reported separately and
    # NOT folded in. With no split (other experiments) fall back to mean over all d>=1 (unchanged).
    held_depths = [d for d in depths if d >= 2]
    hl_depths = held_depths if (trained and held_depths) else [d for d in depths if d >= 1]
    baked_dps = [extra[f"dprime_baked_d{d}"] for d in hl_depths]
    finite = [x for x in baked_dps if isinstance(x, float) and math.isfinite(x)]
    headline = sum(finite) / len(finite) if finite else float("nan")
    extra["dprime_baked_d1_baseline"] = extra.get("dprime_baked_d1", float("nan"))
    extra["headline_depths"] = hl_depths
    extra["n_trained_relations"] = len(trained)

    return MetricResult(
        name="dprime",
        value=headline,
        matrix=[[extra[f"hit_baked_d{d}"] for d in depths],
                [extra[f"fa_baked_d{d}"] for d in depths],
                [extra[f"dprime_baked_d{d}"] for d in depths]],
        labels=["hit_baked", "fa_baked", "dprime_baked"],
        extra=extra,
    )


def _prop_distance(extra, state, depths, key, thresh) -> int:
    """Deepest d>=1 reached contiguously from d=1 with `key`(d) finite and >= thresh."""
    out = 0
    for d in sorted(x for x in depths if x >= 1):
        v = extra.get(f"{key}_{state}_d{d}")
        if isinstance(v, float) and math.isfinite(v) and v >= thresh:
            out = d
        else:
            break
    return out
