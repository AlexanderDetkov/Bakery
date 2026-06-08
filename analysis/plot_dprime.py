"""Plot signal-detection (d′) propagation curves from bake_fact/bake_logic runs (metrics.json ONLY).

The `dprime` metric logs, per eval step, flat scalars `dprime.<stat>_<state>_d<d>` (stat in
dprime|hit|fa|crit|bacc|auroc; state in prior|prompted|baked) and per-neg_type variants. This
plotter turns them into the readings that matter:

  * dprime    — d′ vs proof depth per state (and bake-baked vs sft-baked across runs): how FAR an
                injected fact propagates as DISCRIMINATION, not response bias.
  * roc       — hit-rate vs false-alarm vs depth (baked): yes-saturation is H≈FA collapsing → d′≈0.
  * criterion — response bias c vs depth (baked yes-bias shows c≪0).
  * negtype   — baked d′ per negative type (converse / cross / missing_edge): WHICH negative fails.
  * grokking  — d′ at one depth vs epoch with eval_kl overlaid: does discrimination rise AFTER
                eval_kl plateaus?

`--stat {dprime,auroc,bacc}` (default `auroc`) selects WHICH per-depth statistic the dprime/negtype/grokking
modes plot. d′ passes hit/FA through Φ⁻¹, which AMPLIFIES variance near 0/1 (one flipped probe in a ~16+16
cell moves d′ by ±0.2–0.5); AUROC (rank-based, threshold-free) and bacc (linear in hit/FA) are computed from
the SAME forward pass but are far lower variance. Lead with AUROC; use `--stat dprime` to see the noisy view.

Imports only matplotlib + analysis.load (never the training stack — see test_analysis_isolation).

    python -m analysis.plot_dprime dprime results/bake_fact/lw_alpha-bake-8b:bake \
        results/bake_fact/lw_alpha-sft-8b:sft --stat auroc --out results/bake_fact/_fig_auroc.png
    python -m analysis.plot_dprime grokking results/bake_fact/lw_alpha-bake-8b --depth 3 \
        --stat auroc --out results/bake_fact/_fig_grok_d3.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.load import load_run
from analysis.plot_grokking import _save, _xy

THRESH = 1.0

# Per-stat display: (reference line, y-label, draw a zero line?). The metric logs all three under
# dprime.<stat>_<state>_d<d> (dprime.py:189-209), so switching stat is just a key-prefix change.
_STAT_CFG = {
    "dprime": (1.0, "d′  =  Φ⁻¹(hit) − Φ⁻¹(false-alarm)", True),
    "auroc": (0.5, "AUROC  (rank-based; 0.5 = chance)", False),
    "bacc": (0.5, "balanced accuracy  (0.5 = chance)", False),
}


def _final_by_depth(r, prefix):
    """{depth: final value} for keys `prefix` + an integer depth (ignores `_negtype` suffixes)."""
    out = {}
    for key, series in r.metrics.items():
        if key.startswith(prefix) and isinstance(series, list):
            tail = key[len(prefix):]
            if tail.isdigit():
                vals = [x for x in series if isinstance(x, (int, float))]
                if vals:
                    out[int(tail)] = vals[-1]
    return dict(sorted(out.items()))


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def dprime(run_specs, out, stat="auroc"):
    """<stat> vs depth. Single run → prior/prompted/baked; multiple runs → baked <stat> per run."""
    ref, ylab, zero_line = _STAT_CFG[stat]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    if len(run_specs) == 1:
        run_dir, _, label = run_specs[0].partition(":")
        r = load_run(run_dir)
        for state, col, ls in [("prior", "0.6", ":"), ("prompted", "C0", "--"), ("baked", "C3", "-")]:
            pts = _final_by_depth(r, f"dprime.{stat}_{state}_d")
            if pts:
                ax.plot(list(pts), list(pts.values()), ls, marker="o", color=col, label=state)
        ax.set_title(f"{stat} vs proof depth — {label or Path(run_dir).name}")
    else:
        for i, spec in enumerate(run_specs):
            run_dir, _, label = spec.partition(":")
            pts = _final_by_depth(load_run(run_dir), f"dprime.{stat}_baked_d")
            if pts:
                ax.plot(list(pts), list(pts.values()), "-", marker="o", color=f"C{i}",
                        label=(label or Path(run_dir).name) + " (baked)")
        ax.set_title(f"{stat} vs proof depth (baked)")
    ax.axhline(ref, color="0.8", lw=0.8, ls="--", zorder=0)
    if zero_line:
        ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("proof depth d  (modus-ponens steps)")
    ax.set_ylabel(ylab)
    ax.legend(fontsize=8)
    return _save(fig, out)


def roc(spec, out):
    """Hit-rate and false-alarm-rate vs depth (baked) — yes-saturation = H and FA converging."""
    plt = _plt()
    run_dir, _, label = spec.partition(":")
    r = load_run(run_dir)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for stat, col, lab in [("hit", "C2", "hit rate P(Yes|true)"), ("fa", "C3", "false-alarm P(Yes|false)")]:
        pts = _final_by_depth(r, f"dprime.{stat}_baked_d")
        if pts:
            ax.plot(list(pts), list(pts.values()), "-", marker="o", color=col, label=lab)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("proof depth d")
    ax.set_ylabel("response rate (baked)")
    ax.set_title(f"Hit vs false-alarm — {label or Path(run_dir).name}")
    ax.legend(fontsize=8)
    return _save(fig, out)


def criterion(spec, out):
    """Response bias c vs depth, per state (c ≪ 0 = a Yes-bias)."""
    plt = _plt()
    run_dir, _, label = spec.partition(":")
    r = load_run(run_dir)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for state, col in [("prior", "0.6"), ("prompted", "C0"), ("baked", "C3")]:
        pts = _final_by_depth(r, f"dprime.crit_{state}_d")
        if pts:
            ax.plot(list(pts), list(pts.values()), "-", marker="o", color=col, label=state)
    ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("proof depth d")
    ax.set_ylabel("criterion c  (negative = Yes-bias)")
    ax.set_title(f"Response bias — {label or Path(run_dir).name}")
    ax.legend(fontsize=8)
    return _save(fig, out)


def negtype(spec, out, stat="auroc"):
    """Baked <stat> per negative type (converse / cross / missing_edge) vs depth."""
    ref, _, _ = _STAT_CFG[stat]
    plt = _plt()
    run_dir, _, label = spec.partition(":")
    r = load_run(run_dir)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for t, col in [("converse", "C0"), ("cross", "C1"), ("missing_edge", "C3")]:
        pts = {}
        pref, suf = f"dprime.{stat}_baked_d", f"_{t}"
        for key, series in r.metrics.items():
            if key.startswith(pref) and key.endswith(suf):
                mid = key[len(pref):-len(suf)]
                if mid.isdigit():
                    vals = [x for x in series if isinstance(x, (int, float))]
                    if vals:
                        pts[int(mid)] = vals[-1]
        if pts:
            pts = dict(sorted(pts.items()))
            ax.plot(list(pts), list(pts.values()), "-", marker="o", color=col, label=t)
    ax.axhline(ref, color="0.8", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("proof depth d")
    ax.set_ylabel(f"baked {stat} vs this negative type")
    ax.set_title(f"{stat} by negative type — {label or Path(run_dir).name}")
    ax.legend(fontsize=8)
    return _save(fig, out)


def grokking(spec, out, depth, stat="auroc"):
    """<stat> at a fixed depth vs epoch (left) with eval_kl overlaid (right, log)."""
    ref, _, _ = _STAT_CFG[stat]
    plt = _plt()
    run_dir, _, label = spec.partition(":")
    r = load_run(run_dir)
    fig, axL = plt.subplots(figsize=(8, 4.8))
    for state, col in [("prompted", "C0"), ("baked", "C3")]:
        xs, ys = _xy(r, f"dprime.{stat}_{state}_d{depth}")
        if xs:
            axL.plot(xs, ys, "-", marker="o", markersize=3, color=col, label=f"{stat} {state} (d={depth})")
    axL.axhline(ref, color="0.8", lw=0.8, ls="--", zorder=0)
    axL.set_xlabel("epoch")
    axL.set_ylabel(f"{stat} at depth {depth}")
    axR = axL.twinx()
    xs, ys = _xy(r, "eval_kl")
    if xs:
        axR.plot(xs, ys, "-", color="C0", alpha=0.5, label="eval_kl")
        try:
            axR.set_yscale("log")
        except ValueError:
            pass
    axR.set_ylabel("eval_kl (log)")
    axL.set_title(f"Grokking of discrimination [{stat}] (d={depth}) — {label or Path(run_dir).name}")
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL + hR, lL + lR, fontsize=8, loc="center right")
    return _save(fig, out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["dprime", "roc", "criterion", "negtype", "grokking"])
    ap.add_argument("run_specs", nargs="+", help="run_dir:label")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=2, help="(grokking mode) depth to track vs epoch")
    ap.add_argument("--stat", choices=["dprime", "auroc", "bacc"], default="auroc",
                    help="per-depth statistic for dprime/negtype/grokking modes (default auroc, lower variance)")
    a = ap.parse_args(argv)
    if a.mode == "dprime":
        print(dprime(a.run_specs, a.out, a.stat))
    elif a.mode == "roc":
        print(roc(a.run_specs[0], a.out))
    elif a.mode == "criterion":
        print(criterion(a.run_specs[0], a.out))
    elif a.mode == "negtype":
        print(negtype(a.run_specs[0], a.out, a.stat))
    else:
        print(grokking(a.run_specs[0], a.out, a.depth, a.stat))


if __name__ == "__main__":
    main()
