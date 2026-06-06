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

Imports only matplotlib + analysis.load (never the training stack — see test_analysis_isolation).

    python -m analysis.plot_dprime dprime results/bake_fact/lw_alpha-bake-8b:bake \
        results/bake_fact/lw_alpha-sft-8b:sft --out results/bake_fact/_fig_dprime.png
    python -m analysis.plot_dprime grokking results/bake_fact/lw_alpha-bake-8b --depth 3 \
        --out results/bake_fact/_fig_dprime_grok_d3.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.load import load_run
from analysis.plot_grokking import _save, _xy

THRESH = 1.0


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


def dprime(run_specs, out):
    """d′ vs depth. Single run → prior/prompted/baked; multiple runs → baked d′ per run."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    if len(run_specs) == 1:
        run_dir, _, label = run_specs[0].partition(":")
        r = load_run(run_dir)
        for state, col, ls in [("prior", "0.6", ":"), ("prompted", "C0", "--"), ("baked", "C3", "-")]:
            pts = _final_by_depth(r, f"dprime.dprime_{state}_d")
            if pts:
                ax.plot(list(pts), list(pts.values()), ls, marker="o", color=col, label=state)
        ax.set_title(f"d′ vs proof depth — {label or Path(run_dir).name}")
    else:
        for i, spec in enumerate(run_specs):
            run_dir, _, label = spec.partition(":")
            pts = _final_by_depth(load_run(run_dir), "dprime.dprime_baked_d")
            if pts:
                ax.plot(list(pts), list(pts.values()), "-", marker="o", color=f"C{i}",
                        label=(label or Path(run_dir).name) + " (baked)")
        ax.set_title("d′ vs proof depth (baked)")
    ax.axhline(THRESH, color="0.8", lw=0.8, ls="--", zorder=0)
    ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("proof depth d  (modus-ponens steps)")
    ax.set_ylabel("d′  =  Φ⁻¹(hit) − Φ⁻¹(false-alarm)")
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


def negtype(spec, out):
    """Baked d′ per negative type (converse / cross / missing_edge) vs depth."""
    plt = _plt()
    run_dir, _, label = spec.partition(":")
    r = load_run(run_dir)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for t, col in [("converse", "C0"), ("cross", "C1"), ("missing_edge", "C3")]:
        pts = {}
        pref, suf = "dprime.dprime_baked_d", f"_{t}"
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
    ax.axhline(THRESH, color="0.8", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("proof depth d")
    ax.set_ylabel("baked d′ vs this negative type")
    ax.set_title(f"d′ by negative type — {label or Path(run_dir).name}")
    ax.legend(fontsize=8)
    return _save(fig, out)


def grokking(spec, out, depth):
    """d′ at a fixed depth vs epoch (left) with eval_kl overlaid (right, log)."""
    plt = _plt()
    run_dir, _, label = spec.partition(":")
    r = load_run(run_dir)
    fig, axL = plt.subplots(figsize=(8, 4.8))
    for state, col in [("prompted", "C0"), ("baked", "C3")]:
        xs, ys = _xy(r, f"dprime.dprime_{state}_d{depth}")
        if xs:
            axL.plot(xs, ys, "-", marker="o", markersize=3, color=col, label=f"d′ {state} (d={depth})")
    axL.axhline(THRESH, color="0.8", lw=0.8, ls="--", zorder=0)
    axL.set_xlabel("epoch")
    axL.set_ylabel(f"d′ at depth {depth}")
    axR = axL.twinx()
    xs, ys = _xy(r, "eval_kl")
    if xs:
        axR.plot(xs, ys, "-", color="C0", alpha=0.5, label="eval_kl")
        try:
            axR.set_yscale("log")
        except ValueError:
            pass
    axR.set_ylabel("eval_kl (log)")
    axL.set_title(f"Grokking of discrimination (d={depth}) — {label or Path(run_dir).name}")
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
    a = ap.parse_args(argv)
    if a.mode == "dprime":
        print(dprime(a.run_specs, a.out))
    elif a.mode == "roc":
        print(roc(a.run_specs[0], a.out))
    elif a.mode == "criterion":
        print(criterion(a.run_specs[0], a.out))
    elif a.mode == "negtype":
        print(negtype(a.run_specs[0], a.out))
    else:
        print(grokking(a.run_specs[0], a.out, a.depth))


if __name__ == "__main__":
    main()
