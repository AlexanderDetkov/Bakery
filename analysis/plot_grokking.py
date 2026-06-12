"""Plot grokking-style training curves from bake_fact runs (reads metrics.json ONLY).

The `propagation` metric logs, per eval step, accuracy scalars `propagation.{form}_acc_{state}`
(form in forward|converse|negation; state in prior|prompted|baked) alongside `eval_kl`/`train_kl`.
This plotter overlays them vs epoch to expose GROKKING: the distillation loss (eval_kl) plateaus
early while `converse_acc_baked` may keep climbing (late generalization, as the inverse map does in
~/Invertibility) or stay flat (a real LoRA/distillation limit). Imports only matplotlib +
analysis.load — never the training stack (enforced by tests/test_analysis_isolation.py).

    python -m analysis.plot_grokking panel results/bake_fact/e1a-veld8b-long-wd0:wd0 \
        --out results/bake_fact/_fig_grok_e1a.png
    python -m analysis.plot_grokking compare --metric propagation.converse_acc_baked \
        results/bake_fact/e1a-veld8b-long-wd0:wd0 results/bake_fact/e1b-veld8b-long-wd05:wd0.05 \
        --out results/bake_fact/_fig_grok_converse.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.load import load_run


def _xy(run, key):
    """Aligned (epoch, value) pairs for a metric key, dropping non-numeric entries."""
    xs_all = run.metrics.get("epochs") or []
    ys_all = run.metrics.get(key) or []
    xs, ys = [], []
    for i in range(min(len(xs_all), len(ys_all))):
        if isinstance(ys_all[i], (int, float)):
            xs.append(xs_all[i])
            ys.append(ys_all[i])
    return xs, ys


def _save(fig, out):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    return str(out_path)


def panel(spec, out):
    """One run: per-form accuracy (left axis) + eval_kl/train loss (right, log) vs epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir, _, label = spec.partition(":")
    r = load_run(run_dir)
    label = label or Path(run_dir).name

    fig, axL = plt.subplots(figsize=(8, 4.8))
    acc_curves = [
        ("propagation.forward_acc_baked", "forward acc (baked)", "C2", "-", "o"),
        ("propagation.converse_acc_baked", "converse acc (baked)", "C3", "-", "o"),
        ("propagation.converse_acc_prompted", "converse acc (prompted ref)", "C3", ":", None),
        ("propagation.forward_acc_prompted", "forward acc (prompted ref)", "C2", ":", None),
    ]
    for key, lab, col, ls, mk in acc_curves:
        xs, ys = _xy(r, key)
        if xs:
            axL.plot(xs, ys, ls, color=col, marker=mk, markersize=3, label=lab)
    axL.set_xlabel("epoch")
    axL.set_ylabel("accuracy (fraction favouring correct answer)")
    axL.set_ylim(-0.03, 1.03)
    axL.axhline(0.5, color="0.8", lw=0.8, zorder=0)

    axR = axL.twinx()
    for key, lab, col in [("eval_kl", "eval_kl", "C0"), ("train_kl", "train loss", "0.6")]:
        xs, ys = _xy(r, key)
        if xs:
            axR.plot(xs, ys, "-", color=col, alpha=0.7, label=lab)
    axR.set_ylabel("KL / training loss (log scale)")
    try:
        axR.set_yscale("log")
    except ValueError:
        pass

    axL.set_title(f"Grokking panel — {label}")
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL + hR, lL + lR, fontsize=8, loc="center right")
    return _save(fig, out)


def compare(run_specs, out, metric):
    """Overlay one metric key vs epoch across runs (e.g. converse_acc_baked for wd / type arms)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, spec in enumerate(run_specs):
        run_dir, _, label = spec.partition(":")
        r = load_run(run_dir)
        xs, ys = _xy(r, metric)
        if xs:
            ax.plot(xs, ys, "-", marker="o", markersize=3, color=f"C{i}",
                    label=label or Path(run_dir).name)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    if metric.endswith("_acc_baked") or "_acc_" in metric:
        ax.set_ylim(-0.03, 1.03)
        ax.axhline(0.5, color="0.8", lw=0.8, zorder=0)
    ax.set_title(f"{metric} vs epoch")
    ax.legend(fontsize=8)
    return _save(fig, out)


def distance(spec, out):
    """Final-epoch HELD-OUT forward accuracy vs propagation distance d, for prior/prompted/baked.

    Reads `propagation.forward_acc_{state}_heldout_d{d}` (genuine propagation — composed consequences
    never stated in the trajectories). The largest d above 0.5 is the propagation distance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir, _, label = spec.partition(":")
    r = load_run(run_dir)
    label = label or Path(run_dir).name
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for state in ("prior", "prompted", "baked"):
        pref = f"propagation.forward_acc_{state}_heldout_d"
        pts = []
        for key, series in r.metrics.items():
            if key.startswith(pref) and isinstance(series, list):
                vals = [x for x in series if isinstance(x, (int, float))]
                if vals:
                    pts.append((int(key[len(pref):]), vals[-1]))
        if pts:
            pts.sort()
            ax.plot([d for d, _ in pts], [y for _, y in pts], marker="o", label=state)
    ax.set_ylim(-0.03, 1.03)
    ax.axhline(0.5, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel("propagation distance d  (held-out composed hops)")
    ax.set_ylabel("single-pass forward accuracy (held-out)")
    ax.set_title(f"Propagation distance — {label}")
    ax.legend()
    return _save(fig, out)


def depth_panels(specs, out, metric="bacc", max_depth=3):
    """Per-run grokking panel for the `theorem_qa` runs (the `dprime` metric).

    One subplot per run: per-DEPTH baked generalization vs epoch — d1 = the TRAINED-relation recall
    ("train accuracy"), d2/d3/… = HELD-OUT propagation — overlaid (right axis, log) on train loss +
    eval_kl. Exposes grokking: does held-out d2/d3 keep CLIMBING after train_kl and d1 (trained recall)
    have already saturated? `metric` in {"bacc" (balanced accuracy, 0.5 = chance), "dprime" (d′,
    0 = chance)}. epoch on a LOG axis so late generalization is visible. Reads metrics.json only.
    """
    import re

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for s in specs:
        rd, _, lab = s.partition(":")
        runs.append((load_run(rd), lab or Path(rd).name))
    fig, axes = plt.subplots(1, len(runs), figsize=(5.6 * len(runs), 4.8), squeeze=False)
    for ax, (r, label) in zip(axes[0], runs):
        depths = sorted({int(m.group(1)) for k in r.metrics
                         for m in [re.match(rf"dprime\.{metric}_baked_d(\d+)(?:_baseline)?$", k)] if m})
        depths = [d for d in depths if d <= max_depth]   # deeper cells have too few held-out probes (noise)
        for d in depths:
            key = f"dprime.{metric}_baked_d{d}"
            if key not in r.metrics:
                key = f"dprime.{metric}_baked_d{d}_baseline"   # d′ records d1 as the baseline
            xs, ys = _xy(r, key)
            if not xs:
                continue
            trained = d == 1
            ax.plot(xs, ys, "--s" if trained else "-o", color=f"C{d}", markersize=3,
                    label=f"d{d} {'(trained recall)' if trained else '(held-out)'}")
        ax.set_xlabel("epoch")
        ax.set_xscale("log")
        if metric == "bacc":
            ax.set_ylabel("balanced accuracy"); ax.set_ylim(0.35, 1.03); ax.axhline(0.5, color="0.8", lw=0.8)
        else:
            ax.set_ylabel("d′ (bias-immune)"); ax.axhline(0.0, color="0.8", lw=0.8)
        axR = ax.twinx()
        for key, lab2, col in [("train_kl", "train loss", "0.5"), ("eval_kl", "eval_kl", "C0")]:
            xs, ys = _xy(r, key)
            if xs:
                axR.plot(xs, ys, "-", color=col, alpha=0.55, label=lab2)
        axR.set_ylabel("KL / train loss (log)")
        try:
            axR.set_yscale("log")
        except ValueError:
            pass
        ep = r.metrics.get("epochs") or [None]
        ax.set_title(f"{label}  (ep≤{ep[-1]}, {r.status})", fontsize=9)
        hL, lL = ax.get_legend_handles_labels()
        hR, lR = axR.get_legend_handles_labels()
        ax.legend(hL + hR, lL + lR, fontsize=7, loc="lower right")
    fig.suptitle(f"theorem_qa sampled bake — per-depth {metric}: trained recall (d1) vs held-out (d≥2)",
                 fontsize=11)
    return _save(fig, out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["panel", "compare", "distance", "depths"])
    ap.add_argument("run_specs", nargs="+", help="run_dir:label")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="propagation.converse_acc_baked",
                    help="(compare mode) metric key to overlay; (depths mode) 'bacc' | 'dprime'")
    a = ap.parse_args(argv)
    if a.mode == "panel":
        print(panel(a.run_specs[0], a.out))
    elif a.mode == "distance":
        print(distance(a.run_specs[0], a.out))
    elif a.mode == "depths":
        print(depth_panels(a.run_specs, a.out, a.metric if a.metric in ("bacc", "dprime") else "bacc"))
    else:
        print(compare(a.run_specs, a.out, a.metric))


if __name__ == "__main__":
    main()
