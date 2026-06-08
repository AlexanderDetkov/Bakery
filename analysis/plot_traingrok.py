"""Train-vs-eval grokking diagnostic for bake_theorem_qa / bake_logic runs (metrics.json ONLY).

The question: does HELD-OUT generalization (baked d′ at proof-depth ≥ 2) rise LATE — after the
training loss (train_kl) and the TRAINED-depth recall (baked d′/bacc at d1) have already saturated?
That late, post-plateau jump is the grokking signature. If instead d2 settles around when train_kl
plateaus and d3 never moves, generalization simply tracks the loss (no grokking).

One subplot per run: LEFT axis = baked d′ at d1 (trained recall) / d2 / d3 (held-out propagation),
with a faint bacc(d1) "train accuracy" line; RIGHT axis (log) = train_kl + eval_kl. A vertical marker
flags the epoch eval_kl first comes within 5% of its final value (the loss-plateau onset) so you can
read whether any d′ curve climbs to its right.

Imports only matplotlib + analysis.load — never the training stack (tests/test_analysis_isolation.py).

    python -m analysis.plot_traingrok \
        results/bake_theorem_qa/qa-bake-n1-s10:n=1 \
        results/bake_theorem_qa/qa-bake-n2-s11:n=2 \
        --out results/bake_theorem_qa/_fig_traingrok.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.load import load_run
from analysis.plot_grokking import _save, _xy


def _plateau_epoch(xs, ys, frac=0.05):
    """First epoch at which eval_kl is within `frac` of its final value (loss-plateau onset)."""
    if not ys:
        return None
    final = ys[-1]
    for x, y in zip(xs, ys):
        if abs(y - final) <= frac * max(abs(final), 1e-9):
            return x
    return None


# Per-stat display: (reference line, y-limits, y-label, draw a zero line?). d′ is Φ⁻¹-amplified (noisy);
# auroc (rank-based) / bacc (linear) are computed from the same forward pass but far lower variance.
_STAT_CFG = {
    "dprime": (1.0, (-1.0, 2.0), "baked d′", True),
    "auroc": (0.5, (0.0, 1.05), "baked AUROC", False),
    "bacc": (0.5, (0.0, 1.05), "baked balanced accuracy", False),
}


def panel(run_specs, out, stat="auroc"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref, ylim, ylab, zero_line = _STAT_CFG[stat]
    n = len(run_specs)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 4.8), squeeze=False)
    axes = axes[0]

    for ax, spec in zip(axes, run_specs):
        run_dir, _, label = spec.partition(":")
        r = load_run(run_dir)
        label = label or Path(run_dir).name

        # LEFT: baked <stat> per depth (trained d1 vs held-out d2/d3) + bacc(d1) as "train accuracy".
        depth_styles = [
            (f"dprime.{stat}_baked_d1", f"{stat} d1 (TRAINED recall)", "C0", "-", 2.2),
            (f"dprime.{stat}_baked_d2", f"{stat} d2 (held-out)", "C1", "-", 2.0),
            (f"dprime.{stat}_baked_d3", f"{stat} d3 (held-out)", "C3", "-", 2.0),
        ]
        for key, lab, col, ls, lw in depth_styles:
            xs, ys = _xy(r, key)
            if xs:
                ax.plot(xs, ys, ls, color=col, lw=lw, label=lab)
        if stat != "bacc":     # bacc(d1) as an interpretable "train accuracy" reference line
            xs, ys = _xy(r, "dprime.bacc_baked_d1")
            if xs:
                ax.plot(xs, ys, ":", color="0.4", lw=1.2, alpha=0.8, label="bacc d1 (train acc)")

        ax.axhline(ref, color="0.8", lw=0.8, ls="--", zorder=0)
        if zero_line:
            ax.axhline(0.0, color="0.85", lw=0.8, ls="-", zorder=0)
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{ylab}  (+ bacc d1)")
        ax.set_title(f"{label}  ({Path(run_dir).name})")
        ax.set_ylim(*ylim)

        # RIGHT (log): train_kl + eval_kl.
        axR = ax.twinx()
        for key, lab, col in [("train_kl", "train_kl (loss)", "0.45"),
                              ("eval_kl", "eval_kl (held-out)", "C2")]:
            xs, ys = _xy(r, key)
            if xs:
                axR.plot(xs, ys, "--", color=col, lw=1.6, label=lab)
        axR.set_yscale("log")
        axR.set_ylabel("KL (log)")

        # Mark the eval_kl plateau onset.
        ex, ey = _xy(r, "eval_kl")
        pe = _plateau_epoch(ex, ey)
        if pe is not None:
            ax.axvline(pe, color="purple", lw=1.0, ls=":", alpha=0.7)
            ytxt = ylim[1] - 0.05 * (ylim[1] - ylim[0])
            ax.text(pe, ytxt, f" eval_kl plateau ~ep{pe}", color="purple", fontsize=8, va="top")

        # Merge legends from both axes.
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axR.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right", framealpha=0.9)

    fig.suptitle(f"Train vs held-out dynamics [{stat}] — is held-out generalization a LATE (grokking) "
                 f"rise after the loss plateau?", fontsize=10)
    return _save(fig, out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_specs", nargs="+", help="run_dir[:label] ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stat", choices=["dprime", "auroc", "bacc"], default="auroc",
                    help="per-depth statistic (default auroc; lower variance than the Φ⁻¹-amplified d′)")
    a = ap.parse_args(argv)
    print(panel(a.run_specs, a.out, a.stat))


if __name__ == "__main__":
    main()
