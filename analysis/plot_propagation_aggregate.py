"""Master aggregate: prompted vs baked per-family AUROC across the multi-graph sweep (metrics.json ONLY).

One subplot per proof depth; within each, grouped bars over the negative families
{ALL, converse, cross, missing_edge}; bars = prompted / baked(n=1) / baked(n=2); error bars =
population sd across the input runs (the 12 graphs by default). Chance = 0.5. This is the single
picture behind the headline: baking ≈ prompting on the converse, but loses to prompting on `cross`
(over-connecting unrelated, different-component pairs) — and the n=2 curriculum shrinks that gap.

Reuses the torch-free loaders from analysis.neg_family_auroc (never the training stack).

    python -m analysis.plot_propagation_aggregate \
        'results/bake_theorem_qa/graph-lw_*-n1' 'results/bake_theorem_qa/graph-lw_*-n2' \
        --out results/bake_theorem_qa/_fig_propagation_aggregate.png
"""

from __future__ import annotations

import argparse
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.load import load_run
from analysis.neg_family_auroc import _agg, _post_idx, _postmean

FAMILIES = [("ALL", ""), ("converse", "_converse"), ("cross", "_cross"),
            ("missing_edge", "_missing_edge")]


def _series(run_dirs, since):
    runs = []
    for d in run_dirs:
        run = load_run(d)
        idx, _ = _post_idx(run, since)
        runs.append((run, idx))
    return runs


def _cell(runs, state, depth, suffix):
    return _agg([_postmean(r, f"dprime.auroc_{state}_d{depth}{suffix}", ix) for r, ix in runs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n1_glob", help="glob for the baked n=1 runs")
    ap.add_argument("n2_glob", help="glob for the baked n=2 runs")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--since-epoch", type=int, default=100)
    ap.add_argument("--out", default="results/bake_theorem_qa/_fig_propagation_aggregate.png")
    a = ap.parse_args()

    n1 = _series(sorted(glob.glob(a.n1_glob)), a.since_epoch)
    n2 = _series(sorted(glob.glob(a.n2_glob)), a.since_epoch)
    ng = len(n1)

    # prompted is n-independent (same teacher per graph) -> read it from the n=1 set
    bar_specs = [
        ("prompted", n1, "#777777"),
        ("baked n=1", n1, "#4C72B0"),
        ("baked n=2", n2, "#C44E52"),
    ]

    fig, axes = plt.subplots(1, len(a.depths), figsize=(5.4 * len(a.depths), 5.4), sharey=True)
    if len(a.depths) == 1:
        axes = [axes]
    x = range(len(FAMILIES))
    w = 0.26

    for ax, d in zip(axes, a.depths):
        for bi, (label, runs, color) in enumerate(bar_specs):
            state = "prompted" if label == "prompted" else "baked"
            means, sds = [], []
            for _, suf in FAMILIES:
                c = _cell(runs, state, d, suf)
                means.append(c[0] if c else 0.0)
                sds.append(c[1] if c else 0.0)
            offs = [xi + (bi - 1) * w for xi in x]
            ax.bar(offs, means, w, yerr=sds, capsize=2.5, label=label, color=color,
                   edgecolor="white", linewidth=0.5, error_kw=dict(lw=1.0, alpha=0.7))
        ax.axhline(0.5, ls="--", lw=1.0, color="#444", alpha=0.7)
        ax.text(len(FAMILIES) - 0.5, 0.505, "chance", fontsize=7.5, color="#444", va="bottom", ha="right")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f for f, _ in FAMILIES], rotation=20, ha="right", fontsize=9)
        ax.set_title(f"proof depth {d}", fontsize=12)
        ax.set_ylim(0.3, 1.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("AUROC(true@depth  vs  negative)\nhigher = better discrimination")
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.95)

    fig.suptitle(f"Prompting vs baking — knowledge propagation by negative family "
                 f"(mean ± sd over {ng} graphs · Llama-3.1-8B · lw_* directed worlds)\n"
                 f"baking ≈ prompting on converse; baking's one graph-general deficit is CROSS "
                 f"(over-connecting unrelated pairs); n=2 curriculum shrinks it",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print("wrote", a.out, f"({ng} graphs per state)")


if __name__ == "__main__":
    main()
