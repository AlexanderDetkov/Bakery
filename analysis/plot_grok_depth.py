"""Grokking check on the proof-system instrument: does deep-depth discrimination rise LATE,
after eval_kl plateaus? Reads the long grok-* bakes' metrics.json ONLY (torch-free).

Left: eval_kl vs epoch (log y). Right: baked AUROC vs epoch at the propagation depths d2 and d3.
Grokking would be a late RISE in d2/d3 AUROC long after eval_kl flattens. We see the opposite
(flat-to-declining), and the wd=0 n=2 arm diverges — i.e. no compositional transition.

    python -m analysis.plot_grok_depth results/bake_theorem_qa/grok-* --out results/bake_theorem_qa/_fig_grok_depth.png
"""

from __future__ import annotations

import argparse
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.load import load_run

COLORS = {"grok-n1-wd0.0": "#4C72B0", "grok-n1-wd0.1": "#8FB0DD",
          "grok-n2-wd0.0": "#C44E52", "grok-n2-wd0.1": "#E89AA0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/bake_theorem_qa/_fig_grok_depth.png")
    a = ap.parse_args()

    dirs = []
    for r in a.runs:
        dirs.extend(sorted(glob.glob(r)) if any(c in r for c in "*?[") else [r])

    fig, (axk, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))
    for d in dirs:
        run = load_run(d)
        name = run.run_dir.name
        eps = run.metrics.get("epochs") or []
        if not eps:
            continue
        c = COLORS.get(name, "#777")
        kl = run.metrics.get("eval_kl") or []
        d2 = run.metrics.get("dprime.auroc_baked_d2") or []
        d3 = run.metrics.get("dprime.auroc_baked_d3") or []
        n = min(len(eps), len(kl))
        axk.plot(eps[:n], kl[:n], color=c, label=name, lw=1.6)
        m2 = min(len(eps), len(d2)); ax2.plot(eps[:m2], d2[:m2], color=c, label=name, lw=1.6)
        m3 = min(len(eps), len(d3)); ax3.plot(eps[:m3], d3[:m3], color=c, label=name, lw=1.6)

    axk.set_yscale("log"); axk.set_title("eval_kl (distillation loss)"); axk.set_xlabel("epoch")
    axk.axhline(0.25, ls=":", color="#888", lw=1); axk.legend(fontsize=8)
    for ax, title in [(ax2, "baked AUROC @ depth 2 (1 hop out)"),
                      (ax3, "baked AUROC @ depth 3 (2 hops out)")]:
        ax.axhline(0.5, ls="--", color="#444", lw=1, alpha=0.7)
        ax.set_ylim(0.3, 1.0); ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel("AUROC")
        ax.text(ax.get_xlim()[1], 0.505, "chance", fontsize=7.5, color="#444", ha="right", va="bottom")
    fig.suptitle("Grokking check — no late compositional transition through ep2000 "
                 "(eval_kl flat from ~ep50; d2/d3 AUROC flat-to-declining; wd=0 n=2 diverges)",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
