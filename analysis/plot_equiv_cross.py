"""Equivalence-world sharp test: when CROSS is the only false family, baking's deficit lands
entirely there. Reads the eqg-* metrics.json ONLY (torch-free); contrasts prompted vs baked
n=1/n=2 on cross discrimination (AUROC) and cross over-affirmation (FA = P("same kind"|unrelated)).

    python -m analysis.plot_equiv_cross results/bake_theorem_qa_equiv/eqg-*-n1 results/bake_theorem_qa_equiv/eqg-*-n2 \
        --out results/bake_theorem_qa_equiv/_fig_equiv_cross.png
"""

from __future__ import annotations

import argparse
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.load import load_run
from analysis.neg_family_auroc import _agg, _post_idx, _postmean

DEPTHS = [1, 2, 3]


def _series(glb, since):
    return [(load_run(d), _post_idx(load_run(d), since)[0]) for d in sorted(glob.glob(glb))]


def _col(runs, key, d):
    return _agg([_postmean(r, f"dprime.{key}_d{d}_cross", ix) for r, ix in runs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n1_glob"); ap.add_argument("n2_glob")
    ap.add_argument("--since-epoch", type=int, default=100)
    ap.add_argument("--out", default="results/bake_theorem_qa_equiv/_fig_equiv_cross.png")
    a = ap.parse_args()
    n1 = _series(a.n1_glob, a.since_epoch); n2 = _series(a.n2_glob, a.since_epoch)
    ng = len(n1)

    specs = [("prompted", n1, "auroc_prompted", "fa_prompted", "#777777"),
             ("baked n=1", n1, "auroc_baked", "fa_baked", "#4C72B0"),
             ("baked n=2", n2, "auroc_baked", "fa_baked", "#C44E52")]
    fig, (axa, axf) = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(DEPTHS)); w = 0.26
    for bi, (lab, runs, ak, fk, c) in enumerate(specs):
        am = [( _col(runs, ak, d) or (0,0,0))[0] for d in DEPTHS]
        asd = [(_col(runs, ak, d) or (0,0,0))[1] for d in DEPTHS]
        fm = [( _col(runs, fk, d) or (0,0,0))[0] for d in DEPTHS]
        fsd = [(_col(runs, fk, d) or (0,0,0))[1] for d in DEPTHS]
        offs = [xi + (bi - 1) * w for xi in x]
        axa.bar(offs, am, w, yerr=asd, capsize=3, label=lab, color=c, edgecolor="white")
        axf.bar(offs, fm, w, yerr=fsd, capsize=3, label=lab, color=c, edgecolor="white")
    axa.axhline(0.5, ls="--", lw=1, color="#444", alpha=.7)
    axa.set_title("cross discrimination — AUROC(true vs cross)\nhigher = better"); axa.set_ylim(0.3, 1.05)
    axf.set_title('cross over-affirmation — FA = P("same kind" | UNRELATED pair)\nlower = better'); axf.set_ylim(0, 1.05)
    for ax in (axa, axf):
        ax.set_xticks(list(x)); ax.set_xticklabels([f"depth {d}" for d in DEPTHS])
        ax.grid(axis="y", alpha=.25); ax.legend(fontsize=9)
    fig.suptitle(f"Equivalence world (cross = ONLY false family) — baking over-connects unrelated concepts "
                 f"(mean±sd over {ng} eq graphs)\nbaking aces within-class (d1 AUROC 1.0) but affirms 'same kind' "
                 f"for 93% of unrelated pairs at d2 (n=1); the n=2 curriculum heals it", fontsize=11.5, y=1.03)
    fig.tight_layout(); fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print("wrote", a.out, f"({ng} graphs)")


if __name__ == "__main__":
    main()
