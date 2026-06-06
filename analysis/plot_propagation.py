"""Plot per-hop belief-shift propagation curves from bake_fact runs (reads metrics.json ONLY).

The `propagation` metric stores, per epoch, parallel lists keyed `propagation.{state}_h{n}` and
`propagation.{state}_shift_h{n}` for state in {prior, prompted, baked}, plus `propagation.hops`.
This module reads the FINAL epoch and plots belief-shift-vs-hop curves to a PNG (reports the path;
never embeds pixels into context). Imports only matplotlib + analysis.load — never the training stack.

    python -m analysis.plot_propagation prompt_vs_bake --out results/bake_fact/_fig_pvb.png \
        results/bake_fact/prop-tsunami-1b-r16-mixed:1B results/bake_fact/prop-tsunami-8b-r16-mixed:8B
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.load import load_run


def hops_of(run) -> list:
    h = run.metrics.get("propagation.hops")
    if h:
        return list(h[-1])
    keys = [k for k in run.metrics if k.startswith("propagation.prior_h")]
    return sorted(int(k.rsplit("_h", 1)[-1]) for k in keys)


def series(run, state, kind="shift") -> list:
    """Final-epoch per-hop values; kind in {'shift','belief'}."""
    out = []
    for h in hops_of(run):
        key = (f"propagation.{state}_shift_h{h}" if kind == "shift"
               else f"propagation.{state}_h{h}")
        v = run.metrics.get(key)
        out.append(v[-1] if v else None)
    return out


def _plot(curves, out, title, ylabel):
    """curves: list of dict(label, hops, y, style, color)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in curves:
        ax.plot(c["hops"], c["y"], marker=c.get("marker", "o"),
                linestyle=c.get("style", "-"), color=c.get("color"), label=c["label"])
    ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    ax.set_xlabel("reasoning hop distance n")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


def prompt_vs_bake(run_specs, out):
    """run_specs: list of 'run_dir:label'. Plots prompted (dashed) vs baked (solid) shift per model."""
    palette = ["C0", "C1", "C2", "C3"]
    curves = []
    for i, spec in enumerate(run_specs):
        run_dir, _, label = spec.partition(":")
        r = load_run(run_dir)
        hops = hops_of(r)
        curves.append({"label": f"{label} prompted", "hops": hops, "y": series(r, "prompted"),
                       "style": "--", "color": palette[i % len(palette)], "marker": "s"})
        curves.append({"label": f"{label} baked", "hops": hops, "y": series(r, "baked"),
                       "style": "-", "color": palette[i % len(palette)], "marker": "o"})
    return _plot(curves, out, "Knowledge propagation: prompted vs baked",
                 "belief shift vs prior  (logP(pos)-logP(neg))")


def by_type(run_specs, out, reference=None):
    """run_specs: 'run_dir:label' baked curves; optional reference 'run_dir:label' for prompted."""
    curves = []
    if reference:
        run_dir, _, label = reference.partition(":")
        r = load_run(run_dir)
        curves.append({"label": label, "hops": hops_of(r), "y": series(r, "prompted"),
                       "style": "--", "color": "0.4", "marker": "s"})
    for i, spec in enumerate(run_specs):
        run_dir, _, label = spec.partition(":")
        r = load_run(run_dir)
        curves.append({"label": label, "hops": hops_of(r), "y": series(r, "baked"),
                       "style": "-", "color": f"C{i}", "marker": "o"})
    return _plot(curves, out, "Baked-fact propagation by trajectory type",
                 "baked belief shift vs prior")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["prompt_vs_bake", "by_type"])
    ap.add_argument("run_specs", nargs="+", help="run_dir:label")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None, help="run_dir:label for the prompted reference (by_type)")
    a = ap.parse_args(argv)
    if a.mode == "prompt_vs_bake":
        print(prompt_vs_bake(a.run_specs, a.out))
    else:
        print(by_type(a.run_specs, a.out, reference=a.reference))


if __name__ == "__main__":
    main()
