"""Plot metric curves from a run's metrics.json to a PNG file (reports the path; never
embeds pixels into anyone's context). Reads only artifacts.

    python -m analysis.plot_curves results/bake_squad/<run_id> --metrics eval_kl train_kl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.load import load_run


def plot(run_dir, metrics=("eval_kl", "train_kl"), out=None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = load_run(run_dir)
    x = run.metrics.get("epochs")
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in metrics:
        y = run.metrics.get(m)
        if not y:
            continue
        xs = x[: len(y)] if x else list(range(1, len(y) + 1))
        ax.plot(xs, y, marker="o", label=m)
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title(run.manifest.get("run_id", ""))
    ax.legend()
    out_path = Path(out) if out else Path(run_dir) / "curves.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--metrics", nargs="*", default=["eval_kl", "train_kl"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    print(plot(a.run_dir, a.metrics, a.out))


if __name__ == "__main__":
    main()
