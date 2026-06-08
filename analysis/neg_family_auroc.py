"""Per-negative-family AUROC re-read of bake_theorem_qa runs (metrics.json ONLY).

The `dprime` metric already logs, per eval step, `dprime.auroc_<state>_d<d>_<negtype>` (state in
prior|prompted|baked; negtype in converse|cross|missing_edge) plus the aggregate
`dprime.auroc_<state>_d<d>` and the per-family false-alarm rate `dprime.fa_<state>_d<d>_<negtype>`.

This script re-stratifies those logged scalars WITHOUT re-scoring any adapter — it answers
"does the bias-immune instrument reproduce the pre-Hilbert 'baking affirms the converse /
loses direction' result?" by reading:
  * AUROC(true@d  vs  negfamily@d)  — discrimination (0.5 = chance; <0.5 = ranks the negative ABOVE truth)
  * FA = P(yes | negfamily@d)       — absolute affirmation rate of that negative (yes-saturation ⇒ →1.0)
for prompted vs baked, grouped by the curriculum depth n (config.data.train_max_depth).

Imports only the stdlib + analysis.load (never the training stack — see test_analysis_isolation).

    python -m analysis.neg_family_auroc results/bake_theorem_qa/qa-bake-n*-s1* --since-epoch 100
"""

from __future__ import annotations

import argparse
import glob
import statistics
from collections import defaultdict

from analysis.load import load_run

STATES = ("prior", "prompted", "baked")
NEG = ("converse", "cross", "missing_edge")


def _postmean(run, key, post_idx):
    series = run.metrics.get(key)
    if not isinstance(series, list):
        return None
    vals = [series[i] for i in post_idx
            if i < len(series) and isinstance(series[i], (int, float))]
    vals = [v for v in vals if v == v]              # drop NaN
    return statistics.fmean(vals) if vals else None


def _post_idx(run, since):
    epochs = run.metrics.get("epochs") or []
    idx = [i for i, e in enumerate(epochs) if isinstance(e, (int, float)) and e >= since]
    if len(idx) < 2:                                # threshold too high for this run -> last 2 evals
        idx = list(range(len(epochs)))[-2:]
    return idx, [epochs[i] for i in idx]


def _agg(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = statistics.fmean(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, sd, len(xs)


def _fmt(a):
    return f"{a[0]:.2f}±{a[1]:.2f}(n{a[2]})" if a else "   —   "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs (globs ok)")
    ap.add_argument("--since-epoch", type=int, default=100, help="post-plateau cutoff (default 100)")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    dirs = []
    for r in args.runs:
        dirs.extend(sorted(glob.glob(r)) if any(c in r for c in "*?[") else [r])

    # group runs by curriculum depth n = config.data.train_max_depth
    groups: dict = defaultdict(list)
    for d in dirs:
        run = load_run(d)
        n = (((run.config or {}).get("data") or {}).get("train_max_depth"))
        idx, eps = _post_idx(run, args.since_epoch)
        groups[n].append((run, idx, eps))
        print(f"  loaded {run.run_dir.name}: n={n} status={run.status} "
              f"evals@{eps} eval_kl_final={run.final('eval_kl')}")

    for n in sorted(groups, key=lambda x: (x is None, x)):
        runs = groups[n]
        print(f"\n{'='*78}\n  n = {n}   ({len(runs)} runs: "
              f"{', '.join(r.run_dir.name for r, _, _ in runs)})\n{'='*78}")
        # AUROC table (true vs each negative family) + the aggregate-all column
        for d in args.depths:
            print(f"\n  depth {d}  —  AUROC(true@{d} vs negative)   [0.5=chance, <0.5=ranks neg above truth]")
            print(f"    {'state':<9} {'ALL':>13} {'converse':>13} {'cross':>13} {'missing_edge':>13}")
            for st in STATES:
                allc = _agg([_postmean(r, f"dprime.auroc_{st}_d{d}", ix) for r, ix, _ in runs])
                cols = {ng: _agg([_postmean(r, f"dprime.auroc_{st}_d{d}_{ng}", ix) for r, ix, _ in runs])
                        for ng in NEG}
                print(f"    {st:<9} {_fmt(allc):>13} {_fmt(cols['converse']):>13} "
                      f"{_fmt(cols['cross']):>13} {_fmt(cols['missing_edge']):>13}")
            print(f"\n  depth {d}  —  FA = P(answer 'Yes' | negative)   [yes-saturation ⇒ →1.0; affirms the negative]")
            print(f"    {'state':<9} {'converse':>13} {'cross':>13} {'missing_edge':>13}")
            for st in STATES:
                cols = {ng: _agg([_postmean(r, f"dprime.fa_{st}_d{d}_{ng}", ix) for r, ix, _ in runs])
                        for ng in NEG}
                print(f"    {st:<9} {_fmt(cols['converse']):>13} "
                      f"{_fmt(cols['cross']):>13} {_fmt(cols['missing_edge']):>13}")


if __name__ == "__main__":
    main()
