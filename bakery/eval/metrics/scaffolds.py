"""Scaffold metrics — discoverable extension seams, NOT yet implemented.

Each is registered (so `python run.py --list` / the metric registry surfaces it) but raises
NotImplementedError when invoked, with a pointer. This way a run that names an unfinished metric
fails loudly rather than logging fake numbers. To implement one: replace its body with a real
`MetricResult` (ideally in its own bakery/eval/metrics/<name>.py) — see the porting notes below.
"""

from __future__ import annotations

from bakery.eval.registry import register_metric

# name -> what a real implementation should compute (ported from the reference repos / paper).
_SCAFFOLDS = {
    "em_f1": "SQuAD exact-match + token-F1 (with bootstrap CI) on held-out questions (instruct_eval.py).",
    "cot_strict_anywhere": "GSM8k/SVAMP strict-match + anywhere-match accuracy (cot_eval.py).",
    "logit_r2": "r^2 between baked(no-prompt) and base(with-prompt) per-token log-probs; reuse the "
                "supervised alignment in objectives/base.py (r2_compare.py / compare_models.py).",
    "persona_stability": "persona adherence over a multi-turn dialogue (Figure 7).",
    "cross_benchmark_matrix": "trained-prompt x eval-benchmark generalization matrix (Figure 6); "
                              "return MetricResult(matrix=..., labels=...).",
    "four_way_loglik": "−log-likelihood under {base,baked} x {prompt,no-prompt}; also covers "
                       "re-prompting and a half-bake alpha sweep via bundle.half_baked(alpha).",
}


def _make(name, note):
    def _metric(ctx):
        raise NotImplementedError(
            f"Metric {name!r} is a scaffold — implement it in bakery/eval/metrics/. Should compute: {note}"
        )
    return _metric


for _name, _note in _SCAFFOLDS.items():
    register_metric(_name)(_make(_name, _note))
