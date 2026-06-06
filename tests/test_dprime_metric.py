"""Logic tests for the `dprime` signal-detection metric (offline, no model download).

The headline property: a PURE YES-BIAS responder (says Yes to everything) scores forward "accuracy"
1.0 yet d′ ≈ 0 at every depth — because hit rate ≈ false-alarm rate. A perfect discriminator yields
large d′ and AUROC 1.0. The log-linear correction keeps d′ finite at degenerate (all-hit / zero-FA)
cells, and a depth with no negatives is reported NaN (never a silently-wrong number).
"""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from bakery.eval.metrics.dprime import _phi_inv, dprime

YES, NO, MARK = 11, 22, 7
VOCAB = 32


class FakeTok:
    chat_template = None                         # forces build_prefix_ids' manual fallback

    def __call__(self, text, add_special_tokens=True):
        s = text.strip()
        if s == "Yes":
            return SimpleNamespace(input_ids=[YES])
        if s == "No":
            return SimpleNamespace(input_ids=[NO])
        ids = [1]
        if "AFFIRM" in text:                     # the per-probe provability marker
            ids.append(MARK)
        ids.append(3)
        return SimpleNamespace(input_ids=ids)


class _Model:
    """`yes_if(prefix)` decides whether mass goes on YES (else NO) at the predicting position."""
    def __init__(self, yes_if):
        self.yes_if = yes_if

    def __call__(self, input_ids, attention_mask=None):
        L = input_ids.shape[1]
        logits = torch.zeros(1, L, VOCAB)
        prefix = input_ids[0].tolist()[:-1]
        logits[0, L - 2, YES if self.yes_if(prefix) else NO] = 5.0
        return SimpleNamespace(logits=logits)


class FakeBundle:
    def __init__(self, model):
        self.tokenizer = FakeTok()
        self.peft_model = SimpleNamespace(eval=lambda: None)
        self._model = model

    @contextmanager
    def base(self):
        yield self._model

    @contextmanager
    def baked(self):
        yield self._model


def _probe(d, provable, form, neg_type):
    return {"proof_depth": d, "hop": d, "provable": provable, "form": form, "neg_type": neg_type,
            "entities": ["X", "Y"], "question": f"q d{d} {'AFFIRM' if provable else 'deny'}",
            "pos": " Yes" if provable else " No", "neg": " No" if provable else " Yes"}


def _bank(tmp_path):
    probes = []
    for d in (1, 2, 3):
        probes += [_probe(d, True, "forward", None), _probe(d, True, "forward", None),
                   _probe(d, False, "converse", "converse"), _probe(d, False, "cross", "cross")]
    probes += [_probe(4, True, "forward", None), _probe(4, True, "forward", None)]   # imbalanced cell
    p = tmp_path / "probes.json"
    p.write_text(json.dumps({"probes": probes}))
    return p


def _ctx(tmp_path, model):
    data = SimpleNamespace(prompts={"base_u": "the fact", "baked": ""})
    run_cfg = SimpleNamespace(data={"probe_bank": str(_bank(tmp_path))})
    return SimpleNamespace(bundle=FakeBundle(model), data=data, run_cfg=run_cfg, device="cpu")


def test_pure_yes_bias_has_dprime_near_zero(tmp_path):
    # always YES regardless of probe -> H == FA at every depth -> d' == 0 (yes-saturation exposed)
    e = dprime(_ctx(tmp_path, _Model(lambda prefix: True))).extra
    for d in (1, 2, 3):
        assert abs(e[f"dprime_baked_d{d}"]) < 1e-9
        assert abs(e[f"hit_baked_d{d}"] - e[f"fa_baked_d{d}"]) < 1e-9
        assert abs(e[f"auroc_baked_d{d}"] - 0.5) < 1e-9        # all-equal beliefs -> ties
    assert e["prop_distance_dprime_baked"] == 0


def test_perfect_discriminator_has_high_dprime(tmp_path):
    # YES iff the provability marker is present -> provable->Yes, non-provable->No
    e = dprime(_ctx(tmp_path, _Model(lambda prefix: MARK in prefix))).extra
    for d in (1, 2, 3):
        assert e[f"dprime_baked_d{d}"] > 1.5
        assert e[f"auroc_baked_d{d}"] == 1.0
        import math
        assert math.isfinite(e[f"dprime_baked_d{d}"])          # log-linear correction -> finite, not +inf
    assert e["prop_distance_dprime_baked"] == 3                # d1..d3 above threshold; d4 has no negatives


def test_missing_negative_cell_is_nan_with_reason(tmp_path):
    import math
    e = dprime(_ctx(tmp_path, _Model(lambda prefix: MARK in prefix))).extra
    assert math.isnan(e["dprime_baked_d4"])
    assert e["dprime_skipped"]["d4_baked"] == "no_negatives"


def test_negtype_stratification_present(tmp_path):
    e = dprime(_ctx(tmp_path, _Model(lambda prefix: MARK in prefix))).extra
    assert "dprime_baked_d1_converse" in e and "dprime_baked_d1_cross" in e
    assert e["auroc_baked_d1_converse"] == 1.0                 # perfect model separates this negative too


def test_phi_inv_sanity():
    assert abs(_phi_inv(0.5)) < 1e-6
    assert abs(_phi_inv(0.8413447) - 1.0) < 1e-3
    assert abs(_phi_inv(0.1586553) + 1.0) < 1e-3
