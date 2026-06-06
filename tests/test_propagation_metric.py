"""Logic tests for the `propagation` metric (fully offline, no model download).

We drive the metric with a deterministic fake model + fake tokenizer so the belief math and the
three-state wiring (prior=base+empty, prompted=base+u, baked=adapter+empty) are pinned exactly:
  * `_seq_logprob` returns the summed log-prob of the answer tokens (verified against a hand
    computation), using the same logit->token shift as the KL primitive.
  * the metric reports a POSITIVE prompted_shift and baked_shift when the (fake) model is built so
    that the presence of the fact OR the adapter pushes mass onto the fact-consistent answer.
All test probes use pos=" Yes" so the fake's single rule aligns with `pos` (polarity-balancing is
a property of the real probe bank, not the metric).
"""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from bakery.eval.metrics.propagation import _seq_logprob, propagation

YES, NO, MARK = 11, 22, 7          # token ids the fake tokenizer emits
VOCAB = 32


class FakeTok:
    chat_template = None            # forces build_prefix_ids' manual fallback
    def __call__(self, text, add_special_tokens=True):
        s = text.strip()
        if s == "Yes":
            return SimpleNamespace(input_ids=[YES])
        if s == "No":
            return SimpleNamespace(input_ids=[NO])
        ids = [1]                   # BOS-ish
        if "TSUNAMI" in text:       # the fact marker (only present when u is the system text)
            ids.append(MARK)
        ids.append(3)               # the question content
        return SimpleNamespace(input_ids=ids)


class FakeModel:
    """Puts mass on YES iff the fact marker is in the prefix OR the adapter is on; else on NO."""
    def __init__(self, flag):
        self.flag = flag
    def __call__(self, input_ids, attention_mask=None):
        L = input_ids.shape[1]
        logits = torch.zeros(1, L, VOCAB)
        prefix = input_ids[0].tolist()[:-1]
        boost_yes = (MARK in prefix) or self.flag["adapter_on"]
        logits[0, L - 2, YES if boost_yes else NO] = 5.0     # logit at L-2 predicts the answer at L-1
        return SimpleNamespace(logits=logits)


class FakeBundle:
    def __init__(self):
        self.flag = {"adapter_on": False}
        self.tokenizer = FakeTok()
        self.peft_model = SimpleNamespace(eval=lambda: None)
        self._model = FakeModel(self.flag)
    @contextmanager
    def base(self):
        self.flag["adapter_on"] = False
        yield self._model
    @contextmanager
    def baked(self):
        self.flag["adapter_on"] = True
        try:
            yield self._model
        finally:
            self.flag["adapter_on"] = False


def test_seq_logprob_matches_hand_computation():
    flag = {"adapter_on": True}
    model = FakeModel(flag)
    tok = FakeTok()
    prefix = [1, MARK, 3]
    got = _seq_logprob(model, tok, prefix, " Yes", device="cpu")
    # logits at the predicting position: YES=5.0, everything else 0 over VOCAB.
    logits = torch.zeros(VOCAB)
    logits[YES] = 5.0
    expected = float(torch.log_softmax(logits, dim=-1)[YES])
    assert abs(got - expected) < 1e-5


def _ctx(tmp_path, bundle):
    probes = [
        {"hop": 0, "question": "Is there a disaster?", "pos": " Yes", "neg": " No"},
        {"hop": 0, "question": "Is it dangerous?", "pos": " Yes", "neg": " No"},
        {"hop": 1, "question": "Should I worry?", "pos": " Yes", "neg": " No"},
    ]
    p = tmp_path / "probes.json"
    p.write_text(json.dumps({"probes": probes}))
    data = SimpleNamespace(prompts={"base_u": "A TSUNAMI struck.", "baked": ""})
    run_cfg = SimpleNamespace(data={"probe_bank": str(p)})
    return SimpleNamespace(bundle=bundle, data=data, run_cfg=run_cfg, device="cpu")


def test_prompting_and_baking_both_shift_belief_positive(tmp_path):
    res = propagation(_ctx(tmp_path, FakeBundle()))
    assert res.labels == ["prior", "prompted", "baked"]
    assert res.extra["hops"] == [0, 1]
    # prompting (u in context) and baking (adapter on) both move belief toward the fact-consistent
    # answer relative to the prior (no fact, no adapter).
    assert res.extra["prompted_shift_h0"] > 0.5
    assert res.extra["baked_shift_h0"] > 0.5
    assert res.extra["prompted_shift_h1"] > 0.5
    # in this fake, baking exactly reproduces prompting -> fidelity ~ 0.
    assert abs(res.extra["baked_fidelity_h0"]) < 1e-5
    assert res.value > 0.5                       # headline = mean baked shift across hops
    assert len(res.matrix) == 3 and len(res.matrix[0]) == 2
