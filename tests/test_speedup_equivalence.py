"""Equivalence tests for the opt-in baking speedups — proof of NO BEHAVIOR CHANGE.

Each speedup ships with a test that runs the optimized path against the baseline and asserts the
outputs match. On the deterministic CPU fakes here the match is BIT-EXACT (`torch.equal` / atol=0);
the contract for GPU bf16 (where batched-GEMM/kernel reduction order differs) is tight-tolerance +
ZERO decision-flips, exercised end-to-end by the slow smoke equivalence test.

This file covers Optimization C (batched probe scoring). The fake model is mask-aware and computes
each real position's logits ONLY from that row's real-token prefix (via attention_mask), so it is
invariant to LEFT padding and batch composition — exactly the property that makes batched scoring
equal to the unbatched per-probe loop. It also accepts `position_ids` (which the batched path passes).
"""

import json
import math
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from bakery.eval.metrics.propagation import (
    _belief_per_probe, _paired_belief_batched, propagation,
)
from bakery.eval.metrics.dprime import _yesness_per_probe, dprime

VOCAB = 48
# Token ids the fake tokenizer emits. Leading-space vs no-space answers tokenize DIFFERENTLY (like a
# real BPE tokenizer) so `_variant_id_lists` yields multiple variants → exercises the logsumexp join.
SP_YES, NS_YES = 11, 12        # " Yes" vs "Yes"
SP_NO, NS_NO = 21, 22          # " No"  vs "No"
MAYBE = (31, 32)               # a MULTI-token answer (" Maybe so") → exercises n_ans>1 gather
MARK = 7                       # fact marker, present only when u is the system text


class EqFakeTok:
    chat_template = None        # force build_prefix_ids' manual fallback
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        # Answers: space and no-space variants get distinct ids; a multi-token answer too.
        table = {" Yes": [SP_YES], "Yes": [NS_YES], " No": [SP_NO], "No": [NS_NO],
                 " Maybe so": list(MAYBE), "Maybe so": list(MAYBE)}
        if text in table:
            return SimpleNamespace(input_ids=table[text])
        # A "prefix" text (system+question via the manual fallback): vary length by content so the
        # batch has MIXED prefix lengths (tests left-padding).
        ids = [1]
        if "MARK" in text:
            ids.append(MARK)
        ids.append(3)
        if "LONGER" in text:
            ids += [4, 5]       # a longer question → ragged batch
        return SimpleNamespace(input_ids=ids)


class EqFakeModel:
    """Causal, mask-aware, deterministic. logits[b,t] = sin(0.07 * run_b(t) * (arange(V)+1)) where
    run_b(t) = sum of REAL token ids (per attention_mask) at positions <= t in row b — plus a constant
    bump when the adapter is on. Depends ONLY on the row's real-token prefix ⇒ invariant to left padding
    and batch size ⇒ batched scoring is bit-identical to the unbatched single-row path."""

    def __init__(self, flag):
        self.flag = flag

    def __call__(self, input_ids, attention_mask=None, position_ids=None):
        B, L = input_ids.shape
        base = torch.arange(VOCAB, dtype=torch.float32) + 1.0
        bump = 100.0 if self.flag["adapter_on"] else 0.0
        logits = torch.zeros(B, L, VOCAB)
        for b in range(B):
            run = 0.0
            for t in range(L):
                m = 1 if attention_mask is None else int(attention_mask[b, t])
                if m:
                    run += float(input_ids[b, t].item())
                logits[b, t] = torch.sin(0.07 * (run + bump) * base)
        return SimpleNamespace(logits=logits)


class EqFakeBundle:
    def __init__(self):
        self.flag = {"adapter_on": False}
        self.tokenizer = EqFakeTok()
        self.peft_model = SimpleNamespace(eval=lambda: None)
        self._model = EqFakeModel(self.flag)

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


# Probes with mixed prefix lengths, both polarities, and a multi-token answer variant.
_PROBES = [
    {"hop": 1, "match_depth": 1, "proof_depth": 1, "form": "forward", "provable": True,
     "question": "MARK short?", "pos": " Yes", "neg": " No", "subj": "a", "obj": "b"},
    {"hop": 2, "match_depth": 2, "proof_depth": 2, "form": "forward", "provable": True,
     "question": "LONGER ragged question?", "pos": " Yes", "neg": " No", "subj": "c", "obj": "d"},
    {"hop": 1, "match_depth": 1, "proof_depth": None, "form": "converse", "provable": False,
     "neg_type": "converse", "question": "converse one?", "pos": " No", "neg": " Yes",
     "subj": "e", "obj": "f"},
    {"hop": 2, "match_depth": 2, "proof_depth": None, "form": "converse", "provable": False,
     "neg_type": "cross", "question": "LONGER converse two?", "pos": " Maybe so", "neg": " No",
     "subj": "g", "obj": "h"},
]


def _equal(a, b, tol=0.0):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(x - y) <= tol, f"{x} != {y} (|Δ|={abs(x-y)})"


def test_batched_belief_equals_unbatched_propagation():
    bundle = EqFakeBundle()
    for system in ("", "MARK"):
        for ctx_name in ("base", "baked"):
            ctx_mgr = getattr(bundle, ctx_name)
            unb = _belief_per_probe(bundle, _PROBES, system, ctx_mgr, "cpu")
            bat = _paired_belief_batched(bundle, _PROBES, system, ctx_mgr, "cpu",
                                         pos_of=lambda pr: pr["pos"], neg_of=lambda pr: pr["neg"])
            _equal(unb, bat, tol=0.0)                       # BIT-EXACT on the deterministic fake
            # zero decision flips (sign of belief identical)
            assert [b > 0 for b in unb] == [b > 0 for b in bat]


def test_batched_yesness_equals_unbatched_dprime():
    bundle = EqFakeBundle()
    for system in ("", "MARK"):
        for ctx_name in ("base", "baked"):
            ctx_mgr = getattr(bundle, ctx_name)
            unb = _yesness_per_probe(bundle, _PROBES, system, ctx_mgr, "cpu")
            bat = _yesness_per_probe(bundle, _PROBES, system, ctx_mgr, "cpu",
                                     batch_probes=True, probe_batch_size=0)
            _equal(unb, bat, tol=0.0)
            assert [y > 0.0 for y in unb] == [y > 0.0 for y in bat]


def test_probe_batch_size_chunking_is_invariant():
    """Chunk boundaries must not change per-row results (still bit-exact)."""
    bundle = EqFakeBundle()
    full = _paired_belief_batched(bundle, _PROBES, "MARK", bundle.base, "cpu",
                                  pos_of=lambda pr: pr["pos"], neg_of=lambda pr: pr["neg"], chunk=0)
    chunked = _paired_belief_batched(bundle, _PROBES, "MARK", bundle.base, "cpu",
                                     pos_of=lambda pr: pr["pos"], neg_of=lambda pr: pr["neg"], chunk=2)
    _equal(full, chunked, tol=0.0)


def _ctx(tmp_path, bundle, batch_probes):
    p = tmp_path / "probes.json"
    p.write_text(json.dumps({"probes": _PROBES}))
    data = SimpleNamespace(prompts={"base_u": "MARK", "baked": ""}, stats={})
    run_cfg = SimpleNamespace(
        data={"probe_bank": str(p)},
        eval=SimpleNamespace(batch_probes=batch_probes, probe_batch_size=0),
    )
    return SimpleNamespace(bundle=bundle, data=data, run_cfg=run_cfg, device="cpu")


def _flat(metric_result):
    return metric_result.to_metrics()


def test_propagation_metric_scalars_identical_batched_vs_unbatched(tmp_path):
    off = _flat(propagation(_ctx(tmp_path, EqFakeBundle(), batch_probes=False)))
    on = _flat(propagation(_ctx(tmp_path, EqFakeBundle(), batch_probes=True)))
    assert set(off) == set(on)
    for k in off:
        if k.endswith(".per_probe") or k.endswith(".matrix") or k.endswith(".labels"):
            assert off[k] == on[k] or _nested_close(off[k], on[k]), k
        else:
            v0, v1 = off[k], on[k]
            if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                assert v0 == v1 or (math.isnan(v0) and math.isnan(v1)), f"{k}: {v0} != {v1}"
            else:
                assert v0 == v1, k


def test_dprime_metric_scalars_identical_batched_vs_unbatched(tmp_path):
    off = _flat(dprime(_ctx(tmp_path, EqFakeBundle(), batch_probes=False)))
    on = _flat(dprime(_ctx(tmp_path, EqFakeBundle(), batch_probes=True)))
    assert set(off) == set(on)
    for k in off:
        v0, v1 = off[k], on[k]
        if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
            assert v0 == v1 or (math.isnan(v0) and math.isnan(v1)), f"{k}: {v0} != {v1}"
        else:
            assert v0 == v1, k


def _nested_close(a, b):
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(_nested_close(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


# ======================================================================================
# Optimization A — teacher-logit cache. The teacher is a pure function of fixed inputs, so caching
# its log-probs is exact memoization: cached == recompute → identical KL → identical training. On the
# deterministic CPU tiny-Llama these tests assert BIT-EXACT equality.
# ======================================================================================

import pytest  # noqa: E402

from bakery.objectives.base import (  # noqa: E402
    TeacherLogitCache, collate_framings, supervised_kl_terms,
)
from bakery.trajectories.encoding import FramedTrajectory  # noqa: E402

KL_VOCAB = 64


def _tiny_kl_bundle():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    from bakery.models.peft_factory import ModelBundle
    from bakery.trajectories.base import CheckpointId, TokenizerFingerprint

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=KL_VOCAB, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                      max_position_embeddings=128)
    model = LlamaForCausalLM(cfg)
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=4, lora_alpha=8,
                      lora_dropout=0.0, bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    # Perturb the adapter so teacher (disabled) and student (enabled) genuinely differ.
    pm = get_peft_model(model, lora)
    for name, p in pm.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.3)
    pm.eval()
    return ModelBundle(peft_model=pm, tokenizer=None, device="cpu",
                       base_checkpoint_id=CheckpointId(model_name="stub"),
                       tokenizer_fingerprint=TokenizerFingerprint(
                           name="stub", revision=None, vocab_size=KL_VOCAB, bos_id=1, eos_id=2, pad_id=0))


def _kl_batch():
    trajs = [
        FramedTrajectory(base_input_ids=(5, 6, 10, 11, 12), base_sup_mask=(False, False, True, True, True),
                         baked_input_ids=(5, 6, 10, 11, 12), baked_sup_mask=(False, False, True, True, True),
                         x0_id=0, num_supervised=3),
        FramedTrajectory(base_input_ids=(7, 8, 9, 13), base_sup_mask=(False, False, True, True),
                         baked_input_ids=(7, 8, 9, 13), baked_sup_mask=(False, False, True, True),
                         x0_id=1, num_supervised=2),
    ]
    return collate_framings(trajs, pad_id=0, traj_keys=(0, 1))


def test_teacher_cache_is_bit_exact():
    bundle = _tiny_kl_bundle()
    with torch.no_grad():
        baseline = supervised_kl_terms(bundle, _kl_batch(), device="cpu")
        cache = TeacherLogitCache(backend="cpu")
        miss = supervised_kl_terms(bundle, _kl_batch(), device="cpu", teacher_cache=cache)   # populate
        hit = supervised_kl_terms(bundle, _kl_batch(), device="cpu", teacher_cache=cache)     # reuse
    assert cache.has(0) and cache.has(1)
    assert torch.equal(baseline, miss)      # storing didn't change the value
    assert torch.equal(baseline, hit)       # REUSING the cached teacher is bit-exact


def test_teacher_cache_verify_hard_fails_on_drift():
    bundle = _tiny_kl_bundle()
    cache = TeacherLogitCache(backend="cpu", verify_every=1)
    with torch.no_grad():
        supervised_kl_terms(bundle, _kl_batch(), device="cpu", teacher_cache=cache)   # populate
        cache._store[0] = torch.zeros_like(cache._store[0])                            # corrupt a block
        with pytest.raises(RuntimeError, match="VERIFY FAILED"):
            supervised_kl_terms(bundle, _kl_batch(), device="cpu", teacher_cache=cache)


def test_teacher_cache_rejects_non_full_vocab_block():
    cache = TeacherLogitCache(backend="cpu")
    with pytest.raises(RuntimeError, match="(?i)full vocab"):
        cache.put(0, torch.zeros(3, KL_VOCAB - 1), vocab_size=KL_VOCAB)     # top-k would land here


def test_teacher_cache_dtype_and_backend_guards():
    with pytest.raises(ValueError):
        TeacherLogitCache(backend="cpu", dtype="bfloat16")
    with pytest.raises(NotImplementedError):
        TeacherLogitCache(backend="memmap")


@pytest.mark.xfail(strict=True, reason=(
    "FINDING (2026-06-08): the teacher-logit cache is NOT bit-exact to the cache-off baseline. "
    "The baseline recomputes the teacher in a PADDED batch each epoch, and the right-padding (hence "
    "the float-level result) changes with the epoch's shuffle/batchmates — so the baseline's own "
    "teacher targets vary per epoch at float scale. A fixed cached value therefore cannot equal the "
    "per-epoch recompute (verify-mode confirms cached != recompute). On the near-degenerate stub "
    "(eval_kl ~2e-5) this float-noise amplifies into large adapter divergence. The cache is a "
    "NUMERICALLY-EQUIVALENT (not bit-exact) optimization; pending a decision it is opt-in/default-off. "
    "If a future change makes the teacher computation padding-invariant, this xpass flags it for update."))
def test_bake_smoke_teacher_cache_bitexact_end_to_end(tmp_path):
    """Documents the non-bit-exactness finding (xfail). The strongest proof we CAN make is the
    fixed-batch bit-exactness above + default-off byte-identity (the existing 134-test suite)."""
    import json
    from bakery.config import load_config
    from bakery.runner import run

    def _bake(cache, name):
        ov = {"output_root": str(tmp_path), "run_name": name, "seed": "0",
              "train.cache_teacher_logits": cache}
        return run(load_config("bake_smoke", cli_overrides=ov))

    d_off, d_on = _bake("off", "tc_off"), _bake("cpu", "tc_on")
    from safetensors.torch import load_file
    a_off = load_file(str(d_off / "checkpoints" / "final" / "adapter_model.safetensors"))
    a_on = load_file(str(d_on / "checkpoints" / "final" / "adapter_model.safetensors"))
    for k in a_off:
        assert torch.equal(a_off[k], a_on[k]), f"adapter weight {k} differs cache-off vs cache-on"
