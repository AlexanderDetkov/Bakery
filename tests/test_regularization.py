"""Tests for baking regularization: anchor trajectories + the `behavior_drift` metric. Network-free.

An ANCHOR is a `FramedTrajectory` whose two framings are IDENTICAL (both no-prompt) and whose
supervised span is a continuation the BASE model generated. Mixing anchors into training adds a
`KL(base_no_prompt ‖ baked_no_prompt)` restoring force pulling the adapter toward IDENTITY on
irrelevant inputs. These tests pin: the anchor framing + that the gate accepts it (mask alignment
holds when base==baked); the reserved `x0_id` band; disjoint train/held-out pool slices; the strength
knob growing the theorem_qa train set WITHOUT disturbing depth-1 coverage or leaking; `KL≈0` at adapter
identity; and the `behavior_drift` metric (None when OFF, ≈0 at identity).

If a test here fails you broke an invariant — fix the code, not the test.
"""

import contextlib
from types import SimpleNamespace

from bakery.objectives.base import collate_framings, supervised_kl_terms
from bakery.trajectories.encoding import iter_supervised_ids
from bakery.trajectories.regularization import (
    anchor_windows,
    append_train_anchors,
    build_anchor_trajectories,
    load_anchor_pool,
)
from tests import _invariant_kit as kit


# --- offline fakes ------------------------------------------------------------------------------

class WordTok:
    """Word-level tokenizer: forces build_prefix_ids' manual fallback (chat_template=None) and
    round-trips decode so the contamination validator runs without a model."""
    chat_template = None

    def __init__(self):
        self._v = {"<pad>": 0}
        self._inv = {0: "<pad>"}

    def _id(self, w):
        if w not in self._v:
            i = len(self._v)
            self._v[w] = i
            self._inv[i] = w
        return self._v[w]

    def __call__(self, text, add_special_tokens=True):
        for ch in ".,?":
            text = text.replace(ch, f" {ch} ")
        return SimpleNamespace(input_ids=[self._id(w) for w in text.split()])

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self._inv.get(i, "") for i in ids if not (skip_special_tokens and i == 0))


class FakeGen:
    """Injected generator: one fixed continuation per prefix (ignores gen_cfg). Model/network-free."""
    backend_name = "fake"

    def __init__(self, y):
        self._y = list(y)

    def generate(self, prefixes, gen_cfg):
        return [list(self._y) for _ in prefixes]


def _stub_bundle(tok=None):
    """A bundle exposing only what anchor building needs offline: base() (nullcontext) + tokenizer."""
    return SimpleNamespace(base=lambda: contextlib.nullcontext(), tokenizer=tok or WordTok())


def _reg(num_train, *, eval_n=2, source="synthetic", split="train", seed=0,
         max_new_tokens=4, do_sample=False):
    return SimpleNamespace(num_train_contexts=num_train, eval_num_contexts=eval_n, source=source,
                           context_split=split, seed=seed, max_new_tokens=max_new_tokens,
                           do_sample=do_sample)


# --- anchor framing + gate ----------------------------------------------------------------------

def test_anchor_is_base_equals_baked_supervised_on_y():
    anchors = build_anchor_trajectories(
        bundle=_stub_bundle(), tokenizer=WordTok(), contexts=["the cat sat", "a dog ran"],
        generator=FakeGen([10, 11, 12]),
    )
    assert len(anchors) == 2
    for t in anchors:
        assert t.base_input_ids == t.baked_input_ids          # identical framings (NO prompt)
        assert t.base_sup_mask == t.baked_sup_mask
        base_sup, baked_sup = iter_supervised_ids(t)
        assert base_sup == baked_sup == [10, 11, 12]          # supervised span == the generated y
        assert t.num_supervised == 3
        assert t.base_sup_mask[-3:] == (True, True, True)     # mask is the trailing y only
        assert not any(t.base_sup_mask[:-3])


def test_anchors_pass_the_gate_alongside_main_trajectories():
    # The whole point of base==baked anchors: assert_mask_alignment accepts them, and they coexist
    # with ordinary (base!=baked) main trajectories in one gate-validated dataset.
    main = [kit.make_traj(0, (10, 11, 12)), kit.make_traj(1, (13, 14))]
    anchors = build_anchor_trajectories(
        bundle=_stub_bundle(), tokenizer=WordTok(), contexts=["x y", "z w"],
        generator=FakeGen([20, 21]),
    )
    ds = kit.gate(main + anchors, [kit.make_traj(2, (15, 16))])   # raises if any check fails
    assert ds.stats["n_train_traj"] == 4


def test_anchor_x0_ids_in_reserved_band():
    anchors = build_anchor_trajectories(
        bundle=_stub_bundle(), tokenizer=WordTok(), contexts=["q one", "q two", "q three"],
        generator=FakeGen([20, 21]), x0_start=20_000,
    )
    assert [t.x0_id for t in anchors] == [20_000, 20_001, 20_002]   # disjoint from main (0..) / eval (10000+)


def test_empty_continuations_are_dropped():
    anchors = build_anchor_trajectories(
        bundle=_stub_bundle(), tokenizer=WordTok(), contexts=["a b", "c d"],
        generator=FakeGen([]),                                 # all-stop -> nothing supervised
    )
    assert anchors == []


# --- deterministic pool + disjoint windows ------------------------------------------------------

def test_anchor_windows_disjoint_and_deduped():
    train_ctx, held = anchor_windows(_reg(6, eval_n=4))
    assert len(train_ctx) == 6 and len(held) == 4
    assert set(train_ctx).isdisjoint(held)                     # held-out window disjoint from trained
    assert len(set(train_ctx + held)) == 10                    # the pool is deduped


def test_load_anchor_pool_is_deterministic():
    a = load_anchor_pool(source="synthetic", split="train", count=8, seed=0)
    b = load_anchor_pool(source="synthetic", split="train", count=8, seed=0)
    assert a == b and len(a) == 8 and len(set(a)) == 8


# --- the strength knob, OFF semantics -----------------------------------------------------------

def test_append_train_anchors_off_is_noop():
    train = [kit.make_traj(0, (10, 11))]
    cfg = SimpleNamespace(regularization=_reg(0), generation=None)
    out = append_train_anchors(train, cfg=cfg, bundle=_stub_bundle(), tokenizer=WordTok())
    assert out is train                                        # OFF -> the exact same list, no anchors


# --- KL == 0 when the adapter is identity (the crisp correctness anchor) ------------------------

def test_kl_zero_at_adapter_identity():
    # PEFT zero-inits LoRA's B matrix => the adapter contributes nothing at init => base()==baked()
    # on the identical no-prompt sequence => KL == 0 for every anchor token.
    bundle = kit.tiny_peft_bundle(vocab=64, r=4)
    anchors = build_anchor_trajectories(
        bundle=bundle, tokenizer=WordTok(), contexts=["the cat sat on mat", "a dog ran fast"],
        generator=FakeGen([10, 11, 12]),
    )
    assert anchors and all(t.base_input_ids == t.baked_input_ids for t in anchors)
    batch = collate_framings(anchors, kit.DEFAULT_TOK.pad_id)
    terms = supervised_kl_terms(bundle, batch, base_with_adapter=False, device="cpu")
    assert float(terms.abs().max()) < 1e-4


# --- behavior_drift metric ----------------------------------------------------------------------

def test_behavior_drift_returns_none_when_off():
    from bakery.eval.metrics.behavior_drift import behavior_drift
    ctx = SimpleNamespace(run_cfg=SimpleNamespace(regularization=_reg(0)),
                          bundle=None, data=None, device="cpu")
    assert behavior_drift(ctx) is None                         # OFF -> dropped by the runner


def test_behavior_drift_zero_at_identity(monkeypatch):
    import bakery.trajectories.regularization as regmod
    from bakery.config import GenerationConfig, RegularizationConfig, TrainConfig
    from bakery.eval.metrics.behavior_drift import behavior_drift

    monkeypatch.setattr(regmod, "make_generator", lambda backend, bundle: FakeGen([10, 11, 12]))
    bundle = kit.tiny_peft_bundle(vocab=64, r=4)
    bundle.tokenizer = WordTok()
    run_cfg = SimpleNamespace(
        regularization=RegularizationConfig(num_train_contexts=2, eval_num_contexts=2,
                                            source="synthetic", context_split="train", max_new_tokens=4),
        generation=GenerationConfig(), train=TrainConfig(batch_size=4),
    )
    data = SimpleNamespace(tokenizer_fingerprint=kit.DEFAULT_TOK)
    ctx = SimpleNamespace(bundle=bundle, data=data, run_cfg=run_cfg, device="cpu")
    res = behavior_drift(ctx)
    assert res is not None and res.value is not None
    assert abs(res.value) < 1e-4                                # adapter identity => no drift
    assert res.extra["n_anchors"] == 2


# --- integration through the theorem_qa builder + gate ------------------------------------------

def _build_theorem_qa(tmp_path, monkeypatch, num_train_contexts):
    """Build a tiny theorem_qa dataset (offline world + QA bank) with regularization on/off, using a
    fake generator for the anchors so no model/network is touched."""
    import bakery.trajectories.regularization as regmod
    from bakery.config import load_config
    from bakery.trajectories.base import CheckpointId, TokenizerFingerprint
    from bakery.trajectories.theorem_qa import TheoremQABuilder
    from tests.test_theorem_qa import _world_and_bank

    monkeypatch.setattr(regmod, "make_generator", lambda backend, bundle: FakeGen([30, 31]))
    _, wpath, bpath, _ = _world_and_bank(tmp_path)
    cfg = load_config("bake_theorem_qa", cli_overrides={
        "data.world_spec": str(wpath), "data.probe_bank": str(bpath),
        "data.train_max_depth": "1", "data.per_depth_train_cap": "3",
        "data.sample_trajectories": "false",       # isolate the reg mechanism: teacher-force the main
        "generation.base_prompt": "", "generation.baked_prompt": "",
        "regularization.num_train_contexts": str(num_train_contexts),
        "regularization.eval_num_contexts": "2", "regularization.source": "synthetic",
        "regularization.context_split": "train", "regularization.max_new_tokens": "4",
    })
    fp = TokenizerFingerprint(name="wordtok", revision=None, vocab_size=100000,
                              bos_id=1, eos_id=2, pad_id=0)
    bundle = SimpleNamespace(tokenizer=WordTok(), tokenizer_fingerprint=fp,
                             base_checkpoint_id=CheckpointId(model_name="stub", revision=None,
                                                             dtype="float32"),
                             base=lambda: contextlib.nullcontext())
    return TheoremQABuilder().build(cfg, bundle=bundle, data_seed=0)


def test_regularization_grows_train_without_breaking_coverage(tmp_path, monkeypatch):
    ds0 = _build_theorem_qa(tmp_path, monkeypatch, num_train_contexts=0)
    ds3 = _build_theorem_qa(tmp_path, monkeypatch, num_train_contexts=3)

    assert ds3.stats["n_train_traj"] == ds0.stats["n_train_traj"] + 3      # exactly the anchors added
    # Coverage is computed from the trained RELATIONS, not the trajectory list -> anchors don't touch it.
    assert ds3.stats["pairing"]["coverage_depth1"] == ds0.stats["pairing"]["coverage_depth1"]
    assert ds0.stats["pairing"]["coverage_depth1"].split("/")[0] == ds0.stats["pairing"]["coverage_depth1"].split("/")[1]
    # OFF leaves the spec free of the provenance key; ON records it.
    assert "regularization" not in ds0.spec.extra
    assert ds3.spec.extra["regularization"]["num_train_contexts"] == 3


def test_regularization_anchor_x0_ids_disjoint_from_eval(tmp_path, monkeypatch):
    ds = _build_theorem_qa(tmp_path, monkeypatch, num_train_contexts=3)
    train_x0 = {t.x0_id for t in ds.train_trajectories}
    eval_x0 = {t.x0_id for t in ds.eval_trajectories}
    assert train_x0.isdisjoint(eval_x0)                        # criterion B held (anchors in 20000+ band)
    assert any(x >= 20_000 for x in train_x0)                  # the anchors are present
