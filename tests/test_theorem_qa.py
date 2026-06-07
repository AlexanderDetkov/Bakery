"""Tests for the teacher-forced QA curriculum builder (`theorem_qa`).

Pins the load-bearing properties: the supervised span is the WHOLE declarative answer (not one
token), the two framings differ ONLY by the system prompt, depth-1 coverage is total (every axiom
trained — hard-failed otherwise), the n-curriculum holds out depths > n, the eval-KL set is disjoint
from training, and the training question is IDENTICAL to the eval probe question (so the baked
yes/no decision transfers). Offline: a word-level fake tokenizer round-trips decode so the gate's
contamination validator runs without a model.

If a test here fails you broke an invariant — fix the code, not the test.
"""

import json
import random
from types import SimpleNamespace

import pytest

from bakery.config import load_config
from bakery.logic import phrasing
from bakery.logic.proof_engine import ProofEngine
from bakery.trajectories.base import CheckpointId, TokenizerFingerprint
from bakery.trajectories.encoding import iter_supervised_ids
from bakery.trajectories.theorem_qa import TheoremQABuilder


# --- offline word-level tokenizer (round-trips decode for the contamination validator) ----------

class WordTok:
    chat_template = None                          # forces build_prefix_ids' manual fallback

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


_FP = TokenizerFingerprint(name="wordtok", revision=None, vocab_size=100000,
                           bos_id=1, eos_id=2, pad_id=0)
_CKPT = CheckpointId(model_name="stub", revision=None, dtype="float32")


def _bundle():
    return SimpleNamespace(tokenizer=WordTok(), tokenizer_fingerprint=_FP, base_checkpoint_id=_CKPT)


def _world_and_bank(tmp_path, true_per_depth=4):
    """A tiny 2-component world + its QA probe bank, written to tmp_path (built via the generator)."""
    from scripts.make_logic_world import WorldGenConfig, generate_world, make_probes
    cfg = WorldGenConfig(name="tw", seed=3, n_atoms=16, max_depth=3, n_components=2,
                         true_per_depth=true_per_depth)
    world = generate_world(cfg)
    probes, realized = make_probes(world, ProofEngine(world), cfg, random.Random(cfg.seed + 1))
    wpath = tmp_path / "tw.json"
    wpath.write_text(json.dumps(world.to_spec()))
    bpath = tmp_path / "tw_qa.json"
    bpath.write_text(json.dumps({"probes": probes, "realized_counts": realized}))
    return world, wpath, bpath, probes


def _cfg(tmp_path, wpath, bpath, n, cap=3):
    # These tests pin the TEACHER-FORCED framing (the deterministic declarative answer), so they opt
    # into it explicitly; the bake DEFAULT is now sampling from the prompted teacher (covered by
    # test_sampling_frames_from_prompted_teacher, which needs a generator).
    return load_config("bake_theorem_qa", cli_overrides={
        "data.world_spec": str(wpath), "data.probe_bank": str(bpath),
        "data.train_max_depth": str(n), "data.per_depth_train_cap": str(cap),
        "data.sample_trajectories": "false",
        "generation.baked_prompt": "",
    })


def _build(tmp_path, n=2, cap=3, bank_mutator=None):
    world, wpath, bpath, probes = _world_and_bank(tmp_path)
    if bank_mutator is not None:
        probes = bank_mutator(probes)
        bpath.write_text(json.dumps({"probes": probes}))
    bundle = _bundle()
    ds = TheoremQABuilder().build(_cfg(tmp_path, wpath, bpath, n, cap), bundle=bundle, data_seed=0)
    return world, ds, bundle


# --- framing: supervised span = the WHOLE answer; framings differ only by system prompt ---------

def test_supervised_span_is_the_whole_declarative_answer(tmp_path):
    world, ds, bundle = _build(tmp_path, n=2)
    tok = bundle.tokenizer                                  # the SAME tokenizer the builder framed with
    for t in ds.train_trajectories:
        base_sup, baked_sup = iter_supervised_ids(t)
        assert base_sup == baked_sup                       # the gate guarantees this; double-check
        assert t.num_supervised >= 4                       # "Yes/No , every X is a Z ." -> many tokens
        # the two framings share the SAME suffix (answer) and differ only in the (system) prefix
        assert t.base_input_ids[-t.num_supervised:] == t.baked_input_ids[-t.num_supervised:]
        assert t.base_input_ids[: -t.num_supervised] != t.baked_input_ids[: -t.num_supervised]
        first = tok.decode([base_sup[0]]).strip()
        assert first in ("Yes", "No")                      # leading token is the scorable decision


def test_one_trajectory_per_relation_with_unique_ids(tmp_path):
    world, ds, _ = _build(tmp_path, n=2)
    xids = [t.x0_id for t in ds.train_trajectories]
    assert len(xids) == len(set(xids))                     # one context per trained relation
    eval_ids = {t.x0_id for t in ds.eval_trajectories}
    assert eval_ids.isdisjoint(set(xids))                  # eval-KL disjoint from train (criterion B)
    pairs = [(r[0], r[1]) for r in ds.stats["pairing"]["trained_relations"]]
    assert len(pairs) == len(set(pairs))                   # no relation trained twice (deduped)


# --- bake DEFAULT: sample y from the prompted teacher (canonical on-policy baking) ---------------

def test_sampling_frames_from_prompted_teacher(tmp_path, monkeypatch):
    # With the bake default (sample_trajectories=True), y is SAMPLED from base+u (adapter OFF). Drive
    # it with a fake generator (no model): the base framing carries the prompt, the baked framing is
    # empty, the supervised span is the SAMPLED y, trajectories_per_context multiplies the train set,
    # and coverage still holds (it keys on the relation list, not the sampled tokens).
    import contextlib

    import bakery.trajectories.generator as genmod

    class FakeGen:
        backend_name = "fake"

        def generate(self, prefixes, gen_cfg):
            return [[50000, 50001, 50002] for _ in prefixes]   # ids WordTok never assigns -> decode ""

    monkeypatch.setattr(genmod, "make_generator", lambda backend, bundle: FakeGen())
    world, wpath, bpath, _ = _world_and_bank(tmp_path)
    bundle = _bundle()
    bundle.base = lambda: contextlib.nullcontext()             # adapter-toggle stub (FakeGen ignores it)
    cfg = load_config("bake_theorem_qa", cli_overrides={
        "data.world_spec": str(wpath), "data.probe_bank": str(bpath),
        "data.train_max_depth": "1", "data.per_depth_train_cap": "3",
        "data.sample_trajectories": "true",
        "generation.base_prompt": "PROMPTHEADER", "generation.baked_prompt": "",
        "generation.trajectories_per_context": "2", "generation.max_new_tokens": "8",
    })
    ds = TheoremQABuilder().build(cfg, bundle=bundle, data_seed=0)

    t = ds.train_trajectories[0]
    assert t.base_input_ids != t.baked_input_ids              # base carries the prompt; baked is empty
    base_sup, baked_sup = iter_supervised_ids(t)
    assert base_sup == baked_sup == [50000, 50001, 50002]     # supervised span == the SAMPLED y
    n_trained = ds.stats["pairing"]["n_trained"]
    assert len(ds.train_trajectories) == n_trained * 2        # trajectories_per_context = 2, no drops
    assert ds.spec.extra["sampler_kind"] == "sampled_qa"
    assert ds.spec.trajectories_per_context == 2
    cov = ds.stats["pairing"]["coverage_depth1"]              # coverage unaffected by sampling
    assert cov.split("/")[0] == cov.split("/")[1]


# --- coverage: every axiom trained, hard-failed otherwise ---------------------------------------

def test_full_depth1_coverage_recorded(tmp_path):
    world, ds, _ = _build(tmp_path, n=1)
    pairing = ds.stats["pairing"]
    n_edges = len(set(world.edges()))
    assert pairing["coverage_depth1"] == f"{n_edges}/{n_edges}"
    trained = {(a, b) for a, b, d, prov in pairing["trained_relations"] if d == 1 and prov}
    assert trained == set(world.edges())                   # EVERY axiom is in training
    assert pairing["n_trained_neg"] > 0                    # closure carried


def test_coverage_gate_raises_when_an_axiom_is_excluded(tmp_path):
    # Marking a depth-1 edge probe expect_heldout forces the builder to drop it from training, so the
    # coverage criterion must HARD-FAIL (the baked model couldn't have all the prompt's information).
    def drop_one_edge(probes):
        for p in probes:
            if p["provable"] and p.get("proof_depth") == 1:
                p["expect_heldout"] = True                 # exclude this edge from training
                break
        return probes
    with pytest.raises(AssertionError, match="COVERAGE"):
        _build(tmp_path, n=1, bank_mutator=drop_one_edge)


def test_all_yes_would_fail_balance(tmp_path):
    # If a depth trained only Yes answers, the balance criterion fires (guards the all-Yes collapse).
    b = TheoremQABuilder()
    world, wpath, bpath, _ = _world_and_bank(tmp_path)
    b._plan(_cfg(tmp_path, wpath, bpath, n=1))
    b._plan_cache["trained"] = [{"subj": a, "obj": c, "depth": 1, "provable": True, "neg_type": None}
                                for (a, c) in world.edges()]          # strip the negatives
    with pytest.raises(AssertionError, match="BALANCE"):
        b.pairing_validator()(None, (), ())


# --- curriculum: depths > n are held out; training is disjoint from held-out probes -------------

def test_curriculum_holds_out_deeper_depths(tmp_path):
    _, ds1, _ = _build(tmp_path, n=1)
    depths1 = {d for _, _, d, _ in ds1.stats["pairing"]["trained_relations"]}
    assert depths1 == {1}                                  # n=1: nothing beyond depth-1 trained
    _, ds2, _ = _build(tmp_path, n=2)
    depths2 = {d for _, _, d, _ in ds2.stats["pairing"]["trained_relations"]}
    assert 2 in depths2 and depths2 <= {1, 2}              # n=2: depth-2 added, depth-3 still held out


def test_trained_relations_disjoint_from_heldout_probes(tmp_path):
    world, ds, _ = _build(tmp_path, n=2)
    trained = {(a, b) for a, b, d, prov in ds.stats["pairing"]["trained_relations"]}
    bank = json.loads((tmp_path / "tw_qa.json").read_text())["probes"]
    heldout = {(p["subj"], p["obj"]) for p in bank if p.get("expect_heldout")}
    assert trained.isdisjoint(heldout)                     # no leakage: trained never overlaps held-out


def test_trained_negatives_avoid_heldout_unordered_pairs(tmp_path):
    # A negative answer says "No, not every X is a Y", which mentions both entities and teaches an
    # explicit failure. That must not share an unordered entity pair with any held-out probe.
    world, ds, _ = _build(tmp_path, n=2)
    bank = json.loads((tmp_path / "tw_qa.json").read_text())["probes"]
    heldout_both = {(p["subj"], p["obj"]) for p in bank if p.get("expect_heldout")}
    heldout_both |= {(b, a) for (a, b) in heldout_both}
    trained_neg = {(a, b) for a, b, d, prov in ds.stats["pairing"]["trained_relations"] if not prov}
    assert trained_neg.isdisjoint(heldout_both)


def test_curriculum_split_is_stable_as_n_increases(tmp_path):
    # The depth-2 training subset for n=2 should be byte-identical to the depth-2 subset for n=3;
    # otherwise sweep arms differ in more than their added deeper curriculum.
    _, ds2, _ = _build(tmp_path, n=2, cap=2)
    _, ds3, _ = _build(tmp_path, n=3, cap=2)
    d2_at_n2 = sorted(r for r in ds2.stats["pairing"]["trained_relations"] if r[2] == 2)
    d2_at_n3 = sorted(r for r in ds3.stats["pairing"]["trained_relations"] if r[2] == 2)
    assert d2_at_n2 == d2_at_n3


def test_theorem_qa_contamination_stats_are_enforced(tmp_path):
    _, ds, _ = _build(tmp_path, n=2)
    stats = ds.stats["probe_contamination"]
    assert stats["enforced"] is True
    bank = json.loads((tmp_path / "tw_qa.json").read_text())["probes"]
    leaked = [
        p["question"]
        for p, label in zip(bank, stats["labels"])
        if p.get("expect_heldout") and label["label"] == "stated"
    ]
    assert leaked == []


def test_theorem_qa_contamination_validator_hard_fails_on_leak():
    # Integration check for the builder's own closure: if a train answer states an expect_heldout
    # theorem, the validator must raise, not merely record a label.
    from bakery.logic.world import Rule, make_world
    from tests import _invariant_kit as kit

    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return {tuple([10, 11]): "Yes, every Aaru is a Cesi."}[tuple(ids)]

    world = make_world("tw", ["Aaru", "Bofo", "Cesi", "Pulo", "Qora"], [
        Rule(("Aaru",), "Bofo"), Rule(("Bofo",), "Cesi"), Rule(("Pulo",), "Qora"),
    ])
    bank = [
        {"form": "forward", "neg_type": None, "provable": True, "subj": "Aaru", "obj": "Cesi",
         "entities": ["Aaru", "Cesi"], "proof_depth": 2, "hop": 2, "match_depth": 2,
         "expect_heldout": True, "question": phrasing.question("tw", "Aaru", "Cesi"),
         "pos": " Yes", "neg": " No"},
        {"form": "converse", "neg_type": "converse", "provable": False, "subj": "Cesi", "obj": "Aaru",
         "entities": ["Aaru", "Cesi"], "proof_depth": None, "hop": 2, "match_depth": 2,
         "expect_heldout": True, "question": phrasing.question("tw", "Cesi", "Aaru"),
         "pos": " No", "neg": " Yes"},
    ]
    b = TheoremQABuilder()
    b._plan_cache = {"world": world, "bank": bank}
    b._tokenizer = Tok()
    with pytest.raises(AssertionError, match="PROBE CONTAMINATION"):
        b.contamination_validator()([kit.make_traj(0, (10, 11))], [])


def test_theorem_qa_contamination_records_not_raises_when_sampling():
    # SAMPLED trajectories: the same leak that HARD-FAILS the teacher-forced path is RECORDED (not
    # raised) — the explicit research decision to keep on-policy CoT trajectories. enforced=False and
    # the leak is counted so the manifest stays honest about the contamination.
    from bakery.logic.world import Rule, make_world
    from tests import _invariant_kit as kit

    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return {tuple([10, 11]): "Yes, every Aaru is a Cesi."}[tuple(ids)]

    world = make_world("tw", ["Aaru", "Bofo", "Cesi", "Pulo", "Qora"], [
        Rule(("Aaru",), "Bofo"), Rule(("Bofo",), "Cesi"), Rule(("Pulo",), "Qora"),
    ])
    bank = [
        {"form": "forward", "neg_type": None, "provable": True, "subj": "Aaru", "obj": "Cesi",
         "entities": ["Aaru", "Cesi"], "proof_depth": 2, "hop": 2, "match_depth": 2,
         "expect_heldout": True, "question": phrasing.question("tw", "Aaru", "Cesi"),
         "pos": " Yes", "neg": " No"},
        {"form": "converse", "neg_type": "converse", "provable": False, "subj": "Cesi", "obj": "Aaru",
         "entities": ["Aaru", "Cesi"], "proof_depth": None, "hop": 2, "match_depth": 2,
         "expect_heldout": True, "question": phrasing.question("tw", "Cesi", "Aaru"),
         "pos": " No", "neg": " Yes"},
    ]
    b = TheoremQABuilder()
    b._plan_cache = {"world": world, "bank": bank}
    b._tokenizer = Tok()
    b._sampling = True                                  # the relaxation applies only to the sampled path
    stats = b.contamination_validator()([kit.make_traj(0, (10, 11))], [])
    assert stats["enforced"] is False and stats["sampled_cot_leak"] is True
    assert stats["n_heldout_stated"] >= 1               # the leak is RECORDED, transparently


# --- the training question is IDENTICAL to the eval probe question ------------------------------

def test_training_question_matches_eval_probe_question(tmp_path):
    _, _, _, probes = _world_and_bank(tmp_path)
    p = probes[0]
    assert p["question"] == phrasing.question("tw", p["subj"], p["obj"])
