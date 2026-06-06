"""Probe<->trajectory contamination guard (criterion F).

Genuine propagation can only be claimed on consequences the trajectories never stated. These tests
pin the guard's two jobs: the atomic-source FILTER (drop composed/reverse continuations) and the
held-out LABELER (stated vs held_out, direction-aware), plus the gate wiring (stats recorded; an
`expect_heldout` probe that is leaked makes the dataset un-constructable).

If a test here fails, you broke an invariant — fix the code, not the test.
"""

import pytest

from bakery.trajectories.contamination import (
    assert_probes_heldout,
    label_probes,
    probe_distance,
    states_composed_or_reverse,
)
from tests import _invariant_kit as kit

CHAIN = ["Grizzit", "Plonth", "Vurmal", "Kesdo"]   # positions 0..3; atomic = consecutive pairs


# --- FILTER: keep a single forward atomic link; drop composed / reverse -----------------

def test_filter_keeps_single_atomic_link():
    assert states_composed_or_reverse("Every Grizzit is a Plonth.", CHAIN) is False
    assert states_composed_or_reverse("Grizzits are a kind of creature.", CHAIN) is False


def test_filter_drops_composed_relation():
    # mentions non-adjacent entities (Grizzit pos0 + Vurmal pos2) -> a composed statement
    assert states_composed_or_reverse("A Grizzit is a Plonth, which is a Vurmal.", CHAIN) is True
    assert states_composed_or_reverse("Every Grizzit is ultimately a Kesdo.", CHAIN) is True


def test_filter_drops_reverse_statement():
    # adjacent entities but a reversal cue -> a one-directionality / reverse claim
    assert states_composed_or_reverse(
        "Every Grizzit is a Plonth, but not every Plonth is a Grizzit.", CHAIN) is True


# --- LABELER: direction-aware stated vs held_out ----------------------------------------

def _label(probe, continuations):
    labels, _ = label_probes([probe], continuations, CHAIN)
    return labels[0]["label"]


ATOMIC_CONTS = ["Every Grizzit is a Plonth.", "Every Plonth is a Vurmal."]


def test_forward_atomic_is_stated():
    pr = {"form": "forward", "entities": ["Grizzit", "Plonth"], "hop": 1}
    assert _label(pr, ATOMIC_CONTS) == "stated"


def test_forward_composed_is_heldout_under_atomic_source():
    # Grizzit+Vurmal never co-occur in a single atomic continuation -> genuinely held out
    pr = {"form": "forward", "entities": ["Grizzit", "Vurmal"], "hop": 2}
    assert _label(pr, ATOMIC_CONTS) == "held_out"


def test_converse_not_stated_by_bare_forward_sentence():
    # "Every Grizzit is a Plonth" co-occurs the entities but does NOT assert the converse
    pr = {"form": "converse", "entities": ["Grizzit", "Plonth"], "hop": 1}
    assert _label(pr, ATOMIC_CONTS) == "held_out"


def test_converse_stated_only_with_reversal_cue():
    pr = {"form": "converse", "entities": ["Grizzit", "Plonth"], "hop": 1}
    assert _label(pr, ["Not every Plonth is a Grizzit."]) == "stated"


def test_probe_distance():
    assert probe_distance({"hop": 1}) == 0      # atomic link
    assert probe_distance({"hop": 3}) == 2      # two composition steps beyond the source


# --- HARD-FAIL: a leaked expect_heldout probe is un-constructable ------------------------

def test_assert_probes_heldout_raises_on_leak():
    probes = [{"form": "forward", "entities": ["Grizzit", "Vurmal"], "hop": 2, "expect_heldout": True,
               "question": "is every Grizzit a Vurmal?"}]
    leaked_labels = [{"label": "stated", "form": "forward", "distance": 1}]
    with pytest.raises(AssertionError, match="PROBE CONTAMINATION"):
        assert_probes_heldout(probes, leaked_labels)


def test_assert_probes_heldout_ok_when_held_out():
    probes = [{"form": "forward", "entities": ["Grizzit", "Vurmal"], "hop": 2, "expect_heldout": True}]
    assert_probes_heldout(probes, [{"label": "held_out", "form": "forward", "distance": 1}])  # no raise


# --- GATE WIRING: stats recorded; validator failure propagates --------------------------

def test_gate_records_contamination_stats():
    stats = {"labels": [{"label": "held_out", "form": "forward", "distance": 1}], "n_stated": 0}
    data = kit.gate([kit.make_traj(0, (10, 11))], [kit.make_traj(1, (12,))],
                    contamination_validator=lambda tr, ev: stats)
    assert data.stats["probe_contamination"] == stats


def test_gate_propagates_contamination_failure():
    def boom(train, eval_):
        raise AssertionError("PROBE CONTAMINATION: held-out probe leaked")
    with pytest.raises(AssertionError, match="PROBE CONTAMINATION"):
        kit.gate([kit.make_traj(0, (10, 11))], [kit.make_traj(1, (12,))],
                 contamination_validator=boom)


def test_gate_without_validator_has_empty_contamination():
    data = kit.gate([kit.make_traj(0, (10, 11))], [kit.make_traj(1, (12,))])
    assert data.stats["probe_contamination"] == {}


# --- the REAL builder closure: decode -> label -> assert -> stats (no model needed) -----

class _FakeTok:
    def __init__(self, mapping):
        self.mapping = mapping

    def decode(self, ids, skip_special_tokens=True):
        return self.mapping[tuple(ids)]


def _builder(probes, texts):
    from bakery.trajectories.fact_propagation import FactPropagationBuilder
    b = FactPropagationBuilder()
    b._tokenizer = _FakeTok(texts)
    b._chain = CHAIN
    b._probes = probes
    return b


def test_builder_closure_labels_and_records():
    # supervised baked ids of make_traj(x0,(a,b)) are [a,b]; map them to atomic-link sentences
    t1, t2 = kit.make_traj(0, (10, 11)), kit.make_traj(1, (12, 13))
    texts = {(10, 11): "Every Grizzit is a Plonth.", (12, 13): "Every Plonth is a Vurmal."}
    probes = [
        {"form": "forward", "entities": ["Grizzit", "Plonth"], "hop": 1, "expect_heldout": False},
        {"form": "forward", "entities": ["Grizzit", "Vurmal"], "hop": 2, "expect_heldout": True},
        {"form": "converse", "entities": ["Grizzit", "Plonth"], "hop": 1, "expect_heldout": True},
    ]
    stats = _builder(probes, texts).contamination_validator()([t1, t2], [])
    assert [l["label"] for l in stats["labels"]] == ["stated", "held_out", "held_out"]
    assert stats["n_stated"] == 1 and stats["n_held_out"] == 2


def test_builder_closure_hard_fails_on_leak():
    t1 = kit.make_traj(0, (10, 11))
    texts = {(10, 11): "Every Grizzit is a Plonth."}
    # an expect_heldout forward probe whose relation IS stated by the continuation -> un-constructable
    probes = [{"form": "forward", "entities": ["Grizzit", "Plonth"], "hop": 1,
               "expect_heldout": True, "question": "is every Grizzit a Plonth?"}]
    with pytest.raises(AssertionError, match="PROBE CONTAMINATION"):
        _builder(probes, texts).contamination_validator()([t1], [])


def test_builder_closure_free_mode_labels_without_raising():
    # Observational (free) source: the guard LABELS coverage but must NOT hard-fail, even when a
    # composed probe tagged expect_heldout is stated (free sampling legitimately states it).
    t1 = kit.make_traj(0, (10, 11))
    texts = {(10, 11): "Every Grizzit is a Plonth."}
    probes = [{"form": "forward", "entities": ["Grizzit", "Plonth"], "hop": 1,
               "expect_heldout": True, "question": "is every Grizzit a Plonth?"}]
    b = _builder(probes, texts)
    b._source_control = "free"
    stats = b.contamination_validator()([t1], [])
    assert stats["labels"][0]["label"] == "stated"
    assert stats["enforced"] is False


def test_builder_closure_opts_out_without_chain_or_probes():
    from bakery.trajectories.fact_propagation import FactPropagationBuilder
    b = FactPropagationBuilder()
    b._tokenizer, b._chain, b._probes = _FakeTok({}), [], []
    assert b.contamination_validator() is None
