"""DAG contamination guard (the multi-component analogue of the linear-chain guard).

Pins the guard's two jobs on a small explicit world: the atomic-source FILTER
(`states_beyond_atomic`) and the DIRECTION-AWARE labeler (`label_probes_dag`) + the schema/balance
check. The crux test is `test_reverse_direction_link_does_not_leak_probe`: a true atomic link
"Every Bofo is a Cesi" shares both entities with the reverse probe "is every Cesi a Bofo?", yet
must NOT mark it stated — only a SUBJ→OBJ assertion (or an explicit reversal) leaks a probe.

If a test here fails, you broke the contamination invariant — fix the code, not the test.
"""

import pytest

from bakery.trajectories.contamination import assert_probe_schema_and_balance
from bakery.trajectories.contamination_dag import (
    label_probes_dag,
    present_atoms,
    states_beyond_atomic,
)
from bakery.logic.world import Rule, make_world

# comp1: Aaru→Bofo→Cesi→Dovu  ;  comp2: Pulo→Qora   (multi-letter, substring-free names)
WORLD = make_world("tw", ["Aaru", "Bofo", "Cesi", "Dovu", "Pulo", "Qora"], [
    Rule(("Aaru",), "Bofo"), Rule(("Bofo",), "Cesi"), Rule(("Cesi",), "Dovu"), Rule(("Pulo",), "Qora"),
])
ATOMIC = ["Every Aaru is a Bofo.", "Every Bofo is a Cesi.", "Every Cesi is a Dovu.",
          "Every Pulo is a Qora."]


def _p(form, neg_type, provable, subj, obj, d, expect_heldout):
    return {"form": form, "neg_type": neg_type, "provable": provable, "subj": subj, "obj": obj,
            "entities": [subj, obj] if form != "converse" else [obj, subj],
            "proof_depth": d, "hop": d, "expect_heldout": expect_heldout,
            "question": f"is every {subj} a {obj}?", "pos": " Yes" if provable else " No",
            "neg": " No" if provable else " Yes"}


# --- FILTER -----------------------------------------------------------------------------

def test_filter_keeps_single_atomic_link():
    assert states_beyond_atomic("Every Aaru is a Bofo.", WORLD) is False
    assert states_beyond_atomic("Aaru are a kind of creature.", WORLD) is False


def test_filter_drops_composed_relation():
    assert states_beyond_atomic("Every Aaru is a Cesi.", WORLD) is True       # A,C not adjacent
    assert states_beyond_atomic("An Aaru is a Bofo, which is a Cesi.", WORLD) is True   # 3 atoms


def test_filter_drops_cross_component_and_reversal():
    assert states_beyond_atomic("Every Aaru is a Pulo.", WORLD) is True       # cross-component pair
    assert states_beyond_atomic("Every Aaru is a Bofo, but not every Bofo is an Aaru.", WORLD) is True


def test_filter_drops_reversed_atomic_statement():
    # taught edge is Aaru→Bofo; a REVERSED statement "Every Bofo is an Aaru" is false w.r.t. the
    # axioms and would contaminate the converse probe -> must be dropped at generation.
    assert states_beyond_atomic("Every Bofo is an Aaru.", WORLD) is True
    assert states_beyond_atomic("Every Aaru is a Bofo.", WORLD) is False      # taught direction kept


def test_present_atoms():
    assert present_atoms("Every Bofo is a Cesi.", WORLD) == {"Bofo", "Cesi"}


# --- DIRECTION-AWARE LABELER ------------------------------------------------------------

def _label(probe, conts=ATOMIC):
    return label_probes_dag([probe], conts, WORLD)[0][0]["label"]


def test_forward_atomic_is_stated():
    assert _label(_p("forward", None, True, "Aaru", "Bofo", 1, False)) == "stated"


def test_composed_forward_is_heldout():
    assert _label(_p("forward", None, True, "Aaru", "Cesi", 2, True)) == "held_out"


def test_converse_is_heldout_under_atomic_source():
    assert _label(_p("converse", "converse", False, "Cesi", "Aaru", 2, True)) == "held_out"


def test_cross_is_heldout():
    assert _label(_p("cross", "cross", False, "Aaru", "Pulo", 1, True)) == "held_out"


def test_reverse_direction_link_does_not_leak_probe():
    # "Every Bofo is a Cesi" shares both entities with "is every Cesi a Bofo?" but asserts B⊑C only.
    assert _label(_p("missing_edge", "missing_edge", False, "Cesi", "Bofo", 2, True)) == "held_out"


def test_composed_forward_leak_is_stated():
    leaked = ATOMIC + ["Every Aaru is a Cesi."]
    assert _label(_p("forward", None, True, "Aaru", "Cesi", 2, True), leaked) == "stated"


def test_explicit_reverse_assertion_leaks_the_probe():
    leaked = ATOMIC + ["Every Cesi is a Bofo."]            # asserts C⊑B directly
    assert _label(_p("missing_edge", "missing_edge", False, "Cesi", "Bofo", 2, True), leaked) == "stated"


def test_negation_with_cue_leaks_the_probe():
    leaked = ATOMIC + ["Not every Cesi is a Bofo."]        # explicitly teaches the (No) answer
    assert _label(_p("missing_edge", "missing_edge", False, "Cesi", "Bofo", 2, True), leaked) == "stated"


# --- HARD-FAIL on a leaked held-out probe -----------------------------------------------

def test_assert_probes_heldout_via_dag_labels():
    from bakery.trajectories.contamination_dag import assert_probes_heldout
    probes = [_p("forward", None, True, "Aaru", "Cesi", 2, True)]
    labels, _ = label_probes_dag(probes, ATOMIC + ["Every Aaru is a Cesi."], WORLD)
    with pytest.raises(AssertionError, match="PROBE CONTAMINATION"):
        assert_probes_heldout(probes, labels)


# --- SCHEMA + BALANCE -------------------------------------------------------------------

def _balanced_bank():
    return [
        _p("forward", None, True, "Aaru", "Bofo", 1, False),
        _p("cross", "cross", False, "Aaru", "Pulo", 1, True),
        _p("forward", None, True, "Aaru", "Cesi", 2, True),
        _p("converse", "converse", False, "Cesi", "Aaru", 2, True),
    ]


def test_schema_and_balance_ok_on_balanced_bank():
    summary = assert_probe_schema_and_balance(_balanced_bank(), require_balanced_for_dprime=True)
    assert summary["unbalanced"] == {}
    assert summary["by_depth"]["d2"] == {"true": 1, "false": 1}


def test_schema_raises_on_missing_field():
    bad = _balanced_bank()
    del bad[0]["pos"]
    with pytest.raises(AssertionError, match="PROBE SCHEMA"):
        assert_probe_schema_and_balance(bad)


def test_balance_raises_on_one_sided_depth():
    only_true = [_p("forward", None, True, "Aaru", "Bofo", 1, False),
                 _p("forward", None, True, "Aaru", "Cesi", 2, True)]
    with pytest.raises(AssertionError, match="PROBE BALANCE"):
        assert_probe_schema_and_balance(only_true, require_balanced_for_dprime=True)
    # free source: recorded, not enforced
    summary = assert_probe_schema_and_balance(only_true, require_balanced_for_dprime=False)
    assert "d1" in summary["unbalanced"] and "d2" in summary["unbalanced"]
