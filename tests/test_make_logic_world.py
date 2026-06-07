"""Tests for the logic-world dataset generator (scripts/make_logic_world.py).

Guarantees the assets feeding the experiment are reproducible, vocabulary-disjoint across worlds,
depth-balanced, and — most importantly — that EVERY probe's truth label and depth match the
verified proof engine (the generator must never emit a mislabeled probe).
"""

import random

from bakery.logic.proof_engine import ProofEngine
from scripts.make_logic_world import WorldGenConfig, generate_world, make_probes


def _cfg(seed, used=None, **kw):
    base = dict(name=f"t{seed}", seed=seed, n_atoms=21, max_depth=4, n_components=3, true_per_depth=4)
    base.update(kw)
    return WorldGenConfig(name_used=used if used is not None else set(), **base)


def test_world_is_deterministic():
    w1 = generate_world(_cfg(0))
    w2 = generate_world(_cfg(0))
    assert w1 == w2                                   # frozen dataclass over tuples ⇒ structural eq
    assert w1.is_acyclic()


def test_vocabularies_are_disjoint_and_substring_free():
    used: set = set()
    wa = generate_world(_cfg(1, used))
    wb = generate_world(_cfg(2, used))
    sa, sb = set(wa.atoms), set(wb.atoms)
    assert sa.isdisjoint(sb)
    # no atom (from either world) is a substring of another atom — the guard matches by substring
    allnames = list(sa | sb)
    for i, x in enumerate(allnames):
        for y in allnames[i + 1:]:
            assert x.lower() not in y.lower() and y.lower() not in x.lower(), (x, y)


def test_probes_are_balanced_and_engine_verified():
    cfg = _cfg(3)
    w = generate_world(cfg)
    eng = ProofEngine(w)
    probes, realized = make_probes(w, eng, cfg, random.Random(cfg.seed + 1))
    comp_of = {a: c for c in w.components for a in c}

    n_true = sum(1 for p in probes if p["provable"])
    n_false = sum(1 for p in probes if not p["provable"])
    assert n_true > 0 and abs(n_true - n_false) <= cfg.max_depth   # balanced up to deep-cell starvation

    for p in probes:
        r = eng.query(p["subj"], p["obj"])
        assert r.provable == p["provable"], p["question"]
        if p["provable"]:
            assert p["form"] == "forward" and p["neg_type"] is None
            assert r.depth == p["proof_depth"]
            assert p["match_depth"] == p["proof_depth"]          # theorem: proof depth == pairing depth
            assert p["pos"] == " Yes"
            assert p["expect_heldout"] == (p["proof_depth"] >= 2)
        else:
            assert p["pos"] == " No" and p["expect_heldout"] is True
            assert p["proof_depth"] is None                      # non-theorem: no proof depth ...
            assert p["match_depth"] == p["hop"]                  # ... only a matched control depth
            a, b = p["entities"]
            if p["neg_type"] == "converse":
                assert eng.query(a, b).provable                   # underlying forward IS a theorem
            elif p["neg_type"] == "cross":
                assert comp_of[p["subj"]] != comp_of[p["obj"]]    # genuinely cross-component
            elif p["neg_type"] == "missing_edge":
                assert comp_of[p["subj"]] == comp_of[p["obj"]]    # same component, but unreachable
            else:
                raise AssertionError(f"unexpected neg_type {p['neg_type']!r}")

    # every requested depth produced both truths and falsehoods
    for d in range(1, cfg.max_depth + 1):
        assert realized[f"d{d}"]["true"] > 0 and realized[f"d{d}"]["false"] > 0
