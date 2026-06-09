"""Tests for the logic-world dataset generator (scripts/make_logic_world.py).

Guarantees the assets feeding the experiment are reproducible, vocabulary-disjoint across worlds,
depth-balanced, and — most importantly — that EVERY probe's truth label and depth match the
verified proof engine (the generator must never emit a mislabeled probe).
"""

import random

from bakery.logic import relations
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


# --- EQUIVALENCE mode: symmetric same-kind truth (additive; directed mode untouched) ------------

def test_equivalence_candidate_pools_have_empty_converse_and_missing():
    cfg = _cfg(7, relation_mode="equivalence")
    w = generate_world(cfg)
    eng = ProofEngine(w, relation_mode="equivalence")
    pools = relations.candidate_pools(w, eng, cfg.max_depth, mode="equivalence")
    comp_of = {a: c for c in w.components for a in c}
    for d in range(1, cfg.max_depth + 1):
        assert pools[d]["converse"] == []          # reverse of a same-kind pair is also true -> no converse neg
        assert pools[d]["missing"] == []           # within a component everything is reachable -> no missing neg
        assert pools[d]["cross"]                    # cross-component is the ONLY (and a populated) negative family
        for (x, z) in pools[d]["true"]:
            assert comp_of[x] == comp_of[z]         # TRUE pairs are same-component
            assert eng.query(x, z).provable and eng.query(z, x).provable   # symmetric
        for (x, w_) in pools[d]["cross"]:
            assert comp_of[x] != comp_of[w_]        # cross pairs are genuinely different components
            assert not eng.query(x, w_).provable


def test_equivalence_probes_balanced_engine_verified_and_only_cross_negatives():
    cfg = _cfg(9, relation_mode="equivalence")
    w = generate_world(cfg)
    eng = ProofEngine(w, relation_mode="equivalence")
    probes, realized = make_probes(w, eng, cfg, random.Random(cfg.seed + 1), mode="equivalence")
    comp_of = {a: c for c in w.components for a in c}

    n_true = sum(1 for p in probes if p["provable"])
    n_false = sum(1 for p in probes if not p["provable"])
    assert n_true > 0 and abs(n_true - n_false) <= cfg.max_depth

    for p in probes:
        r = eng.query(p["subj"], p["obj"])
        assert r.provable == p["provable"], p["question"]      # engine is the truth oracle
        assert "same kind" in p["question"]                    # symmetric phrasing
        if p["provable"]:
            assert p["form"] == "same_kind" and p["neg_type"] is None
            assert comp_of[p["subj"]] == comp_of[p["obj"]]     # same component
        else:
            assert p["neg_type"] == "cross"                    # the ONLY negative family
            assert comp_of[p["subj"]] != comp_of[p["obj"]]

    # balanced true/false per depth (so a constant Yes/No responder scores at chance)
    for d in range(1, cfg.max_depth + 1):
        assert realized[f"d{d}"]["true"] > 0 and realized[f"d{d}"]["false"] > 0
        bt = realized[f"d{d}"]["by_type"]
        assert bt["converse"] == 0 and bt["missing_edge"] == 0 and bt["cross"] > 0


def test_equivalence_world_dataset_passes_the_gate(tmp_path):
    # The validation gate must accept an equivalence-world dataset built through the gated builder
    # (reuses the offline word-level tokenizer + fixtures from test_theorem_qa).
    import json

    from tests.test_theorem_qa import WordTok, _FP, _CKPT
    from types import SimpleNamespace

    from bakery.config import load_config
    from bakery.trajectories.theorem_qa import TheoremQABuilder

    cfg = _cfg(11, relation_mode="equivalence", n_atoms=24, max_depth=3, n_components=2,
               true_per_depth=4)
    w = generate_world(cfg)
    eng = ProofEngine(w, relation_mode="equivalence")
    probes, realized = make_probes(w, eng, cfg, random.Random(cfg.seed + 1), mode="equivalence")
    wpath = tmp_path / "eqw.json"
    wpath.write_text(json.dumps(w.to_spec()))
    bpath = tmp_path / "eqw_qa.json"
    bpath.write_text(json.dumps({"probes": probes, "realized_counts": realized}))

    run_cfg = load_config("bake_theorem_qa_equiv", cli_overrides={
        "data.world_spec": str(wpath), "data.probe_bank": str(bpath),
        "data.relation_mode": "equivalence",
        "data.train_max_depth": "2", "data.per_depth_train_cap": "3",
        "data.sample_trajectories": "false",
        "generation.base_prompt": "data/prompts/empty.md", "generation.baked_prompt": "",
    })
    bundle = SimpleNamespace(tokenizer=WordTok(), tokenizer_fingerprint=_FP, base_checkpoint_id=_CKPT)
    ds = TheoremQABuilder().build(run_cfg, bundle=bundle, data_seed=0)

    assert ds.spec.extra["relation_mode"] == "equivalence"
    # coverage: every taught edge is trained (in equivalence the reverse is trained too)
    cov = ds.stats["pairing"]["coverage_depth1"]
    assert int(cov.split("/")[0]) >= int(cov.split("/")[1])
    assert ds.stats["pairing"]["n_trained_neg"] > 0           # cross-component negatives carried
    # the supervised span is the symmetric same-kind answer
    tok = bundle.tokenizer
    from bakery.trajectories.encoding import iter_supervised_ids
    sup = tok.decode(iter_supervised_ids(ds.train_trajectories[0])[0])
    assert "same kind" in sup
