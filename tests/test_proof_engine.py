"""Self-test for the propositional proof substrate (bakery/logic).

The whole experiment's ground truth (which probes are provable, and at what minimal depth) comes
from `ProofEngine`. These tests prove the engine does what we expect by cross-checking it against
an INDEPENDENT oracle (`brute_force_reachability`, Floyd–Warshall) on every ordered pair of many
random acyclic worlds — provability AND minimal depth must agree. If a test here fails, the engine
(not the test) is wrong: a mislabeled probe would silently invalidate the propagation result.
"""

import random

from bakery.logic import (
    ProofEngine,
    Rule,
    brute_force_reachability,
)
from bakery.logic.world import make_world


def _random_acyclic_world(seed: int, n: int = 12, p: float = 0.25):
    """Random DAG: atoms in a fixed topological order; edges only go forward ⇒ acyclic."""
    rng = random.Random(seed)
    atoms = [f"A{seed}_{i}" for i in range(n)]
    rules = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                rules.append(Rule((atoms[i],), atoms[j]))
    return make_world(f"rand_{seed}", atoms, rules)


def _witness_chains(witness, subj, obj) -> bool:
    """A witness is a valid derivation: rules link subj → … → obj, each single-body."""
    if not witness:
        return subj == obj
    if witness[0].body[0] != subj or witness[-1].head != obj:
        return False
    for r in witness:
        if len(r.body) != 1:
            return False
    for a, b in zip(witness, witness[1:]):
        if a.head != b.body[0]:
            return False
    return True


def test_engine_matches_bruteforce_on_every_pair():
    for seed in range(20):
        w = _random_acyclic_world(seed)
        eng = ProofEngine(w)
        oracle = brute_force_reachability(w)
        for x in w.atoms:
            for z in w.atoms:
                r = eng.query(x, z)
                assert r.provable == ((x, z) in oracle), (seed, x, z, r.provable)
                if r.provable:
                    assert r.depth == oracle[(x, z)], (seed, x, z, r.depth, oracle[(x, z)])
                    assert len(r.witness) == r.depth          # one MP step per edge
                    assert _witness_chains(r.witness, x, z)


def test_generated_worlds_are_acyclic():
    for seed in range(20):
        assert _random_acyclic_world(seed).is_acyclic()


def test_reflexive_identity_is_depth_zero():
    w = _random_acyclic_world(0)
    r = ProofEngine(w).query(w.atoms[0], w.atoms[0])
    assert r.provable and r.depth == 0 and r.witness == ()


def test_reachable_from_matches_query():
    w = _random_acyclic_world(3)
    eng = ProofEngine(w)
    for x in w.atoms:
        dist = eng.reachable_from(x)
        for z in w.atoms:
            r = eng.query(x, z)
            assert (z in dist) == r.provable
            if r.provable:
                assert dist[z] == r.depth


def test_linear_chain_depths_and_witness():
    # The old single-chain assets are a degenerate world: consecutive-pair rules.
    atoms = ["Grizzit", "Plonth", "Vurmal", "Kesdo"]
    rules = [Rule((atoms[i],), atoms[i + 1]) for i in range(len(atoms) - 1)]
    eng = ProofEngine(make_world("grixlike", atoms, rules))
    assert eng.query("Grizzit", "Plonth").depth == 1     # atomic link
    assert eng.query("Grizzit", "Vurmal").depth == 2     # one composition step
    assert eng.query("Grizzit", "Kesdo").depth == 3
    assert not eng.query("Kesdo", "Grizzit").provable    # converse is a non-theorem
    w = eng.query("Grizzit", "Kesdo")
    assert _witness_chains(w.witness, "Grizzit", "Kesdo")


def test_branching_shortest_path():
    # Two paths Plonth→Kesdo: via Vurmal (len 2) and via Morv (len 2); a 3rd longer route.
    rules = [
        Rule(("Grizzit",), "Plonth"),
        Rule(("Plonth",), "Vurmal"), Rule(("Vurmal",), "Kesdo"),
        Rule(("Plonth",), "Morv"), Rule(("Morv",), "Kesdo"),
        Rule(("Grizzit",), "Slo"), Rule(("Slo",), "Tup"), Rule(("Tup",), "Kesdo"),
    ]
    atoms = ["Grizzit", "Plonth", "Vurmal", "Kesdo", "Morv", "Slo", "Tup"]
    eng = ProofEngine(make_world("branch", atoms, rules))
    assert eng.query("Plonth", "Kesdo").depth == 2       # shortest of the two len-2 routes
    assert eng.query("Grizzit", "Kesdo").depth == 3      # min over routes (not the len-4 Slo route)


def test_ground_fact_query():
    rules = [Rule(("Grizzit",), "Plonth"), Rule(("Plonth",), "Vurmal")]
    w = make_world("g", ["Grizzit", "Plonth", "Vurmal"], rules, facts=[("Kim", "Grizzit")])
    eng = ProofEngine(w)
    assert eng.query_ground("Kim", "Vurmal").provable
    assert eng.query_ground("Kim", "Vurmal").depth == 2
    assert not eng.query_ground("Kim", "Nope").provable
