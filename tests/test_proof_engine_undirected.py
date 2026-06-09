"""Self-test for EQUIVALENCE-mode (`relation_mode="equivalence"`) of the proof substrate.

The equivalence-relation world's ground truth is SYMMETRIC "same kind" = "X and Z lie in the same
weakly-connected component" (the rst closure of the taught edges). As with directed mode, the only
guard the gate does NOT provide is the semantic correctness of this truth, so it is cross-checked
here against an INDEPENDENT oracle (`brute_force_reachability_undirected`, Floyd–Warshall on the
SYMMETRIZED edge set — structurally distinct from the engine's symmetrized BFS) on every ordered
pair of many random worlds: provability AND minimal undirected depth must agree, the relation must
be symmetric, and provable must coincide exactly with same-weakly-connected-component.

If a test here fails the engine (not the test) is wrong: a mislabeled same-kind probe would
silently invalidate the equivalence-world propagation result. Directed mode is unaffected — it has
its own self-test in tests/test_proof_engine.py.
"""

import random

from bakery.logic import (
    ProofEngine,
    Rule,
    brute_force_reachability_undirected,
)
from bakery.logic.world import make_world, weak_components


def _random_acyclic_world(seed: int, n: int = 12, p: float = 0.25):
    """Random DAG (same generator as the directed self-test): forward-only edges ⇒ acyclic."""
    rng = random.Random(seed)
    atoms = [f"A{seed}_{i}" for i in range(n)]
    rules = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                rules.append(Rule((atoms[i],), atoms[j]))
    return make_world(f"rand_{seed}", atoms, rules)


def test_equivalence_engine_matches_undirected_oracle_on_every_pair():
    for seed in range(20):
        w = _random_acyclic_world(seed)
        eng = ProofEngine(w, relation_mode="equivalence")
        oracle = brute_force_reachability_undirected(w)
        for x in w.atoms:
            for z in w.atoms:
                r = eng.query(x, z)
                assert r.provable == ((x, z) in oracle), (seed, x, z, r.provable)
                if r.provable:
                    assert r.depth == oracle[(x, z)], (seed, x, z, r.depth, oracle[(x, z)])


def test_equivalence_query_is_symmetric():
    for seed in range(20):
        w = _random_acyclic_world(seed)
        eng = ProofEngine(w, relation_mode="equivalence")
        for x in w.atoms:
            for z in w.atoms:
                a, b = eng.query(x, z), eng.query(z, x)
                assert a.provable == b.provable, (seed, x, z)
                if a.provable:
                    assert a.depth == b.depth, (seed, x, z, a.depth, b.depth)


def test_equivalence_provable_iff_same_weak_component():
    for seed in range(20):
        w = _random_acyclic_world(seed)
        eng = ProofEngine(w, relation_mode="equivalence")
        comp_of = {a: c for c in weak_components(w.atoms, w.edges()) for a in c}
        for x in w.atoms:
            for z in w.atoms:
                same_component = comp_of[x] == comp_of[z]
                assert eng.query(x, z).provable == same_component, (seed, x, z)


def test_equivalence_reflexive_identity_is_depth_zero():
    w = _random_acyclic_world(0)
    r = ProofEngine(w, relation_mode="equivalence").query(w.atoms[0], w.atoms[0])
    assert r.provable and r.depth == 0


def test_equivalence_makes_converse_of_an_edge_provable():
    # In a directed chain A→B→C the converse C⊑A is a NON-theorem; under equivalence it IS a theorem
    # (same component), and at the same undirected depth as the forward pair.
    atoms = ["Grizzit", "Plonth", "Vurmal", "Kesdo"]
    rules = [Rule((atoms[i],), atoms[i + 1]) for i in range(len(atoms) - 1)]
    w = make_world("grixlike", atoms, rules)
    directed = ProofEngine(w)                              # default mode unchanged
    equiv = ProofEngine(w, relation_mode="equivalence")
    assert not directed.query("Kesdo", "Grizzit").provable          # directed converse fails
    fwd, rev = equiv.query("Grizzit", "Kesdo"), equiv.query("Kesdo", "Grizzit")
    assert fwd.provable and rev.provable                            # both directions same-kind
    assert fwd.depth == rev.depth == 3                              # undirected depth, symmetric


def test_directed_mode_unchanged_by_equivalence_addition():
    # The default engine is byte-identical to before: directed reachability + directed depth.
    atoms = ["Grizzit", "Plonth", "Vurmal", "Kesdo"]
    rules = [Rule((atoms[i],), atoms[i + 1]) for i in range(len(atoms) - 1)]
    eng = ProofEngine(make_world("grixlike", atoms, rules))         # default relation_mode
    assert eng.relation_mode == "directed"
    assert eng.query("Grizzit", "Kesdo").depth == 3
    assert not eng.query("Kesdo", "Grizzit").provable
