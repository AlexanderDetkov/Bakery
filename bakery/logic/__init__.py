"""A tiny propositional-logic substrate for the knowledge-propagation study.

The task is reframed as a Hilbert-style proof system: AXIOMS are definite implications
"Every X is a Y" (`X ⊑ Y`), forming a DAG of fictional concepts; a query "Every X is a Z" is a
THEOREM iff `Z` is reachable from `X`, and its MINIMAL PROOF DEPTH = the shortest directed path
length = the number of modus-ponens applications (one per traversed edge). This makes "hop" a
proof length rather than a list index.

Two independent computations of reachability + minimal depth (BFS forward-chaining in
`ProofEngine`, all-pairs relaxation in `brute_force_reachability`) let a self-test prove the engine
does what we expect on every pair of every random world (see tests/test_proof_engine.py).
"""

from bakery.logic.proof_engine import ProofEngine, ProofResult, brute_force_reachability
from bakery.logic.world import Rule, World, is_acyclic, weak_components

__all__ = [
    "Rule",
    "World",
    "is_acyclic",
    "weak_components",
    "ProofEngine",
    "ProofResult",
    "brute_force_reachability",
]
