"""Provability + minimal proof depth over a definite-implication DAG.

Two independent implementations on purpose:

  * `ProofEngine` — BFS forward-chaining from a source atom. For a query `X ⊑ Z` it finds the
    SHORTEST directed path `X → … → Z`; the path length is the minimal proof depth (one
    modus-ponens application per traversed edge — see bakery/logic/__init__.py) and the edge
    sequence is the proof WITNESS.
  * `brute_force_reachability` — all-pairs shortest hops via Floyd–Warshall relaxation, a
    structurally different algorithm. The self-test (tests/test_proof_engine.py) asserts the two
    agree on provability AND minimal depth for every ordered pair of every random world; that
    agreement is the guarantee the labels feeding the experiment are correct.

Convention: `X ⊑ X` is provable at depth 0 (the identity), so both implementations include
`(x, x): 0`. The generator never emits reflexive probes; `proof_depth ≥ 1` for every real probe.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from bakery.logic.world import Rule, World


@dataclass(frozen=True)
class ProofResult:
    provable: bool
    depth: int | None                 # minimal proof depth (edges == MP steps); None if not provable
    witness: tuple[Rule, ...] | None  # the minimal rule sequence X→…→Z; () for the depth-0 identity


class ProofEngine:
    """Reachability + minimal-depth queries over a `World` (Tier-1 single-body rules)."""

    def __init__(self, world: World):
        self.world = world
        self._adj: dict = defaultdict(list)   # atom -> [(head, rule)] for single-body rules
        for r in world.rules:
            if len(r.body) == 1:
                self._adj[r.body[0]].append((r.head, r))

    def reachable_from(self, subj) -> dict:
        """BFS map {atom: minimal_depth} of everything provable from `subj` (incl. subj at 0)."""
        dist = {subj: 0}
        q = deque([subj])
        while q:
            u = q.popleft()
            for v, _ in self._adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def query(self, subj, obj) -> ProofResult:
        """Is `subj ⊑ obj` a theorem? If so, its minimal depth + a shortest-path witness."""
        if subj == obj:
            return ProofResult(True, 0, ())
        dist = {subj: 0}
        prev: dict = {}                        # atom -> (predecessor, rule) along a shortest path
        q = deque([subj])
        while q:
            u = q.popleft()
            for v, rule in self._adj.get(u, []):
                if v in dist:
                    continue
                dist[v] = dist[u] + 1
                prev[v] = (u, rule)
                if v == obj:
                    return ProofResult(True, dist[v], self._witness(prev, subj, obj))
                q.append(v)
        return ProofResult(False, None, None)

    def is_theorem(self, subj, obj) -> bool:
        return self.query(subj, obj).provable

    def query_ground(self, individual, obj) -> ProofResult:
        """Is `obj(individual)` derivable from the ground facts `X(individual)`?

        Minimal depth = the shortest path from any atom the individual is asserted to have to
        `obj` (the membership fact is the hypothesis, mirroring the universal proof in
        bakery/logic/__init__.py). Returns the best (shallowest) such derivation.
        """
        best: ProofResult | None = None
        for ind, atom in self.world.facts:
            if ind != individual:
                continue
            r = self.query(atom, obj)
            if r.provable and (best is None or r.depth < best.depth):
                best = r
        return best if best is not None else ProofResult(False, None, None)

    @staticmethod
    def _witness(prev, subj, obj) -> tuple:
        chain = []
        cur = obj
        while cur != subj:
            pred, rule = prev[cur]
            chain.append(rule)
            cur = pred
        chain.reverse()
        return tuple(chain)


def brute_force_reachability(world: World) -> dict:
    """All-pairs minimal hop counts via Floyd–Warshall relaxation (independent of `ProofEngine`).

    Returns `{(x, z): min_depth}` for every reachable ordered pair, including `(x, x): 0`. Used
    ONLY by the self-test as an oracle to cross-check provability and minimal depth.
    """
    atoms = list(world.atoms)
    INF = float("inf")
    dist = {(a, b): (0 if a == b else INF) for a in atoms for b in atoms}
    for u, v in world.edges():
        if 1 < dist[(u, v)]:
            dist[(u, v)] = 1
    for k in atoms:
        for i in atoms:
            dik = dist[(i, k)]
            if dik == INF:
                continue
            for j in atoms:
                nd = dik + dist[(k, j)]
                if nd < dist[(i, j)]:
                    dist[(i, j)] = nd
    return {pair: int(d) for pair, d in dist.items() if d != INF}
