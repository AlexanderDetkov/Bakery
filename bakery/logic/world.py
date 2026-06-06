"""The logic-world data model: atoms + definite implications (a DAG of `X ⊑ Y` axioms).

A `World` is pure data; all reasoning lives in `proof_engine.py`. Rules are stored as
`Rule(body, head)` so the model is Tier-2-ready: a Tier-1 implication has a single body atom
(`len(body) == 1`); conjunctive multi-premise rules (`len(body) >= 2`) can be added later WITHOUT
changing this representation or the engine's signature. A linear chain `A→B→C` is just a world
whose rules are the consecutive pairs — so this generalizes the old single-chain assets.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    """A definite implication. `body` are premise atoms (Tier-1 ⇒ length 1); `head` the conclusion.

    Surface form for a Tier-1 rule `Rule(("Grizzit",), "Plonth")` is "Every Grizzit is a Plonth".
    """

    body: tuple[str, ...]
    head: str


def _edges_of(rules) -> list:
    """Directed edges (premise → conclusion) of the Tier-1 (single-body) rules only."""
    return [(r.body[0], r.head) for r in rules if len(r.body) == 1]


def weak_components(atoms, edges) -> tuple:
    """Weakly-connected components (edges treated as undirected), each a sorted tuple of atoms.

    Used to build cross-component negatives: two atoms in different components have no directed
    path either way, so `(X, W)` across components is a guaranteed non-theorem.
    """
    parent = {a: a for a in atoms}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv

    groups: dict = defaultdict(list)
    for a in atoms:
        groups[find(a)].append(a)
    comps = [tuple(sorted(g)) for g in groups.values()]
    return tuple(sorted(comps, key=lambda c: c[0]))


def is_acyclic(atoms, edges) -> bool:
    """True iff the directed graph (atoms, edges) is a DAG (Kahn's algorithm)."""
    indeg = {a: 0 for a in atoms}
    adj: dict = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    q = deque(a for a in atoms if indeg[a] == 0)
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return seen == len(atoms)


@dataclass(frozen=True)
class World:
    """A set of fictional concept `atoms` related by definite-implication `rules` (a DAG).

    `components` (weakly-connected) and `facts` (optional ground `(individual, atom)` pairs) are
    carried for the dataset generator / guard; `components` is recomputed in `from_spec` so a
    hand-edited spec can't drift out of sync with `rules`.
    """

    atoms: tuple[str, ...]
    rules: tuple[Rule, ...]
    components: tuple[tuple[str, ...], ...]
    name: str
    facts: tuple[tuple[str, str], ...] = ()

    def edges(self) -> list:
        """Directed Tier-1 edges (premise → conclusion)."""
        return _edges_of(self.rules)

    def out_edges(self, atom) -> list:
        """Heads of the Tier-1 rules whose body is exactly `atom`."""
        return [h for (b, h) in self.edges() if b == atom]

    def is_acyclic(self) -> bool:
        return is_acyclic(self.atoms, self.edges())

    def to_spec(self) -> dict:
        """JSON-serializable world spec (`data/worlds/<name>.json`)."""
        return {
            "name": self.name,
            "atoms": list(self.atoms),
            "rules": [{"body": list(r.body), "head": r.head} for r in self.rules],
            "components": [list(c) for c in self.components],
            "facts": [list(f) for f in self.facts],
            "note": "Definite-implication DAG; edges = taught axioms. Acyclic by construction. "
                    "proof_depth(X,Z) = shortest directed path length = #modus-ponens steps.",
        }

    @classmethod
    def from_spec(cls, d: dict) -> "World":
        atoms = tuple(d["atoms"])
        rules = tuple(Rule(tuple(r["body"]), r["head"]) for r in d["rules"])
        facts = tuple(tuple(f) for f in d.get("facts", []))
        comps = weak_components(atoms, _edges_of(rules))   # recompute — never trust a stored copy
        return cls(atoms=atoms, rules=rules, components=comps, name=d["name"], facts=facts)


def make_world(name, atoms, rules, facts=()) -> World:
    """Construct a World, computing its weakly-connected components from the rules."""
    atoms = tuple(atoms)
    rules = tuple(rules)
    return World(
        atoms=atoms,
        rules=rules,
        components=weak_components(atoms, _edges_of(rules)),
        name=name,
        facts=tuple(facts),
    )
