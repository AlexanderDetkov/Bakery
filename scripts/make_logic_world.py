"""Generate propositional logic-world assets for the knowledge-propagation study (reproducible).

Each WORLD is a DAG of fictional concepts related by definite implications "Every X is a Y". The
ONLY thing injected (prompted as `u`, or baked from atomic-only trajectories) is the set of atomic
implications (the edges). A probe "is every X a Z?" is then a forced-choice test of whether that
fact propagated; its proof depth = the shortest-path length from X to Z (number of modus-ponens
steps), computed by the verified `ProofEngine`.

The crux is DEPTH-MATCHED NEGATIVES so a "always Yes/No" responder scores at chance. At every depth
`d` we emit balanced TRUE (provable forward) and FALSE (non-theorem) probes, the FALSE ones of three
engine-verified types:
  * converse      — the reverse of a true depth-`d` entailment (direction axis);
  * cross         — a pair in two disjoint components (no path either way);
  * missing_edge  — a real prefix path X→…→C of length d-1 then a non-taught terminal edge to a real
                    atom W (the strongest negative: only one absent edge separates yes from no).
Every negative is kept ONLY if `engine.query(...).provable is False` — truth labels come from the
engine, never from graph heuristics.

Emits per world: data/worlds/<name>.json (atoms+rules+components), data/prompts/<name>_u.md (atomic
axioms only), data/contexts/<name>_contexts.json (atomic-eliciting), data/probes/<name>_probes.json
(extended schema). Run:  python scripts/make_logic_world.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from bakery.logic.proof_engine import ProofEngine
from bakery.logic.world import Rule, World, make_world

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "Answer with only Yes or No. In {world}, "

_SYL_C = "bdfgklmnprstvz"
_SYL_V = "aeiou"


# --------------------------------------------------------------------------- names

def _gen_names(rng: random.Random, n: int, used: set) -> list:
    """`n` pronounceable, globally-unique concept names with NO substring collisions (either way).

    Substring-freedom matters: the contamination guard detects an atom in text by case-insensitive
    substring, so one name being a substring of another would cross-trigger co-occurrence.
    """
    out: list = []
    while len(out) < n:
        k = rng.choice((2, 3))
        name = "".join(rng.choice(_SYL_C) + rng.choice(_SYL_V) for _ in range(k)).capitalize()
        if len(name) < 4:
            continue
        low = name.lower()
        if any(low in u or u in low for u in used):
            continue
        used.add(low)
        out.append(name)
    return out


# --------------------------------------------------------------------------- world

@dataclass
class WorldGenConfig:
    name: str
    seed: int
    n_atoms: int = 30
    max_depth: int = 6
    n_components: int = 3
    extra_edge_prob: float = 0.18      # chance of an additional forward edge (branching/shortcuts)
    true_per_depth: int = 6            # TRUE forward probes per depth; negatives matched 1:1
    name_used: set = field(default_factory=set)   # global vocabulary (disjointness across worlds)


def generate_world(cfg: WorldGenConfig) -> World:
    """Build an acyclic multi-component world deep enough to supply depths 1..max_depth."""
    rng = random.Random(cfg.seed)
    names = _gen_names(rng, cfg.n_atoms, cfg.name_used)
    # Split atoms across components; each component must hold a full backbone (max_depth+1 atoms).
    comp_min = cfg.max_depth + 1
    if cfg.n_atoms < cfg.n_components * comp_min:
        raise ValueError(f"n_atoms={cfg.n_atoms} too small for {cfg.n_components} components of "
                         f"backbone length {comp_min}.")
    sizes = [comp_min] * cfg.n_components
    for i in range(cfg.n_atoms - sum(sizes)):       # hand out the remainder round-robin
        sizes[i % cfg.n_components] += 1

    rules: list = []
    idx = 0
    for size in sizes:
        comp = names[idx: idx + size]
        idx += size
        # backbone chain: comp[0] -> comp[1] -> ... -> comp[max_depth]  (guarantees depths 1..max_depth)
        for a, b in zip(comp, comp[1:cfg.max_depth + 1]):
            rules.append(Rule((a,), b))
        # Remaining atoms attach as NEW leaves/side-chains: each receives one forward edge from an
        # earlier atom (any backbone or earlier branch atom). Edges only LAND on these new atoms, so
        # they never shorten a backbone-to-backbone path — the backbone stays the shortest route to
        # its nodes (deep depths are preserved) — while still creating siblings + extra deep pairs
        # (the branching that the missing-edge negatives need). Acyclic: parent index < child index.
        for j in range(cfg.max_depth + 1, size):
            # bias toward extending the most-recent branch atom (deeper side-chains) vs branching off
            parent = comp[j - 1] if (j > cfg.max_depth + 1 and rng.random() < cfg.extra_edge_prob) \
                else comp[rng.randrange(j)]
            rules.append(Rule((parent,), comp[j]))

    w = make_world(cfg.name, names, _dedup_rules(rules))
    assert w.is_acyclic(), "generated world is not acyclic"
    return w


def _dedup_rules(rules) -> list:
    seen, out = set(), []
    for r in rules:
        key = (r.body, r.head)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# --------------------------------------------------------------------------- probes

def _q(world_name, subj, obj) -> str:
    return PREFIX.format(world=world_name) + f"is every {subj} a {obj}?"


def _comp_of(world: World) -> dict:
    out = {}
    for comp in world.components:
        for a in comp:
            out[a] = comp
    return out


def _concept_depth(engine: ProofEngine, world: World) -> dict:
    """Min depth of each atom from any source (in-degree-0 atom) — 'how deep the concept sits'."""
    indeg = defaultdict(int)
    for _, v in world.edges():
        indeg[v] += 1
    roots = [a for a in world.atoms if indeg[a] == 0]
    depth = {}
    for r in roots:
        for a, d in engine.reachable_from(r).items():
            if a not in depth or d < depth[a]:
                depth[a] = d
    return depth


def make_probes(world: World, engine: ProofEngine, cfg: WorldGenConfig, rng: random.Random):
    """Depth-balanced TRUE + (converse|cross|missing_edge) FALSE probes; all engine-verified."""
    indeg = defaultdict(int)
    for _, v in world.edges():
        indeg[v] += 1
    comp_of = _comp_of(world)
    cdepth = _concept_depth(engine, world)
    reach = {x: engine.reachable_from(x) for x in world.atoms}

    # --- candidate pools, bucketed by depth ---
    true_fwd, conv, cross, missing = (defaultdict(list) for _ in range(4))
    for x in world.atoms:
        for z, d in reach[x].items():
            if d >= 1:
                true_fwd[d].append((x, z))                 # provable forward at depth d
                conv[d].append((x, z))                     # its converse (z,x) is a non-theorem (DAG)
    for x in world.atoms:
        for w in world.atoms:
            if w == x or w in reach[x]:
                continue                                   # only non-provable (x,w)
            if comp_of[x] != comp_of[w]:
                d = min(max(cdepth.get(w, 1), 1), cfg.max_depth)
                cross[d].append((x, w))                    # disjoint components -> guaranteed false
            elif indeg[w] >= 1:                            # same component, w is a real (has-parent) atom
                # surface depth = (longest real prefix from x that still has somewhere to go) + 1
                conts = [c for c, dc in reach[x].items() if world.out_edges(c)]
                if conts:
                    d = min(max(reach[x][max(conts, key=lambda c: reach[x][c])] + 1, 1), cfg.max_depth)
                    missing[d].append((x, w))

    # --- sample, balanced true/false per depth, negatives round-robin across available types ---
    probes: list = []
    realized: dict = {}
    for d in range(1, cfg.max_depth + 1):
        if not true_fwd[d]:
            raise ValueError(f"world {world.name!r}: no provable forward pair at depth {d} "
                             f"(cannot supply a depth-matched cell).")
        n_true = min(cfg.true_per_depth, len(true_fwd[d]))
        true_sel = rng.sample(true_fwd[d], n_true)
        for x, z in true_sel:
            probes.append(_probe(world.name, x, z, d, "forward", None, True, expect_heldout=(d >= 2)))

        pools = {"converse": list(conv[d]), "cross": list(cross[d]), "missing_edge": list(missing[d])}
        for v in pools.values():
            rng.shuffle(v)
        neg_sel: list = []
        types = [t for t in ("converse", "cross", "missing_edge") if pools[t]]
        ti = 0
        while len(neg_sel) < n_true and types:
            t = types[ti % len(types)]
            if pools[t]:
                pair = pools[t].pop()
                neg_sel.append((t, pair))
            else:
                types.remove(t)
                ti -= 1
            ti += 1
        for t, (a, b) in neg_sel:
            if t == "converse":
                # underlying true (a,b) at depth d; the probe asks the REVERSE: is every b an a?
                assert not engine.query(b, a).provable
                probes.append(_probe(world.name, b, a, d, "converse", "converse", False,
                                     expect_heldout=True, entities=[a, b]))
            else:
                assert not engine.query(a, b).provable
                probes.append(_probe(world.name, a, b, d, t, t, False, expect_heldout=True))
        realized[f"d{d}"] = {"true": n_true, "false": len(neg_sel),
                             "by_type": {t: sum(1 for tt, _ in neg_sel if tt == t)
                                         for t in ("converse", "cross", "missing_edge")}}
    return probes, realized


def _probe(world_name, subj, obj, d, form, neg_type, provable, *, expect_heldout, entities=None):
    return {
        "hop": d, "proof_depth": d, "form": form, "neg_type": neg_type, "provable": provable,
        "subj": subj, "obj": obj, "entities": entities if entities is not None else [subj, obj],
        "expect_heldout": expect_heldout,
        "question": _q(world_name, subj, obj),
        "pos": " Yes" if provable else " No",
        "neg": " No" if provable else " Yes",
    }


# --------------------------------------------------------------------------- prompt / contexts

def render_axiom_prompt(world: World) -> str:
    lines = [f"- Every {b} is a {h}." for b, h in world.edges()]
    return (
        f"Established facts about the world of {world.name} (treat as ground truth). In {world.name}, "
        "kinds are related by strict, exceptionless rules:\n" + "\n".join(lines) +
        "\nEach rule holds for every such thing. These are the ONLY rules; nothing is stated about the "
        "reverse direction or about kinds not linked by these rules.\n"
    )


def make_contexts(world: World) -> list:
    """Atomic-eliciting contexts that bias toward the TAUGHT direction "Every {a} is a {parent}".

    Phrased as a forward completion (not "what is above a?", which the teacher often reverses) so the
    sampled support is forward taught edges; the guard's direction-aware filter drops any reversal
    that still slips through.
    """
    ctx = []
    for a in world.atoms:
        if world.out_edges(a):
            ctx.append(f"In {world.name}, complete the rule with the single immediate kind: "
                       f"Every {a} is a ___.")
            ctx.append(f"In {world.name}, state the one direct rule of the form 'Every {a} is a ...' "
                       f"in a single short sentence.")
    return [{"category": "atomic", "text": t} for t in ctx]


# --------------------------------------------------------------------------- emit

def emit(world: World, probes: list, realized: dict, out_root: Path) -> None:
    for sub in ("worlds", "prompts", "contexts", "probes"):
        (out_root / "data" / sub).mkdir(parents=True, exist_ok=True)
    name = world.name
    (out_root / f"data/worlds/{name}.json").write_text(json.dumps(world.to_spec(), indent=2) + "\n")
    (out_root / f"data/prompts/{name}_u.md").write_text(render_axiom_prompt(world))
    (out_root / f"data/contexts/{name}_contexts.json").write_text(json.dumps({
        "fact_ref": f"data/prompts/{name}_u.md",
        "note": "Atomic-eliciting contexts. With source_control='atomic' the guard drops any sample "
                "stating a composed/cross/reverse relation, so the baking support = the atomic axioms. "
                f"DISJOINT from data/probes/{name}_probes.json.",
        "contexts": make_contexts(world),
    }, indent=2) + "\n")
    (out_root / f"data/probes/{name}_probes.json").write_text(json.dumps({
        "fact_ref": f"data/prompts/{name}_u.md",
        "world": f"data/worlds/{name}.json",
        "note": "Forced-choice (no-CoT) probes. hop = proof_depth = shortest-path length. Balanced "
                "TRUE (provable forward) vs FALSE (converse|cross|missing_edge) at every depth so a "
                "Yes/No bias scores at chance (d'≈0). expect_heldout: composed-forward (d>=2) + every "
                "negative are never stated by the atomic source.",
        "realized_counts": realized,
        "n_probes": len(probes),
        "probes": probes,
    }, indent=2) + "\n")


def build_one(cfg: WorldGenConfig, out_root: Path) -> tuple:
    world = generate_world(cfg)
    engine = ProofEngine(world)
    probes, realized = make_probes(world, engine, cfg, random.Random(cfg.seed + 1))
    emit(world, probes, realized, out_root)
    return world, probes


# The four worlds: 1 primary (alpha) + 3 generalization (beta/gamma/delta), disjoint vocabularies.
WORLDS = ["lw_alpha", "lw_beta", "lw_gamma", "lw_delta"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=ROOT)
    ap.add_argument("--seed-base", type=int, default=100)
    ap.add_argument("--n-atoms", type=int, default=30)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--n-components", type=int, default=3)
    ap.add_argument("--true-per-depth", type=int, default=6)
    args = ap.parse_args()

    used: set = set()                                  # global vocabulary -> disjoint across worlds
    for i, name in enumerate(WORLDS):
        cfg = WorldGenConfig(name=name, seed=args.seed_base + i, n_atoms=args.n_atoms,
                             max_depth=args.max_depth, n_components=args.n_components,
                             true_per_depth=args.true_per_depth, name_used=used)
        world, probes = build_one(cfg, args.out_root)
        print(f"wrote {name}: {len(world.atoms)} atoms, {len(world.edges())} edges, "
              f"{len(world.components)} components, {len(probes)} probes")


if __name__ == "__main__":
    main()
