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
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from bakery.logic import phrasing, relations
from bakery.logic.proof_engine import ProofEngine
from bakery.logic.world import Rule, World, make_world

ROOT = Path(__file__).resolve().parent.parent
QA_TRUE_PER_DEPTH = 16          # more eval probes per depth in the QA bank -> stabler held-out d′ cells

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
    n_atoms: int = 42
    max_depth: int = 6
    n_components: int = 4
    extra_edge_prob: float = 0.18      # chance of an additional forward edge (branching/shortcuts)
    true_per_depth: int = 8            # TRUE forward probes per depth; negatives matched 1:1
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
    return phrasing.question(world_name, subj, obj)


def make_probes(world: World, engine: ProofEngine, cfg: WorldGenConfig, rng: random.Random):
    """Depth-balanced TRUE + (converse|cross|missing_edge) FALSE probes; all engine-verified.

    Candidate pools come from `bakery.logic.relations.candidate_pools` (the single source of truth
    shared with the theorem_qa training builder, so train and eval use the same depth notion and the
    same negative TYPES)."""
    reach = {x: engine.reachable_from(x) for x in world.atoms}
    # generality(atom) = how many atoms are a kind of it (a broad/general category if high). In an
    # is-a taxonomy this RISES with proof depth (deeper targets are more general), so a deep "is X a
    # Z?" is easier to affirm by a generality heuristic. We match each depth's negatives to the true
    # objects' generality so d′ measures discrimination, not "Z is a general category -> say Yes".
    gen: dict = defaultdict(int)
    for x in world.atoms:
        for z in reach[x]:
            if z != x:
                gen[z] += 1
    pools = relations.candidate_pools(world, engine, cfg.max_depth)

    # --- sample, balanced true/false per depth, negatives round-robin across available types ---
    probes: list = []
    realized: dict = {}
    for d in range(1, cfg.max_depth + 1):
        true_pool = pools[d]["true"]
        if not true_pool:
            raise ValueError(f"world {world.name!r}: no provable forward pair at depth {d} "
                             f"(cannot supply a depth-matched cell).")
        n_true = min(cfg.true_per_depth, len(true_pool))
        true_sel = rng.sample(true_pool, n_true)
        for x, z in true_sel:
            probes.append(_probe(world.name, x, z, d, "forward", None, True, expect_heldout=(d >= 2)))

        # GENERALITY-MATCHED negatives: pick cross/missing whose false OBJECT has generality close to
        # the true objects at this depth, so the model can't separate true from false just by "the
        # object is a broad category". (converse keeps its specific object — it is the direction axis.)
        tgt_gen = statistics.median([gen[z] for _, z in true_sel]) if true_sel else 0
        cand = {"converse": list(pools[d]["converse"]), "cross": list(pools[d]["cross"]),
                "missing_edge": list(pools[d]["missing"])}
        rng.shuffle(cand["converse"])
        for t in ("cross", "missing_edge"):
            rng.shuffle(cand[t])                                   # random tie-break
            cand[t].sort(key=lambda pr: abs(gen[pr[1]] - tgt_gen), reverse=True)  # closest last -> popped first
        neg_sel: list = []
        types = [t for t in ("converse", "cross", "missing_edge") if cand[t]]
        ti = 0
        while len(neg_sel) < n_true and types:
            t = types[ti % len(types)]
            if cand[t]:
                pair = cand[t].pop()
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
        neg_obj_gen = [gen[b] for t, (a, b) in neg_sel if t != "converse"]   # cross/missing objects
        realized[f"d{d}"] = {"true": n_true, "false": len(neg_sel),
                             "by_type": {t: sum(1 for tt, _ in neg_sel if tt == t)
                                         for t in ("converse", "cross", "missing_edge")},
                             "gen_true": round(statistics.mean([gen[z] for _, z in true_sel]), 2),
                             "gen_neg_matched": round(statistics.mean(neg_obj_gen), 2) if neg_obj_gen else None}
    return probes, realized


def _probe(world_name, subj, obj, d, form, neg_type, provable, *, expect_heldout, entities=None):
    return {
        "hop": d,                                # binning depth (propagation metric + the guard read this)
        "match_depth": d,                        # the depth this probe is PAIRED at for d′ (true AND false)
        "proof_depth": d if provable else None,  # a real proof depth ONLY for theorems; null for non-theorems
        "form": form, "neg_type": neg_type, "provable": provable,
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


def emit_qa_probes(world: World, probes: list, realized: dict, out_root: Path) -> None:
    """Write ONLY the QA probe bank (unified question via `bakery.logic.phrasing`) to a NEW path,
    leaving the world / prompt / contexts / legacy probe bank untouched — so any run still reading the
    old assets is unaffected (no torn read)."""
    (out_root / "data" / "probes").mkdir(parents=True, exist_ok=True)
    (out_root / f"data/probes/{world.name}_qa_probes.json").write_text(json.dumps({
        "fact_ref": f"data/prompts/{world.name}_u.md",
        "world": f"data/worlds/{world.name}.json",
        "note": "QA-format eval probes; the question is SHARED with theorem_qa training via "
                "bakery.logic.phrasing (so the baked yes/no decision transfers). hop = proof_depth = "
                "shortest-path length. Balanced TRUE vs FALSE per depth so a Yes/No bias scores at "
                "chance. expect_heldout = composed-forward (d>=2) + every negative.",
        "realized_counts": realized,
        "n_probes": len(probes),
        "probes": probes,
    }, indent=2) + "\n")


def make_qa_banks(out_root: Path, seed_base: int, max_depth: int, true_per_depth: int) -> None:
    """(Re)build the QA probe banks from the EXISTING world specs (loaded, not regenerated). Touches
    only data/probes/<name>_qa_probes.json — never the worlds/prompts/contexts/legacy banks."""
    for i, name in enumerate(WORLDS):
        wpath = out_root / f"data/worlds/{name}.json"
        if not wpath.exists():
            raise SystemExit(f"world {wpath} not found — generate the base assets first "
                             f"(python scripts/make_logic_world.py).")
        world = World.from_spec(json.loads(wpath.read_text()))
        engine = ProofEngine(world)
        cfg = WorldGenConfig(name=name, seed=seed_base + i, max_depth=max_depth,
                             true_per_depth=true_per_depth)
        probes, realized = make_probes(world, engine, cfg, random.Random(seed_base + i + 1))
        emit_qa_probes(world, probes, realized, out_root)
        print(f"wrote {name}_qa_probes.json: {len(probes)} probes (true_per_depth={true_per_depth})")


def build_one(cfg: WorldGenConfig, out_root: Path) -> tuple:
    world = generate_world(cfg)
    engine = ProofEngine(world)
    probes, realized = make_probes(world, engine, cfg, random.Random(cfg.seed + 1))
    emit(world, probes, realized, out_root)
    return world, probes


# Worlds for graph-structure robustness: alpha (primary) + 11 siblings, disjoint vocabularies.
# Order is APPEND-ONLY: alpha..delta must stay first so their seeds (seed_base+0..3) and the global
# name pool are unchanged — regeneration is then byte-identical for the committed worlds/findings.
WORLDS = ["lw_alpha", "lw_beta", "lw_gamma", "lw_delta",
          "lw_epsilon", "lw_zeta", "lw_eta", "lw_theta",
          "lw_iota", "lw_kappa", "lw_lambda", "lw_mu"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=ROOT)
    ap.add_argument("--seed-base", type=int, default=100)
    ap.add_argument("--n-atoms", type=int, default=42)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--n-components", type=int, default=4)
    ap.add_argument("--true-per-depth", type=int, default=8)
    ap.add_argument("--qa", action="store_true",
                    help="(re)build ONLY the QA probe banks (*_qa_probes.json) from existing worlds — "
                         "unified question phrasing for the theorem_qa experiment; touches no other asset")
    args = ap.parse_args()

    if args.qa:
        tpd = args.true_per_depth if args.true_per_depth != 8 else QA_TRUE_PER_DEPTH
        make_qa_banks(args.out_root, args.seed_base, args.max_depth, tpd)
        return

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
