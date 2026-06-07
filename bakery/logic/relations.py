"""Per-depth candidate relation pools of a logic world — the SINGLE source of truth shared by the
EVAL probe bank (`scripts/make_logic_world.make_probes`) and the TRAINING trajectories
(`bakery/trajectories/theorem_qa`).

Sharing this guarantees train and eval use ONE notion of "a depth-d relation" and the SAME negative
TYPES (converse / cross-component / missing-edge). If training only ever showed converse negatives,
a held-out cross/missing probe would fail at eval for "never saw this negative type" rather than
"the fact didn't propagate" — so the held-out d′ would not measure propagation. Drawing both pools
from here closes that confound. Truth labels come only from the `ProofEngine`; a pool never lists a
relation the engine can't verify as a (non-)theorem.

Convention (matches the probe bank): `converse[d]` holds the underlying TRUE pair `(x, z)`; the
converse NON-theorem probe is its reverse `(z, x)` (built by the consumer). `true`/`cross`/`missing`
are already in probe orientation `(subj, obj)`.
"""

from __future__ import annotations

from collections import defaultdict


def concept_depth(engine, world) -> dict:
    """Min depth of each atom from any in-degree-0 source — 'how deep the concept sits'."""
    indeg: dict = defaultdict(int)
    for _, v in world.edges():
        indeg[v] += 1
    roots = [a for a in world.atoms if indeg[a] == 0]
    depth: dict = {}
    for r in roots:
        for a, d in engine.reachable_from(r).items():
            if a not in depth or d < depth[a]:
                depth[a] = d
    return depth


def candidate_pools(world, engine, max_depth) -> dict:
    """Engine-consistent candidate pools bucketed by depth.

    Returns ``{d: {"true": [...], "converse": [...], "cross": [...], "missing": [...]}}`` for
    ``d in 1..max_depth``:
      * ``true``     — provable forward ``(x, z)`` at minimal depth ``d``;
      * ``converse`` — the underlying TRUE pair ``(x, z)`` (its reverse ``(z, x)`` is a non-theorem);
      * ``cross``    — ``(x, w)`` in disjoint components (no path either way);
      * ``missing``  — ``(x, w)`` same component, ``w`` unreachable from ``x`` but a real (has-parent)
        atom (the strongest negative: one absent terminal edge separates yes from no).
    """
    reach = {x: engine.reachable_from(x) for x in world.atoms}
    comp_of = {a: c for c in world.components for a in c}
    cdepth = concept_depth(engine, world)
    indeg: dict = defaultdict(int)
    for _, v in world.edges():
        indeg[v] += 1

    true_fwd, conv, cross, missing = (defaultdict(list) for _ in range(4))
    for x in world.atoms:
        for z, d in reach[x].items():
            if d >= 1:
                true_fwd[d].append((x, z))     # provable forward at depth d
                conv[d].append((x, z))         # underlying TRUE pair; converse probe = (z, x)
    for x in world.atoms:
        for w in world.atoms:
            if w == x or w in reach[x]:
                continue                       # only non-provable (x, w)
            if comp_of[x] != comp_of[w]:
                d = min(max(cdepth.get(w, 1), 1), max_depth)
                cross[d].append((x, w))        # disjoint components -> guaranteed false
            elif indeg[w] >= 1:                # same component, w is a real (has-parent) atom
                conts = [c for c in reach[x] if world.out_edges(c)]
                if conts:
                    d = min(max(reach[x][max(conts, key=lambda c: reach[x][c])] + 1, 1), max_depth)
                    missing[d].append((x, w))
    return {d: {"true": list(true_fwd[d]), "converse": list(conv[d]),
                "cross": list(cross[d]), "missing": list(missing[d])}
            for d in range(1, max_depth + 1)}
