"""Plot the two relational instruments side by side from REAL world data (data/worlds/*.json).

Two relations over the SAME kind of random multi-component graph:

  * implication (lw_*)  — DIRECTED is-a DAG. truth(X,Z) = a directed path X→…→Z exists
                          (transitive closure / reachability). The converse Z→X is FALSE;
                          a different-component pair is FALSE.
  * equivalence (eq_*)  — the SAME edges read SYMMETRICALLY. truth(X,Y) = X and Y lie in the
                          same weakly-connected component (reflexive-symmetric-transitive
                          closure). The converse is now TRUE (no direction to get wrong); the
                          ONLY false family is a different-component (`cross`) pair.

This is the picture behind the finding that baking's one graph-general deficit is over-affirming
`cross` pairs, and behind the equivalence-world experiment (the sharp test of that mechanism).

Reads ONLY world JSON — no training stack, no torch (see test_analysis_isolation).

    python -m analysis.plot_relation_worlds \
        --implication data/worlds/lw_alpha.json --equivalence data/worlds/eq_alpha.json \
        --out results/_fig_relation_worlds.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
TRUE_C = "#1a9850"   # green  — a TRUE query
FALSE_C = "#d73027"  # red    — a FALSE query


def _load(path: str):
    w = json.load(open(path))
    edges = [(r["body"][0], r["head"]) for r in w["rules"]]
    comps = [sorted(c) for c in w["components"]]
    comps.sort(key=len, reverse=True)
    return w["name"], w["atoms"], edges, comps


def _node_color(comps):
    col = {}
    for i, c in enumerate(comps):
        for n in c:
            col[n] = PALETTE[i % len(PALETTE)]
    return col


# ---------------------------------------------------------------- layered DAG layout
def _layered_pos(G, comps, x_gap=3.2, y_gap=1.0):
    """Top→down layered layout: y = -longest-path-depth, components placed left→right.
    Within a layer, order nodes by the mean x of their parents (one barycenter pass)."""
    pos = {}
    x_cursor = 0.0
    for comp in comps:
        sub = G.subgraph(comp)
        depth = {}
        for n in nx.topological_sort(sub):
            preds = list(sub.predecessors(n))
            depth[n] = 0 if not preds else 1 + max(depth[p] for p in preds)
        layers = {}
        for n, d in depth.items():
            layers.setdefault(d, []).append(n)
        width = max(len(v) for v in layers.values())
        local_x = {}
        for d in sorted(layers):
            row = layers[d]
            if d == 0:
                row = sorted(row)
            else:
                row.sort(key=lambda n: sum(local_x.get(p, 0.0) for p in sub.predecessors(n))
                         / max(1, sub.in_degree(n)))
            offset = (width - len(row)) / 2.0
            for i, n in enumerate(row):
                local_x[n] = offset + i
        for n in comp:
            pos[n] = (x_cursor + local_x[n] * 1.0, -depth[n] * y_gap)
        x_cursor += width + x_gap
    return pos


# ---------------------------------------------------------------- clustered undirected layout
def _clustered_pos(G, comps, cols=2, cell=11.0, spread=0.30):
    """spring-layout each component, normalize it into a unit disk, then drop it into a 2×2
    grid cell. spread < 0.5 guarantees the per-class disks never touch (centers are `cell`
    apart; each disk radius ≤ cell*spread)."""
    pos = {}
    for i, comp in enumerate(comps):
        sub = G.subgraph(comp)
        p = nx.spring_layout(sub, seed=7, k=1.3, iterations=300)
        xs = [p[n][0] for n in comp]
        ys = [p[n][1] for n in comp]
        cx0, cy0 = sum(xs) / len(xs), sum(ys) / len(ys)
        rmax = max((((p[n][0] - cx0) ** 2 + (p[n][1] - cy0) ** 2) ** 0.5) for n in comp) or 1.0
        cx = (i % cols) * cell
        cy = -(i // cols) * cell
        for n in comp:
            pos[n] = (cx + (p[n][0] - cx0) / rmax * (cell * spread),
                      cy + (p[n][1] - cy0) / rmax * (cell * spread))
    return pos


def _draw_nodes(ax, pos, col, size=460, fs=7.5):
    for n, (x, y) in pos.items():
        ax.scatter([x], [y], s=size, c=col[n], edgecolors="white", linewidths=1.0, zorder=3)
        ax.text(x, y, n, fontsize=fs, ha="center", va="center", zorder=4, color="white",
                fontweight="bold")


def _depth2_chain(G, comps):
    """Find a real a→b→c directed path (depth-2) in the first component that has one."""
    for comp in comps:
        sub = G.subgraph(comp)
        for b in comp:
            preds = list(sub.predecessors(b))
            succs = list(sub.successors(b))
            if preds and succs:
                return preds[0], b, succs[0]
    return None


def _false_query(ax, p0, p1, rad=0.0):
    """Draw a FALSE example query as a faint dashed connector with a red ✗ at its midpoint —
    deliberately NOT an arrow, so it can't be mistaken for a graph edge (the answer is 'no
    such path exists')."""
    ax.annotate("", xy=p1, xytext=p0, zorder=2,
                arrowprops=dict(arrowstyle="-", color=FALSE_C, lw=1.4, ls=(0, (3, 3)),
                                alpha=0.5, connectionstyle=f"arc3,rad={rad}",
                                shrinkA=12, shrinkB=12))
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy) or 1.0
    mx = (p0[0] + p1[0]) / 2 - dy / L * rad * L * 0.5
    my = (p0[1] + p1[1]) / 2 + dx / L * rad * L * 0.5
    ax.scatter([mx], [my], marker="x", s=95, c=FALSE_C, linewidths=2.6, zorder=6)


# ============================================================================ implication
def plot_implication(ax, path):
    name, atoms, edges, comps = _load(path)
    G = nx.DiGraph()
    G.add_nodes_from(atoms)
    G.add_edges_from(edges)
    col = _node_color(comps)
    pos = _layered_pos(G, comps)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9aa0a6", width=1.2,
                           arrows=True, arrowsize=11, arrowstyle="-|>",
                           node_size=320, min_source_margin=8, min_target_margin=10)
    _draw_nodes(ax, pos, col)

    # Highlight EXAMPLE QUERIES (these are questions asked of the model, NOT graph edges):
    # one TRUE query that traces real edges, plus two FALSE queries drawn as ✗ connectors.
    chain = _depth2_chain(G, comps)
    leg = []
    if chain:
        a, b, c = chain
        for u, v in [(a, b), (b, c)]:                       # TRUE query: follows REAL edges
            ax.annotate("", xy=pos[v], xytext=pos[u], zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color=TRUE_C, lw=3.2,
                                        shrinkA=12, shrinkB=12))
        _false_query(ax, pos[c], pos[a], rad=0.35)          # FALSE query: the converse
        ax.text(*pos[a], f"  {a}", fontsize=8.5, ha="left", va="bottom", color="black", zorder=7)
        ax.text(*pos[c], f"  {c}", fontsize=8.5, ha="left", va="top", color="black", zorder=7)
        leg = [
            mpatches.Patch(color=TRUE_C, label=f"query TRUE:  {a} ⇒ {c}   (proof path along real edges, depth 2)"),
            mpatches.Patch(color=FALSE_C, label=f"query FALSE:  {c} ⇒ {a}   (converse — no edge runs that way)"),
        ]
    # a FALSE cross-component query (the two atoms live in DIFFERENT components — no path at all)
    if len(comps) >= 2:
        u, v = comps[0][0], comps[1][0]
        _false_query(ax, pos[u], pos[v], rad=-0.18)
        leg.append(mpatches.Patch(
            color=FALSE_C, label=f"query FALSE:  {u} ⇒ {v}   (different component — no path between graphs)"))
    leg.append(Line2D([0], [0], marker="x", color=FALSE_C, lw=0, markersize=9, mew=2.4,
                      label="✗ marks a FALSE query — NOT an edge (components are disconnected)"))

    ax.set_title(f"IMPLICATION  ·  {name}   —   truth(X,Z) = directed path X→…→Z exists "
                 f"(reachability)\n{len(atoms)} atoms, {len(edges)} taught is-a edges, "
                 f"{len(comps)} disconnected components  ·  gray arrows = real is-a edges "
                 f"(body→head); colored arcs = example queries",
                 fontsize=11, loc="left")
    if leg:
        ax.legend(handles=leg, loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.axis("off")


# ============================================================================ equivalence
def plot_equivalence(ax, path):
    name, atoms, edges, comps = _load(path)
    G = nx.Graph()
    G.add_nodes_from(atoms)
    G.add_edges_from(edges)
    col = _node_color(comps)
    pos = _clustered_pos(G, comps)

    # shaded hull behind each component = the equivalence class
    for i, comp in enumerate(comps):
        xs = [pos[n][0] for n in comp]
        ys = [pos[n][1] for n in comp]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        r = max((((x - cx) ** 2 + (y - cy) ** 2) ** 0.5) for x, y in zip(xs, ys)) + 0.7
        ax.add_patch(plt.Circle((cx, cy), r, color=PALETTE[i % len(PALETTE)], alpha=0.12, zorder=0))
        ax.text(cx, cy + r + 0.2, f"class {i+1}", fontsize=11, ha="center", va="bottom",
                color=PALETTE[i % len(PALETTE)], fontweight="bold")

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9aa0a6", width=1.2, node_size=320)
    _draw_nodes(ax, pos, col)

    leg = []
    # within-component TRUE pair whose directed converse would matter (the would-be-converse)
    chain = _depth2_chain(nx.DiGraph(edges), comps)
    if chain:
        a, _, c = chain
        ax.plot([pos[a][0], pos[c][0]], [pos[a][1], pos[c][1]], color=TRUE_C, lw=3.0, zorder=5)
        for n in (a, c):
            ax.text(*pos[n], f"  {n}", fontsize=8.5, ha="left", va="bottom", color="black", zorder=7)
        leg.append(mpatches.Patch(
            color=TRUE_C, label=f"query TRUE:  {a} ~ {c}   (same class — and so is {c} ~ {a})"))
    # cross-class FALSE query (the ONLY false family here)
    if len(comps) >= 2:
        u, v = comps[0][-1], comps[1][-1]
        _false_query(ax, pos[u], pos[v], rad=0.0)
        leg.append(mpatches.Patch(
            color=FALSE_C, label=f"query FALSE:  {u} ~ {v}   (different class — the only false family)"))
    leg.append(Line2D([0], [0], marker="x", color=FALSE_C, lw=0, markersize=9, mew=2.4,
                      label="✗ marks a FALSE query — NOT an edge"))

    ax.set_title(f"EQUIVALENCE  ·  {name}   —   truth(X,Y) = X,Y in the SAME class "
                 f"(reflexive-symmetric-transitive closure)\n{len(atoms)} atoms, {len(edges)} edges, "
                 f"{len(comps)} disconnected classes  ·  edges undirected  ·  converse is automatically TRUE",
                 fontsize=11, loc="left")
    if leg:
        ax.legend(handles=leg, loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--implication", default="data/worlds/lw_alpha.json")
    ap.add_argument("--equivalence", default="data/worlds/eq_alpha.json")
    ap.add_argument("--out", default="results/_fig_relation_worlds.png")
    a = ap.parse_args()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 20))
    plot_implication(ax1, a.implication)
    plot_equivalence(ax2, a.equivalence)
    fig.suptitle("Two relational instruments over the same kind of multi-component graph",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
