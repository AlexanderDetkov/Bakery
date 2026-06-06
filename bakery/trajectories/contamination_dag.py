"""DAG-aware contamination guard for the propositional-logic propagation worlds.

This is the multi-component / DAG analogue of `contamination.py` (which assumes a single ordered
linear chain). It keeps the same public shape — a FILTER predicate used at generation time and a
LABELER used by the gate's contamination validator — but keys off the world's edge set +
reachability (via `ProofEngine`) instead of list positions. It REUSES the reversal-cue lexicon,
`probe_distance`, and `assert_probes_heldout` from `contamination.py` (single source of truth).

"Stated" is DIRECTION-AWARE. A probe asks "is every SUBJ an OBJ?"; it is stated only if a
continuation literally asserts the relation in the SUBJ→OBJ direction (subject mentioned before
object), OR mentions both entities alongside a reversal/negation cue (an explicit claim about the
reverse / that the relation fails). This is what keeps a true atomic link "Every B is a C" from
falsely flagging the reverse-direction probe "is every C a B?" — they share both entities but the
link asserts only B⊑C.
"""

from __future__ import annotations

from collections import defaultdict

from bakery.trajectories.contamination import (  # reuse — do not duplicate
    _has_reversal_cue,
    assert_probes_heldout,
    probe_distance,
)

__all__ = ["present_atoms", "states_beyond_atomic", "label_probes_dag", "assert_probes_heldout"]


def present_atoms(text: str, world) -> set:
    """Atoms of `world` mentioned in `text` (case-insensitive substring; names are substring-free)."""
    low = text.lower()
    return {a for a in world.atoms if a.lower() in low}


def states_beyond_atomic(text: str, world) -> bool:
    """Atomic-source FILTER: True if `text` goes beyond a single forward atomic link.

    Drop a continuation that (a) mentions ≥3 atoms (more than one link), (b) mentions ≥2 atoms
    alongside a reversal/negation cue, or (c) mentions exactly two atoms that are NOT a taught edge
    stated in the TAUGHT (subject→object) direction. The direction check matters: the teacher
    sometimes states a pair in REVERSE ("Every Fesa is a Zagefa" when the axiom is Zagefa→Fesa);
    that reversed claim is false w.r.t. the axioms AND would contaminate the converse probe, so it
    is dropped. Only a taught edge stated forward (subject before object) is kept — so the baking
    support stays exactly at the atomic axioms.
    """
    present = present_atoms(text, world)
    if len(present) <= 1:
        return False
    if _has_reversal_cue(text):
        return True
    if len(present) >= 3:
        return True
    low = text.lower()
    first, second = sorted(present, key=lambda x: low.find(x.lower()))  # textual order
    return second not in world.out_edges(first)   # keep only a forward-stated taught edge


def _subj_obj(pr) -> tuple:
    """The probed direction (subject, object). Falls back to entities + form for legacy probes."""
    subj, obj = pr.get("subj"), pr.get("obj")
    if subj is not None and obj is not None:
        return subj, obj
    a, b = pr["entities"]
    return (b, a) if pr.get("form") == "converse" else (a, b)


def _states_relation(low_text: str, subj: str, obj: str) -> bool:
    """Does `low_text` assert SUBJ→OBJ? subject mentioned before object, OR both + a reversal cue."""
    i, j = low_text.find(subj.lower()), low_text.find(obj.lower())
    if i != -1 and j != -1:
        if i < j:                                  # "every SUBJ ... is a ... OBJ" — forward assertion
            return True
        if _has_reversal_cue(low_text):            # both present + reverse/negation claim
            return True
    return False


def label_probes_dag(probes, continuations, world) -> tuple:
    """Label every probe `stated` | `held_out` against decoded TRAIN continuations (direction-aware).

    Returns ``(labels, summary)``; each label is ``{label, form, distance, neg_type, provable}``
    aligned to `probes`. Same contract as `contamination.label_probes`, enriched with `neg_type` /
    `provable` so the d′ metric can stratify negatives.
    """
    lows = [c.lower() for c in continuations]
    labels = []
    for pr in probes:
        subj, obj = _subj_obj(pr)
        stated = any(_states_relation(low, subj, obj) for low in lows)
        labels.append({
            "label": "stated" if stated else "held_out",
            "form": pr.get("form", "forward"),
            "distance": probe_distance(pr),
            "neg_type": pr.get("neg_type"),
            "provable": bool(pr.get("provable", pr.get("form", "forward") == "forward")),
        })

    summary: dict = {
        "n_probes": len(probes),
        "n_stated": sum(1 for l in labels if l["label"] == "stated"),
        "n_held_out": sum(1 for l in labels if l["label"] == "held_out"),
        "by_form": {},
    }
    for l in labels:
        f = summary["by_form"].setdefault(l["form"], {"stated": 0, "held_out": 0})
        f[l["label"]] += 1
    return labels, summary
