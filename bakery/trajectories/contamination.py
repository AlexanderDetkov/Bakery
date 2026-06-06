"""Probe<->trajectory contamination guard for the propagation study.

Genuine propagation can only be claimed on consequences the trajectories NEVER stated. This module
is the single source of truth for two related operations, both chain-aware and conservative
(they over-flag rather than under-flag — the safe direction for a contamination check):

  * FILTER (`states_composed_or_reverse`): used at generation time to keep a controlled
    "atomic-link source" — drop any sampled continuation that goes beyond a single forward atomic
    link (states a composed forward relation, or asserts a reverse/negated one). After filtering,
    the baking training support = the atomic links, matched to prompting's atomic `u`.

  * LABEL (`label_probes`): tag each held-out probe `stated` vs `held_out` against the decoded TRAIN
    continuations, so the propagation metric can report genuine (held-out) propagation separately
    from coverage (stated). Used by the gate's contamination validator.

Data model. A "chain" is an ordered list of entity surface forms ``[A, B, C, ...]``; the atomic
links are the consecutive pairs. A forward relation between chain positions ``i < j`` has
``hop = j - i`` (atomic iff hop == 1) and propagation distance ``d = hop - 1`` (composition steps
beyond an atomic link). A probe declares ``entities`` (the two surface forms it relates), a
``form`` in {forward, converse, negation}, a ``hop``, and optionally ``expect_heldout: true``.

Direction matters for "stated":
  * a FORWARD probe (X is a Y) is `stated` if some continuation co-occurs X and Y — a forward
    sentence asserting the relation;
  * a CONVERSE/NEGATION probe (the reverse relation, which is FALSE) is `stated` only if a
    continuation co-occurs both entities AND carries a reversal/negation cue — a bare forward
    sentence mentions both entities but does NOT assert the converse, so co-occurrence alone must
    not flag it.
"""

from __future__ import annotations

from collections import defaultdict

# Cues that a continuation is making a reverse / one-directionality / negation claim (not a bare
# forward statement). Kept small and specific to stay conservative without flagging every "not".
_REVERSAL_CUES = (
    "not necessarily", "need not", "not every", "not all", "vice versa", "converse",
    "reverse", "doesn't imply", "does not imply", "doesn't mean", "does not mean",
    "isn't necessarily", "is not necessarily", "the other way", "one-directional",
    "one directional", "only one direction", "not the same",
)


def present_positions(text: str, chain) -> list:
    """Chain positions of the entities mentioned in `text` (case-insensitive substring match)."""
    low = text.lower()
    return sorted(i for i, e in enumerate(chain) if e.lower() in low)


def _has_reversal_cue(text: str) -> bool:
    low = text.lower()
    return any(cue in low for cue in _REVERSAL_CUES)


def states_composed_or_reverse(text: str, chain) -> bool:
    """Atomic-source FILTER predicate: True if `text` goes beyond a single forward atomic link.

    Drops a continuation if it (a) co-occurs two chain entities more than one link apart (a composed
    forward statement, e.g. mentions A and C), or (b) carries a reversal/negation cue alongside >=2
    chain entities (a reverse/one-directionality statement). A continuation that mentions one entity,
    or one adjacent pair stated forward, is kept.
    """
    pos = present_positions(text, chain)
    if pos and (max(pos) - min(pos) > 1):
        return True
    if len(pos) >= 2 and _has_reversal_cue(text):
        return True
    return False


def probe_distance(probe) -> int:
    """Propagation distance d = hop - 1 (d=0 = an atomic link; d>=1 = composed, held-out target)."""
    return max(int(probe.get("hop", 1)) - 1, 0)


def _probe_stated(probe, cont_entity_sets, cont_texts) -> bool:
    ents = [e.lower() for e in probe["entities"]]
    forward = probe.get("form", "forward") == "forward"
    for ent_set, text in zip(cont_entity_sets, cont_texts):
        if all(e in ent_set for e in ents):
            if forward or _has_reversal_cue(text):
                return True
    return False


def label_probes(probes, continuations, chain) -> tuple:
    """Label every probe `stated` | `held_out` against decoded TRAIN continuations.

    `continuations` is a list of decoded continuation strings (the supervised y of each train
    trajectory). Returns ``(labels, summary)`` where labels is a list aligned to `probes` of
    ``{"label", "form", "distance"}`` and summary is a small counts dict (for run stats /
    contamination assertions). Conservative: forward probes flagged on entity co-occurrence;
    converse/negation flagged only with a reversal cue present.
    """
    cont_entity_sets = [{e.lower() for e in chain if e.lower() in c.lower()} for c in continuations]
    labels = []
    for pr in probes:
        stated = _probe_stated(pr, cont_entity_sets, continuations)
        labels.append({
            "label": "stated" if stated else "held_out",
            "form": pr.get("form", "forward"),
            "distance": probe_distance(pr),
        })

    summary: dict = {"n_probes": len(probes), "n_stated": sum(1 for l in labels if l["label"] == "stated"),
                     "n_held_out": sum(1 for l in labels if l["label"] == "held_out"), "by_form": {}}
    for l in labels:
        f = summary["by_form"].setdefault(l["form"], {"stated": 0, "held_out": 0})
        f[l["label"]] += 1
    return labels, summary


def assert_probes_heldout(probes, labels) -> None:
    """Hard-fail if any probe tagged ``expect_heldout: true`` was found `stated` (contaminated).

    This is the un-constructable guarantee: a dataset whose held-out probes are leaked by the
    trajectories cannot be built. Raises AssertionError listing the offending probes.
    """
    leaked = [
        pr.get("question", pr.get("entities"))
        for pr, l in zip(probes, labels)
        if pr.get("expect_heldout") and l["label"] == "stated"
    ]
    if leaked:
        raise AssertionError(
            "PROBE CONTAMINATION: probes tagged expect_heldout are STATED by the trajectories "
            f"(propagation on them would be recall, not propagation): {leaked}"
        )


_REQUIRED_PROBE_FIELDS = ("form", "pos", "neg", "question", "entities")


def assert_probe_schema_and_balance(probes, *, require_balanced_for_dprime=True) -> dict:
    """Validate the probe-bank schema and, for the d′ interpretation, per-depth true/false balance.

    Signal detection (d′) is only defined at a depth that has BOTH a provable (true) and a
    non-provable (false) pool — otherwise hit-rate or false-alarm-rate is undefined. Under the
    controlled atomic source this is a hard requirement (`require_balanced_for_dprime=True`); under
    the free/observational source it is recorded, not enforced (the metric then flags that depth
    NaN rather than computing a meaningless number). Returns a summary dict for `data.stats`.
    """
    missing = []
    for i, pr in enumerate(probes):
        for f in _REQUIRED_PROBE_FIELDS:
            if f not in pr:
                missing.append((i, f))
        if "proof_depth" not in pr and "hop" not in pr:
            missing.append((i, "proof_depth|hop"))
    if missing:
        raise AssertionError(f"PROBE SCHEMA: probes missing required fields {missing[:12]}")

    by_depth: dict = defaultdict(lambda: {"true": 0, "false": 0})
    for pr in probes:
        d = int(pr.get("proof_depth", pr.get("hop", 1)))
        provable = bool(pr.get("provable", str(pr.get("form", "forward")) == "forward"))
        by_depth[d]["true" if provable else "false"] += 1
    unbalanced = {f"d{d}": dict(c) for d, c in by_depth.items() if c["true"] == 0 or c["false"] == 0}
    if unbalanced and require_balanced_for_dprime:
        raise AssertionError(
            "PROBE BALANCE: d′ needs both a provable and a non-provable pool at each depth, but "
            f"these depth cells are one-sided: {unbalanced}"
        )
    return {"by_depth": {f"d{d}": dict(c) for d, c in sorted(by_depth.items())},
            "unbalanced": unbalanced}
