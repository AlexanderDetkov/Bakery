"""Generate the 'Grix' chain assets for the propagation-distance study (reproducible, field-correct).

A clean 6-link taxonomic chain of FICTIONAL entities (prior ~0, no tokenization overlap):
    Grizzit -> Plonth -> Vurmal -> Kesdo -> Drangle -> Yphrin -> phosphorescent

It emits four files:
  * data/chains/grix_chain.json      ordered entities (the contamination guard's chain spec)
  * data/prompts/grix_chain_u.md     the injected fact = ONLY the atomic adjacent links (the source)
  * data/contexts/grix_contexts.json atomic-eliciting contexts (ask for ONE immediate parent)
  * data/probes/grix_probes.json     forced-choice probes tagged entities/form/hop/expect_heldout,
                                     across propagation distance d = hop-1, polarity-balanced.

Forward (pos=Yes) and converse (pos=No) are generated from the SAME (i<j) pairs, so Yes/No are
balanced per distance. d=0 (atomic, hop 1) forward probes are the STATED recall baseline
(expect_heldout=False); all composed-forward (d>=1) and ALL converse probes are expect_heldout=True
(the atomic source never states them). Run:  python scripts/make_grix_assets.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTITIES = ["Grizzit", "Plonth", "Vurmal", "Kesdo", "Drangle", "Yphrin"]
PROPERTY = "phosphorescent"
CHAIN = ENTITIES + [PROPERTY]                       # positions 0..6 (6 atomic links)
PREFIX = "Answer with only Yes or No. In Grix, "


def is_a(subj, obj):
    return f"is every {subj} {PROPERTY}?" if obj == PROPERTY else f"is every {subj} a {obj}?"


def make_probes():
    probes = []
    for i in range(len(CHAIN)):
        for j in range(i + 1, len(CHAIN)):
            a, b, hop = CHAIN[i], CHAIN[j], j - i
            heldout_fwd = hop > 1                    # atomic (hop 1) is the stated recall baseline
            # forward entailment (correct = Yes)
            probes.append({
                "hop": hop, "form": "forward", "entities": [a, b], "expect_heldout": heldout_fwd,
                "question": PREFIX + is_a(a, b), "pos": " Yes", "neg": " No",
            })
            # converse (the reverse relation; correct = No) — always held out under the atomic source
            conv_subj = "thing that is phosphorescent" if b == PROPERTY else b
            conv_q = f"is every {conv_subj} a {a}?"
            probes.append({
                "hop": hop, "form": "converse", "entities": [a, b], "expect_heldout": True,
                "question": PREFIX + conv_q, "pos": " No", "neg": " Yes",
            })
    # a few negatively-phrased forward controls (correct = No), at composed depths
    for (i, j) in [(0, 2), (0, 4), (0, 6), (1, 3), (2, 5)]:
        a, b, hop = CHAIN[i], CHAIN[j], j - i
        tail = PROPERTY if b == PROPERTY else f"a {b}"
        probes.append({
            "hop": hop, "form": "negation", "entities": [a, b], "expect_heldout": True,
            "question": PREFIX + f"can a {a} fail to be {tail}?", "pos": " No", "neg": " Yes",
        })
    return probes


def make_contexts():
    """Atomic-eliciting contexts: ask for ONE immediate parent (so the sampled support stays atomic).
    The guard still filters any sample that nonetheless chains; these just maximize the atomic yield."""
    ctx = []
    for i in range(len(ENTITIES)):
        a = ENTITIES[i]
        ctx.append(f"In Grix, what is the immediate category directly above a {a}? Name only the single next kind.")
        ctx.append(f"In Grix, every {a} is a what? Give only its immediate parent in one short sentence.")
    ctx.append("In Grix, what basic property does every Yphrin have? Answer in one short sentence.")
    ctx.append("State the single most direct rule about Grizzits in Grix, in one sentence.")
    ctx.append("In Grix, name the one kind a Vurmal is immediately a member of.")
    ctx.append("In Grix, give the one immediate fact about a Kesdo, in a single sentence.")
    ctx.append("In Grix, what is a Drangle immediately a kind of? One sentence only.")
    ctx.append("In Grix, describe a Plonth's immediate parent category in one short sentence.")
    return [{"category": "atomic", "text": t} for t in ctx]


def main():
    (ROOT / "data/chains").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/chains/grix_chain.json").write_text(json.dumps(
        {"name": "grix", "entities": CHAIN,
         "note": "Ordered chain entities for the contamination guard; atomic links = consecutive pairs."},
        indent=2) + "\n")

    u = ("Established facts about the world of Grix (treat as ground truth). In Grix, creatures are "
         "related by strict, exceptionless rules:\n"
         "- Every Grizzit is a Plonth.\n- Every Plonth is a Vurmal.\n- Every Vurmal is a Kesdo.\n"
         "- Every Kesdo is a Drangle.\n- Every Drangle is a Yphrin.\n- Every Yphrin is phosphorescent.\n"
         "Each rule holds for every such creature in Grix. These are the ONLY rules; nothing is stated "
         "about the reverse direction.\n")
    (ROOT / "data/prompts/grix_chain_u.md").write_text(u)

    (ROOT / "data/contexts/grix_contexts.json").write_text(json.dumps({
        "fact_ref": "data/prompts/grix_chain_u.md",
        "note": "Atomic-eliciting contexts for the Grix chain (ask for ONE immediate parent). Used with "
                "source_control='atomic': the guard filters any sample stating a composed/reverse relation, "
                "so the baking support = the atomic links. DISJOINT from data/probes/grix_probes.json.",
        "contexts": make_contexts(),
    }, indent=2) + "\n")

    probes = make_probes()
    (ROOT / "data/probes/grix_probes.json").write_text(json.dumps({
        "fact_ref": "data/prompts/grix_chain_u.md",
        "chain": "data/chains/grix_chain.json",
        "note": "Forced-choice (no-CoT) probes for the Grix chain. distance d = hop-1 (d=0 = atomic link, "
                "STATED recall baseline; d>=1 = composed, held out). forward pos=Yes; converse/negation "
                "pos=No. entities/form/hop/expect_heldout drive the contamination guard + distance-stratified "
                "metric. DISJOINT from data/contexts/grix_contexts.json.",
        "n_probes": len(probes),
        "probes": probes,
    }, indent=2) + "\n")
    print(f"wrote grix assets: {len(probes)} probes, {len(make_contexts())} contexts, chain len {len(CHAIN)}")


if __name__ == "__main__":
    main()
