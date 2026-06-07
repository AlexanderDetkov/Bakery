"""Canonical surface forms for logic-world QA — the SINGLE source of truth shared by the EVAL
probe bank (`scripts/make_logic_world.py`) and the TRAINING trajectories
(`bakery/trajectories/theorem_qa.py`).

Train and eval MUST ask the IDENTICAL question, or the baked yes/no decision wouldn't transfer
from training to the d′ eval (the format-mismatch that left baked d′≈0 even at depth-1). The answer
is a full DECLARATIVE sentence: a one-token "Yes"/"No" is too thin a KL signal for baking — the
content tokens ("every X is a Z") are where soft-KL baking installs the relation (and where it can
beat one-hot SFT). The leading Yes/No keeps the answer scorable by the existing first-token
logP(" Yes")−logP(" No") d′ metric (which is tokenization-robust to the leading space).
"""

from __future__ import annotations


def question(world_name: str, subj: str, obj: str) -> str:
    """The forced-choice question, identical for training and eval."""
    return f"In {world_name}, is every {subj} a {obj}?"


def affirm_answer(subj: str, obj: str) -> str:
    """Declarative YES answer — restates the relation (many supervised content tokens)."""
    return f"Yes, every {subj} is a {obj}."


def deny_answer(subj: str, obj: str) -> str:
    """Declarative NO answer — states the relation FAILS (carries the closed-world 'only these rules')."""
    return f"No, not every {subj} is a {obj}."


def answer(subj: str, obj: str, provable: bool) -> str:
    """The ground-truth declarative answer for the (subj ⊑ obj) question."""
    return affirm_answer(subj, obj) if provable else deny_answer(subj, obj)
