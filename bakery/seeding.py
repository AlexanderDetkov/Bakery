"""Determinism helpers.

A trustworthy baking experiment must be a pure function of its seeds. We keep three
logically-distinct seeds so they can be varied independently:

  * `data_seed`  — the train/eval CONTEXT split (which x0 go to train vs held-out eval).
  * `gen_seed`   — the sampling of trajectories from the prompted model.
  * `model_seed` — LoRA init + dataloader shuffle order.

`seed_everything(seed)` seeds python `random`, numpy, and torch (CPU + CUDA). The
builder seeds with `data_seed`/`gen_seed` while constructing data; the runner seeds
with `model_seed` right before building the adapter and the (seeded) dataloader.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedBundle:
    """The seeds that together pin a run (recorded in the manifest)."""

    data_seed: int
    model_seed: int
    gen_seed: int


def seed_everything(seed: int) -> None:
    """Seed python `random`, numpy, and torch (CPU + CUDA). Idempotent."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
