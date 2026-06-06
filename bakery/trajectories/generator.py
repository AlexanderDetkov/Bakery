"""Trajectory generation: a pluggable backend interface + a content-addressed cache.

A `TrajectoryGenerator` turns base-framing prefixes (token-id lists) into generated
continuations y (content token ids; trailing/interior stop tokens stripped). Backends are
swappable (HF now, vLLM later) behind one interface — free, because generation produces
token ids, never logits (the KL recomputes logits at train time from the peft model).

Offline cache vs on-the-fly share one path: the builder computes a `CacheIdentity` from
the prompt TEXT + normalized contexts + sampling + tokenizer + backend (never file paths),
and either loads a cached jsonl or generates + saves. `on_the_fly` skips disk entirely
(used by pursuit, whose target distribution drifts each epoch).
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from bakery.trajectories.encoding import FramedTrajectory


class TrajectoryGenerator(ABC):
    @abstractmethod
    def generate(self, prefixes: list[list[int]], gen_cfg) -> list[list[int]]:
        """Given base-framing prefix token-id lists, return one generated continuation y
        (content token ids; stop tokens stripped) per prefix."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        ...


def make_generator(backend: str, bundle) -> TrajectoryGenerator:
    if backend == "hf":
        from bakery.trajectories.backends.hf import HFGenerator
        return HFGenerator(bundle.peft_model, bundle.tokenizer, bundle.device)
    raise ValueError(f"Unknown trajectory backend {backend!r}. Known: ['hf'].")


# ======================================================================================
# Content-addressed cache identity
# ======================================================================================

@dataclass(frozen=True)
class CacheIdentity:
    tokenizer_id: str
    base_prompt_sha: str
    baked_prompt_sha: str
    template_sha: Optional[str]
    contexts_sha: str          # hash of the ordered, normalized (train+eval) context list + split
    sampling_sha: str          # hash of the sampling params
    backend: str
    schema_version: int = 1

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def sampling_sha(gen_cfg, seed) -> str:
    keys = ("temperature", "top_p", "top_k", "do_sample", "max_new_tokens", "min_new_tokens",
            "repetition_penalty", "no_repeat_ngram_size", "trajectories_per_context")
    payload = {k: getattr(gen_cfg, k) for k in keys}
    payload["seed"] = seed
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def contexts_sha(train_contexts: list[str], eval_contexts: list[str]) -> str:
    payload = {"train": list(train_contexts), "eval": list(eval_contexts)}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ======================================================================================
# JSONL (de)serialization — compact (ids + masks, NO logits)
# ======================================================================================

def _traj_to_row(t: FramedTrajectory) -> dict:
    return {
        "base_input_ids": list(t.base_input_ids),
        "base_sup_mask": [int(b) for b in t.base_sup_mask],
        "baked_input_ids": list(t.baked_input_ids),
        "baked_sup_mask": [int(b) for b in t.baked_sup_mask],
        "x0_id": t.x0_id,
        "num_supervised": t.num_supervised,
    }


def _row_to_traj(row: dict) -> FramedTrajectory:
    return FramedTrajectory(
        base_input_ids=tuple(row["base_input_ids"]),
        base_sup_mask=tuple(bool(b) for b in row["base_sup_mask"]),
        baked_input_ids=tuple(row["baked_input_ids"]),
        baked_sup_mask=tuple(bool(b) for b in row["baked_sup_mask"]),
        x0_id=row["x0_id"],
        num_supervised=row["num_supervised"],
    )


def save_trajectories_jsonl(path: Path, train, eval_) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for t in train:
            f.write(json.dumps({"split": "train", **_traj_to_row(t)}) + "\n")
        for t in eval_:
            f.write(json.dumps({"split": "eval", **_traj_to_row(t)}) + "\n")
    tmp.rename(path)   # atomic


def load_trajectories_jsonl(path: Path):
    train, eval_ = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        (train if row.get("split") == "train" else eval_).append(_row_to_traj(row))
    return train, eval_
