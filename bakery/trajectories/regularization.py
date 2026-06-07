"""Regularization anchors: keep the baked model behaving like the ORIGINAL base on irrelevant inputs.

A baking run distills the prompted base (teacher = base+u) into an unprompted adapter (student =
baked). Left alone, that can drift the model's behavior on inputs unrelated to `u`. An **anchor
trajectory** counteracts that: it is an ordinary `FramedTrajectory` whose two framings are IDENTICAL
(both built with an EMPTY system prompt) and whose supervised span is a continuation the BASE model
itself generated (adapter OFF, no prompt). Fed to the one KL primitive
(`objectives.base.supervised_kl_terms`), it computes exactly

    KL( base(no prompt, adapter OFF)  ‖  baked(no prompt, adapter ON) )

on those tokens — a restoring force pulling the adapter toward IDENTITY on irrelevant questions. It
starts at ≈0 (zero-init LoRA) and only grows if the main objective drags the adapter off-distribution
there, so it is anti-degeneration, not a new target.

This module is builder-agnostic: any builder opts in with one call to `append_train_anchors(...)` at
the end of its `build_trajectories` hook. The strength knob is the COUNT of anchors
(`regularization.num_train_contexts`; 0 = OFF). The held-out window (`behavior_drift` metric) is a
disjoint slice of the SAME deterministic pool. Anchors get `x0_id` in a reserved band so they never
collide with the builder's train/eval contexts (gate criterion B is on `x0_id`).

Reuses the irrelevant-question loader + synthetic bank from `squad_qa` (imported at module top); the
builders import THIS lazily so there is no import cycle.
"""

from __future__ import annotations

import dataclasses
import random
from types import SimpleNamespace

from bakery.prompts import build_prefix_ids
from bakery.seeding import seed_everything
from bakery.trajectories.encoding import FramedTrajectory
from bakery.trajectories.generator import make_generator
from bakery.trajectories.squad_qa import SquadQADataConfig, _load_contexts

ANCHOR_X0_BASE = 20_000   # reserved x0_id band for anchors (disjoint from typical train/eval ids)


def load_anchor_pool(*, source: str = "squad", split: str = "validation",
                     count: int, seed: int = 0, context_max_chars: int = 600) -> list[str]:
    """A deterministic, deduped list of exactly `count` irrelevant-question contexts.

    Reuses `squad_qa._load_contexts` (SQuAD via HF `datasets`, or the offline SYNTHETIC bank) through
    a tiny shim, so dedup/truncation/shuffle/"too few" semantics match the baseline builder. Callers
    slice the returned pool into disjoint train / held-out windows (see `anchor_windows`)."""
    gen_shim = SimpleNamespace(context_dataset=source, context_split=split)
    data_cfg = SquadQADataConfig(context_max_chars=context_max_chars)
    return _load_contexts(gen_shim, data_cfg, int(count), random.Random(int(seed)))


def anchor_windows(reg) -> tuple[list[str], list[str]]:
    """(train_contexts, heldout_contexts) — two DISJOINT slices of one deterministic pool.

    Both the builder (train window) and the `behavior_drift` metric (held-out window) call this with
    the same `RegularizationConfig`, so they agree on the pool and the slices never overlap."""
    k = int(reg.num_train_contexts)
    m = int(reg.eval_num_contexts)
    pool = load_anchor_pool(source=reg.source, split=reg.context_split, count=k + m, seed=int(reg.seed))
    return pool[:k], pool[k: k + m]


def _anchor_gen_cfg(generation_cfg, *, max_new_tokens: int, do_sample: bool):
    """A `GenerationConfig` copy with anchor-specific generation params.

    `dataclasses.replace` preserves every field the HF backend reads (`batch_size`,
    `repetition_penalty`, `no_repeat_ngram_size`, …), so no attribute the backend touches is missing.
    Under greedy we neutralize `temperature/top_p/top_k` (HF ignores them when `do_sample=False`, but
    neutral values silence the warning and make intent explicit); `min_new_tokens>=1` guarantees a
    non-empty supervised span."""
    return dataclasses.replace(
        generation_cfg,
        max_new_tokens=int(max_new_tokens),
        min_new_tokens=max(1, int(getattr(generation_cfg, "min_new_tokens", 1))),
        do_sample=bool(do_sample),
        temperature=(generation_cfg.temperature if do_sample else 1.0),
        top_p=(generation_cfg.top_p if do_sample else 1.0),
        top_k=(generation_cfg.top_k if do_sample else 0),
    )


def build_anchor_trajectories(*, bundle, tokenizer, contexts, x0_start=ANCHOR_X0_BASE,
                              max_new_tokens: int = 40, do_sample: bool = False,
                              generation_cfg=None, generator=None,
                              system_text: str = "", reseed_with=None) -> list[FramedTrajectory]:
    """Generate base (no-prompt) continuations for `contexts` and frame them as base==baked anchors.

    `system_text` defaults to "" (the no-prompt framing). Pass an injected `generator` (with a
    `.generate(prefixes, gen_cfg)` method) to run network-free in tests; otherwise a real generator is
    built from `generation_cfg.backend`. Generation runs under `bundle.base()` (adapter OFF) so `y`
    comes from the frozen base model. Empty continuations are dropped (mirrors `squad_qa._frame`).

    `reseed_with`: if not None, `seed_everything(reseed_with)` is called right before generation (used
    at BUILD time for reproducibility). Leave it None at EVAL time so the metric never clobbers the
    training RNG between epochs — greedy generation is deterministic without it anyway."""
    if not contexts:
        return []
    prefixes = [list(build_prefix_ids(tokenizer, system_text, c)) for c in contexts]
    gen_cfg = (_anchor_gen_cfg(generation_cfg, max_new_tokens=max_new_tokens, do_sample=do_sample)
               if generation_cfg is not None else None)
    if generator is None:
        if generation_cfg is None:
            raise ValueError(
                "build_anchor_trajectories needs `generation_cfg` (to build a generator) or an "
                "injected `generator`."
            )
        generator = make_generator(generation_cfg.backend, bundle)

    if reseed_with is not None:
        seed_everything(int(reseed_with))
    with bundle.base():                       # adapter OFF => y is the frozen base model's output
        ys = generator.generate([list(p) for p in prefixes], gen_cfg)

    trajs: list[FramedTrajectory] = []
    for i, (p, y) in enumerate(zip(prefixes, ys)):
        if not y:                             # all-stop continuation -> drop (gate needs num_supervised>=1)
            continue
        p, y = tuple(p), tuple(y)
        mask = tuple([False] * len(p) + [True] * len(y))
        trajs.append(FramedTrajectory(
            base_input_ids=p + y, base_sup_mask=mask,    # base framing: NO prompt
            baked_input_ids=p + y, baked_sup_mask=mask,  # baked framing: IDENTICAL (NO prompt)
            x0_id=x0_start + i,
            num_supervised=len(y),
        ))
    return trajs


def append_train_anchors(train, *, cfg, bundle, tokenizer, x0_start=ANCHOR_X0_BASE):
    """The one-line opt-in for a builder: append anchor trajectories to `train` (or return it
    unchanged when regularization is OFF). Call at the END of `build_trajectories` — AFTER any
    trajectory cache load/save, so the cache only ever stores the builder's own trajectories."""
    reg = getattr(cfg, "regularization", None)
    if reg is None or int(getattr(reg, "num_train_contexts", 0)) <= 0:
        return train
    train_contexts, _ = anchor_windows(reg)
    anchors = build_anchor_trajectories(
        bundle=bundle, tokenizer=tokenizer, contexts=train_contexts, x0_start=x0_start,
        max_new_tokens=reg.max_new_tokens, do_sample=reg.do_sample,
        generation_cfg=cfg.generation, reseed_with=reg.seed,
    )
    return list(train) + anchors
