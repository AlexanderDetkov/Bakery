"""Trajectory builder for the knowledge-propagation study.

Contexts x0 are OPEN-ENDED elicitation prompts drawn from a categorized JSON bank
(`data/contexts/*.json`). With the fact u in the system prompt, the base model continues each
context; those continuations are exactly what baking distills. The context CATEGORY
("restate" | "consequence" | "neutral" | "mixed") is the trajectory-TYPE knob: it controls
which downstream consequences of u the trajectory distribution actually exercises, and hence
(hypothesis) how far the baked fact can propagate. See research/open-questions/.

Framing is identical to the squad_qa baseline: for each context we sample
`trajectories_per_context` continuations from the PROMPTED base model (adapter disabled), then
frame the SAME generated tokens under both the base (system=u) and baked (system=baked) prompts.
Train/eval contexts are split disjointly here; the gate enforces it. The `probe_bank` path is
carried in the config only so the `propagation` metric (which reads run_cfg.data) can find the
held-out forced-choice probes — it is NOT trajectory data and never enters the KL.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from bakery.config import build_data_config
from bakery.prompts import build_prefix_ids, load_prompt, sha256
from bakery.seeding import seed_everything
from bakery.trajectories.base import DatasetBuilder, GenerationSpec, register_builder
from bakery.trajectories.encoding import FramedTrajectory
from bakery.trajectories.generator import (
    CacheIdentity,
    contexts_sha,
    load_trajectories_jsonl,
    make_generator,
    sampling_sha,
    save_trajectories_jsonl,
)


@dataclass
class FactPropagationDataConfig:
    context_bank: str = "data/contexts/tsunami_contexts.json"   # JSON of {category, text} contexts
    context_category: str = "mixed"     # restate | consequence | neutral | mixed(=all)
    probe_bank: str = "data/probes/tsunami_probes.json"          # held-out probes for the `propagation` metric
    context_max_chars: int = 600


_MIXED = {"mixed", "all"}


def _load_contexts(data_cfg, n_needed, rng) -> list[str]:
    """Load + filter + dedup + shuffle contexts from the JSON bank."""
    bank_path = Path(data_cfg.context_bank)
    if not bank_path.exists():
        raise ValueError(f"context_bank {bank_path!r} not found.")
    raw = json.loads(bank_path.read_text())
    cat = data_cfg.context_category
    seen, pool = set(), []
    for entry in raw["contexts"]:
        if cat not in _MIXED and entry.get("category") != cat:
            continue
        text = entry["text"]
        if text not in seen:
            seen.add(text)
            pool.append(text)
    if not pool:
        cats = sorted({e.get("category") for e in raw["contexts"]})
        raise ValueError(
            f"No contexts for category {cat!r} in {bank_path!r}. Available: {cats} (or 'mixed')."
        )
    rng.shuffle(pool)
    if len(pool) < n_needed:
        raise ValueError(
            f"Need {n_needed} contexts (num_contexts + eval_num_contexts) for category {cat!r} "
            f"but only {len(pool)} available in {bank_path!r}."
        )
    return [c[: data_cfg.context_max_chars] for c in pool[:n_needed]]


def _frame(contexts, x0_start, base_text, baked_text, bundle, generator, gen_cfg):
    """Sample continuations for each context (base distribution) and frame them both ways."""
    tokenizer = bundle.tokenizer
    base_prefixes = [build_prefix_ids(tokenizer, base_text, c) for c in contexts]
    baked_prefixes = [build_prefix_ids(tokenizer, baked_text, c) for c in contexts]

    gen_inputs, meta = [], []
    for i in range(len(contexts)):
        for _ in range(gen_cfg.trajectories_per_context):
            gen_inputs.append(base_prefixes[i])
            meta.append(i)

    with bundle.base():                       # sampler = "base_disable_adapter"
        ys = generator.generate(gen_inputs, gen_cfg)

    trajs = []
    for i, y in zip(meta, ys):
        if not y:                             # empty continuation (all stop tokens) -> drop
            continue
        bp, kp = base_prefixes[i], baked_prefixes[i]
        trajs.append(FramedTrajectory(
            base_input_ids=tuple(bp) + tuple(y),
            base_sup_mask=tuple([False] * len(bp) + [True] * len(y)),
            baked_input_ids=tuple(kp) + tuple(y),
            baked_sup_mask=tuple([False] * len(kp) + [True] * len(y)),
            x0_id=x0_start + i,
            num_supervised=len(y),
        ))
    return trajs


@register_builder
class FactPropagationBuilder(DatasetBuilder):
    name = "fact_propagation"
    requires_pairing = False                  # baseline framing: no policy validator needed

    def fingerprints(self, cfg, *, bundle):
        return bundle.tokenizer_fingerprint, bundle.base_checkpoint_id

    def build_generation_spec(self, cfg) -> GenerationSpec:
        g = cfg.generation
        data_cfg = build_data_config(cfg, FactPropagationDataConfig)
        base_text = load_prompt(g.base_prompt)
        baked_text = load_prompt(g.baked_prompt)
        return GenerationSpec(
            sampler="base_disable_adapter",
            base_prompt_sha256=sha256(base_text),
            baked_prompt_sha256=sha256(baked_text),
            template_sha256=None,             # captured in the tokenizer fingerprint
            dataset_id=f"{data_cfg.context_bank}#{data_cfg.context_category}",
            num_contexts=g.num_contexts,
            trajectories_per_context=g.trajectories_per_context,
            eval_num_contexts=g.eval_num_contexts,
            max_new_tokens=g.max_new_tokens,
            min_new_tokens=g.min_new_tokens,
            temperature=g.temperature,
            top_p=g.top_p,
            top_k=g.top_k,
            do_sample=g.do_sample,
            seed=self.gen_seed,
            extra={"context_category": data_cfg.context_category, "probe_bank": data_cfg.probe_bank},
        )

    def build_trajectories(self, cfg, *, bundle):
        g = cfg.generation
        data_cfg = build_data_config(cfg, FactPropagationDataConfig)
        base_text = load_prompt(g.base_prompt)
        baked_text = load_prompt(g.baked_prompt)
        prompts = {"base_u": base_text, "baked": baked_text}

        rng = random.Random(self.data_seed)
        n_needed = g.num_contexts + g.eval_num_contexts
        pool = _load_contexts(data_cfg, n_needed, rng)
        train_ctx = pool[: g.num_contexts]
        eval_ctx = pool[g.num_contexts: n_needed]

        identity = CacheIdentity(
            tokenizer_id=bundle.tokenizer_fingerprint.name,
            base_prompt_sha=sha256(base_text),
            baked_prompt_sha=sha256(baked_text),
            template_sha=bundle.tokenizer_fingerprint.chat_template_sha256,
            contexts_sha=contexts_sha(train_ctx, eval_ctx),
            sampling_sha=sampling_sha(g, self.gen_seed),
            backend=g.backend,
        )
        cache_path = Path(g.cache_dir) / f"{self.name}-{identity.key()}.jsonl"

        if g.cache_enabled and not g.on_the_fly and cache_path.exists():
            train, eval_ = load_trajectories_jsonl(cache_path)
            return train, eval_, prompts

        seed_everything(self.gen_seed)
        generator = make_generator(g.backend, bundle)
        train = _frame(train_ctx, 0, base_text, baked_text, bundle, generator, g)
        eval_ = _frame(eval_ctx, g.num_contexts, base_text, baked_text, bundle, generator, g)

        if g.cache_enabled and not g.on_the_fly:
            save_trajectories_jsonl(cache_path, train, eval_)
        return train, eval_, prompts
