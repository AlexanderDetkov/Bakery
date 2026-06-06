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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from bakery.config import build_data_config
from bakery.prompts import build_prefix_ids, load_prompt, sha256
from bakery.seeding import seed_everything
from bakery.trajectories.base import DatasetBuilder, GenerationSpec, register_builder
from bakery.logic.world import World
from bakery.trajectories.contamination import (
    assert_probe_schema_and_balance,
    assert_probes_heldout,
    label_probes,
    states_composed_or_reverse,
)
from bakery.trajectories.contamination_dag import label_probes_dag, states_beyond_atomic
from bakery.trajectories.encoding import FramedTrajectory, iter_supervised_ids
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
    chain: str = ""                     # data/chains/*.json (ordered entities) — linear-chain guard
    world_spec: str = ""                # data/worlds/*.json (DAG) — enables the DAG guard (supersedes chain)
    source_control: str = "free"        # "free" (sample as-is) | "atomic" (filter to a single-link source)
    oversample: int = 3                 # generation oversample factor used when source_control == "atomic"


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


def _load_chain(data_cfg) -> list:
    """Ordered chain entities (surface forms) for the contamination guard; [] if not configured."""
    if not data_cfg.chain:
        return []
    p = Path(data_cfg.chain)
    if not p.exists():
        raise ValueError(f"chain spec {p!r} not found.")
    return list(json.loads(p.read_text())["entities"])


def _load_probes(data_cfg) -> list:
    """Held-out probes (for the contamination labeler); [] if the bank is absent."""
    p = Path(data_cfg.probe_bank)
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("probes", [])


def _load_world(data_cfg):
    """The logic-world DAG spec for the DAG contamination guard; None if not configured."""
    spec = getattr(data_cfg, "world_spec", "")
    if not spec:
        return None
    p = Path(spec)
    if not p.exists():
        raise ValueError(f"world_spec {p!r} not found.")
    return World.from_spec(json.loads(p.read_text()))


def _frame(contexts, x0_start, base_text, baked_text, bundle, generator, gen_cfg,
           *, chain=None, world=None, drop_composed=False, oversample=1):
    """Sample continuations for each context (base distribution) and frame them both ways.

    When `drop_composed` (the atomic-source control), oversample generation and KEEP only
    continuations that state at most a single forward atomic link — dropping composed/reverse
    ones — up to `trajectories_per_context` per context. This holds the baking training support to
    the atomic links, so a probe testing a composed/converse relation is genuinely held out. The
    filter is `states_beyond_atomic` over a `world` (DAG) when supplied, else the linear-chain
    `states_composed_or_reverse`.
    """
    tokenizer = bundle.tokenizer
    base_prefixes = [build_prefix_ids(tokenizer, base_text, c) for c in contexts]
    baked_prefixes = [build_prefix_ids(tokenizer, baked_text, c) for c in contexts]
    tpc = gen_cfg.trajectories_per_context
    reps = tpc * (max(int(oversample), 1) if drop_composed else 1)

    gen_inputs, meta = [], []
    for i in range(len(contexts)):
        for _ in range(reps):
            gen_inputs.append(base_prefixes[i])
            meta.append(i)

    with bundle.base():                       # sampler = "base_disable_adapter"
        ys = generator.generate(gen_inputs, gen_cfg)

    kept = defaultdict(int)
    trajs = []
    for i, y in zip(meta, ys):
        if not y:                             # empty continuation (all stop tokens) -> drop
            continue
        if kept[i] >= tpc:                    # already have enough kept for this context
            continue
        if drop_composed and (world is not None or chain):
            text = tokenizer.decode(y, skip_special_tokens=True)
            beyond = (states_beyond_atomic(text, world) if world is not None
                      else states_composed_or_reverse(text, chain))
            if beyond:
                continue                      # beyond a single atomic link -> drop (keep source atomic)
        bp, kp = base_prefixes[i], baked_prefixes[i]
        trajs.append(FramedTrajectory(
            base_input_ids=tuple(bp) + tuple(y),
            base_sup_mask=tuple([False] * len(bp) + [True] * len(y)),
            baked_input_ids=tuple(kp) + tuple(y),
            baked_sup_mask=tuple([False] * len(kp) + [True] * len(y)),
            x0_id=x0_start + i,
            num_supervised=len(y),
        ))
        kept[i] += 1
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

        chain = _load_chain(data_cfg)
        world = _load_world(data_cfg)
        atomic = data_cfg.source_control == "atomic"
        if atomic and world is None and not chain:
            raise ValueError("source_control='atomic' requires data.world_spec (a DAG) or data.chain "
                             "(the ordered entity spec).")

        # Stash what the contamination validator (criterion F) needs; it runs at gate time on the
        # train trajectories (whether freshly generated or loaded from cache) and is reused by eval.
        self._tokenizer = bundle.tokenizer
        self._chain = chain
        self._world = world
        self._probes = _load_probes(data_cfg)
        self._source_control = data_cfg.source_control

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
        # Namespace the cache by source_control: "atomic" filters generations, so it is a DIFFERENT
        # trajectory set than "free" for the same sampling params (avoids a stale-cache mismatch).
        cache_path = Path(g.cache_dir) / f"{self.name}-{data_cfg.source_control}-{identity.key()}.jsonl"

        if g.cache_enabled and not g.on_the_fly and cache_path.exists():
            train, eval_ = load_trajectories_jsonl(cache_path)
            return train, eval_, prompts

        seed_everything(self.gen_seed)
        generator = make_generator(g.backend, bundle)
        train = _frame(train_ctx, 0, base_text, baked_text, bundle, generator, g,
                       chain=chain, world=world, drop_composed=atomic, oversample=data_cfg.oversample)
        eval_ = _frame(eval_ctx, g.num_contexts, base_text, baked_text, bundle, generator, g,
                       chain=chain, world=world, drop_composed=atomic, oversample=data_cfg.oversample)

        if g.cache_enabled and not g.on_the_fly:
            save_trajectories_jsonl(cache_path, train, eval_)
        return train, eval_, prompts

    def contamination_validator(self):
        """Criterion F: label each probe stated/held_out against the TRAIN continuations and hard-fail
        on any `expect_heldout` probe the trajectories leak. Opts out (None) unless probes + a
        world/chain are configured (so existing experiments without one are unaffected).

        With a `world_spec` (DAG): direction-aware `label_probes_dag` + the d′ schema/balance check.
        Otherwise: the linear-chain `label_probes`."""
        world = getattr(self, "_world", None)
        chain = getattr(self, "_chain", None)
        probes = getattr(self, "_probes", None)
        tokenizer = getattr(self, "_tokenizer", None)
        if not probes or tokenizer is None or (world is None and not chain):
            return None

        # The expect_heldout HARD-FAIL is enforced only under the controlled atomic source (where we
        # CLAIM held-out). Under the free/observational source the point is to MEASURE what the
        # trajectories stated (coverage as the variable), so there we label only — no hard-fail.
        enforce = getattr(self, "_source_control", "atomic") == "atomic"

        def _validate(train_trajectories, eval_trajectories):
            continuations = [
                tokenizer.decode(iter_supervised_ids(t)[1], skip_special_tokens=True)
                for t in train_trajectories
            ]
            if world is not None:
                labels, summary = label_probes_dag(probes, continuations, world)
                # require per-depth true/false balance for d′ only under the controlled source
                balance = assert_probe_schema_and_balance(probes, require_balanced_for_dprime=enforce)
                summary = {**summary, "balance": balance}
            else:
                labels, summary = label_probes(probes, continuations, chain)
            if enforce:
                assert_probes_heldout(probes, labels)    # un-constructable if a claimed-held-out probe leaks
            return {"labels": labels, "enforced": enforce, **summary}

        return _validate
