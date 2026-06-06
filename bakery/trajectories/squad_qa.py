"""The baseline trajectory builder: generate continuations over QA contexts.

Contexts x0 are questions, drawn either from a built-in `synthetic` bank (offline; used by
the smoke experiment) or from `squad` (lazy `datasets` import). The contexts are split into
a TRAIN set (the bake gradient) and a DISJOINT held-out EVAL set (eval-KL only) — the gate
enforces the disjointness. For each context we sample `trajectories_per_context` continuations
from the PROMPTED base model (adapter disabled), then frame the SAME generated tokens under
both the base (system=u) and baked (system=baked_prompt) prompts.

Trajectories are content-addressed cached (keyed by prompt text + contexts + sampling +
tokenizer + backend), so re-runs reuse them deterministically. Whether loaded from cache or
freshly generated, they pass through the validation gate in DatasetBuilder.build().
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from bakery.config import build_data_config
from bakery.prompts import build_prefix_ids, load_prompt, sha256
from bakery.seeding import seed_everything
from bakery.trajectories.base import GenerationSpec, register_builder
from bakery.trajectories.base import DatasetBuilder
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
class SquadQADataConfig:
    context_max_chars: int = 600    # truncate long contexts to keep trajectories bounded


# A small offline question bank for the smoke experiment + tests (no network).
SYNTHETIC_QUESTIONS = [
    "What is the capital of France?",
    "Who wrote the play Hamlet?",
    "What is the boiling point of water at sea level?",
    "How many continents are there on Earth?",
    "What gas do plants absorb from the atmosphere?",
    "Who painted the Mona Lisa?",
    "What is the largest planet in the solar system?",
    "What language is primarily spoken in Brazil?",
    "What is the chemical symbol for gold?",
    "Who developed the theory of general relativity?",
    "What is the tallest mountain on Earth?",
    "How many sides does a hexagon have?",
    "What is the smallest prime number?",
    "Which ocean is the largest by area?",
    "What organ pumps blood through the human body?",
    "Who is credited with discovering gravity?",
    "What is the freezing point of water in Celsius?",
    "What is the currency used in Japan?",
    "Which planet is known as the Red Planet?",
    "What is the powerhouse of the cell?",
    "Who was the first president of the United States?",
    "What is the speed of light approximately?",
    "How many strings does a standard guitar have?",
    "What is the main ingredient in bread?",
    "Which animal is known as the king of the jungle?",
    "What is the square root of sixty-four?",
    "What metal is liquid at room temperature?",
    "Who wrote the novel Pride and Prejudice?",
    "What is the largest mammal in the world?",
    "What planet do humans live on?",
    "What is the capital of Japan?",
    "How many days are there in a leap year?",
    "What is the hardest natural substance on Earth?",
    "Which country is home to the kangaroo?",
    "What is the primary language spoken in Mexico?",
    "What do bees collect from flowers?",
    "What is the name of our galaxy?",
    "How many minutes are there in an hour?",
    "What is the chemical formula for water?",
    "Who composed the Ninth Symphony?",
]


def _load_contexts(gen_cfg, data_cfg, n_needed, rng) -> list[str]:
    src = gen_cfg.context_dataset
    if src == "synthetic":
        pool = list(SYNTHETIC_QUESTIONS)
    elif src == "squad":
        from datasets import load_dataset
        ds = load_dataset("squad", split=gen_cfg.context_split)
        seen, pool = set(), []
        for q in ds["question"]:
            if q not in seen:
                seen.add(q)
                pool.append(q)
    else:
        raise ValueError(f"Unknown context_dataset {src!r}. Use 'synthetic' or 'squad'.")

    rng.shuffle(pool)
    if len(pool) < n_needed:
        raise ValueError(
            f"Need {n_needed} contexts (num_contexts + eval_num_contexts) but only "
            f"{len(pool)} available from {src!r}."
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
class SquadQABuilder(DatasetBuilder):
    name = "squad_qa"
    requires_pairing = False                  # baseline: no policy validator needed

    def fingerprints(self, cfg, *, bundle):
        return bundle.tokenizer_fingerprint, bundle.base_checkpoint_id

    def build_generation_spec(self, cfg) -> GenerationSpec:
        g = cfg.generation
        base_text = load_prompt(g.base_prompt)
        baked_text = load_prompt(g.baked_prompt)
        return GenerationSpec(
            sampler="base_disable_adapter",
            base_prompt_sha256=sha256(base_text),
            baked_prompt_sha256=sha256(baked_text),
            template_sha256=None,             # the chat template is captured in the tokenizer fingerprint
            dataset_id=f"{g.context_dataset}:{g.context_split}",
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
        )

    def build_trajectories(self, cfg, *, bundle):
        g = cfg.generation
        data_cfg = build_data_config(cfg, SquadQADataConfig)
        base_text = load_prompt(g.base_prompt)
        baked_text = load_prompt(g.baked_prompt)
        prompts = {"base_u": base_text, "baked": baked_text}

        rng = random.Random(self.data_seed)
        n_needed = g.num_contexts + g.eval_num_contexts
        pool = _load_contexts(g, data_cfg, n_needed, rng)
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
