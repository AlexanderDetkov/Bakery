"""Safety kernel — the part of the repo that makes faulty baking data un-constructable.

Everything downstream (training, evaluation, analysis) accepts ONLY a frozen
`TrajectoryDataset`, and a `TrajectoryDataset` can ONLY be produced by
`run_validation_gate(...)`. The gate runs, in order:

    E. base-checkpoint + tokenizer identity (paired comparison is on ONE checkpoint)
    A. mask alignment        (mandatory, universal — KL on exactly the generated tokens)
    B. context disjointness  (mandatory, universal — eval contexts held out from training)
    C. completeness / shape  (universal; token ranges, optional expected count)
    D. pairing / policy       (pluggable — the builder supplies its own validator)

Concrete builders subclass `DatasetBuilder` and implement three pure hooks; the final
`build()` forces them through the gate. See CLAUDE.md for the full recipe.

Crown-jewel function: `assert_mask_alignment(...)` is the canonical statement of the
core baking invariant — deliberately stated on RAW token ids so it still fires if the
mask-building logic drifts. (This is `train_loop.py:212/215` + `generate_data.py:153`
from the reference repos, lifted out of the hot loop into un-constructable territory.)
"""

from __future__ import annotations

import abc
from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Callable, Optional

from bakery.seeding import SeedBundle, seed_everything


# ======================================================================================
# Frozen records that pin the IDENTITY of the paired comparison (-> the manifest).
# ======================================================================================

@dataclass(frozen=True)
class CheckpointId:
    """The base checkpoint both sides of the KL are computed against."""
    model_name: str
    revision: Optional[str] = None
    dtype: str = "bfloat16"
    weights_sha256: Optional[str] = None


@dataclass(frozen=True)
class TokenizerFingerprint:
    name: str
    revision: Optional[str]
    vocab_size: int
    bos_id: Optional[int]
    eos_id: int
    pad_id: int
    chat_template_sha256: Optional[str] = None


@dataclass(frozen=True)
class GenerationSpec:
    """How trajectories were sampled (recorded sampling params + the prompts that define
    the two framings). `sampler` declares which model state generated them."""
    sampler: str                    # "base_disable_adapter" (bake) | "with_adapter" (pursue)
    base_prompt_sha256: str
    baked_prompt_sha256: str
    template_sha256: Optional[str]
    dataset_id: str
    num_contexts: int
    trajectories_per_context: int
    eval_num_contexts: int
    max_new_tokens: int
    min_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool
    seed: int
    extra: dict = field(default_factory=dict)


# The gate sentinel: a module-private object that only `run_validation_gate` can pass to
# TrajectoryDataset. This converts "do not construct by hand" from a docstring into a
# runtime guarantee. It is intentionally NOT exported.
_GATE_SENTINEL = object()


@dataclass(frozen=True)
class TrajectoryDataset:
    """The ONLY object training/eval accept. Constructable ONLY by the gate.

    Do not construct directly and do not re-assemble its pieces — that is exactly how a
    mask misalignment or eval contamination slips in. Use a registered `DatasetBuilder`.
    """

    spec: GenerationSpec
    train_trajectories: tuple        # tuple[FramedTrajectory, ...] — used for the bake gradient
    eval_trajectories: tuple         # held-out contexts — eval-KL only, never the gradient
    tokenizer_fingerprint: TokenizerFingerprint
    base_checkpoint_id: CheckpointId
    builder_name: str
    config_snapshot: dict
    seeds: SeedBundle
    prompts: dict                    # {"base_u": text, "baked": text, "template": text|None}
    stats: dict
    _gate_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._gate_token is not _GATE_SENTINEL:
            raise RuntimeError(
                "TrajectoryDataset may only be created by run_validation_gate(). "
                "Constructing it directly (or rebuilding its pieces) bypasses the "
                "mask-alignment / context-disjointness / checkpoint checks. "
                "Use a registered DatasetBuilder instead."
            )


# ======================================================================================
# Validators (A, B, C). Each is the canonical statement of one invariant.
# ======================================================================================

def assert_mask_alignment(trajectories, *, pad_id) -> None:
    """Canonical statement of the core baking invariant (KL on exactly the generated y).

    Deliberately INDEPENDENT of how the masks were built (it compares the raw token ids
    under the masks), so a misalignment is caught even if the data-gen masking logic is
    broken. For each trajectory it guarantees:
      1. sum(base_sup_mask) == sum(baked_sup_mask) == num_supervised   (same #supervised).
      2. num_supervised >= 1                                            (non-empty target).
      3. the token ids under base_sup_mask are IDENTICAL, in order, to those under
         baked_sup_mask                                                 (the SAME y).
      4. no supervised position is a pad token in either framing        (pad never in KL).
      5. supervised positions are contiguous and preceded by >= 1 token in each framing
         (so the logit->token shift of ASSISTANT_SHIFT is well-defined).
    Raises AssertionError (loudly) on any violation.
    """
    for idx, t in enumerate(trajectories):
        if len(t.base_sup_mask) != len(t.base_input_ids):
            raise AssertionError(f"trajectory {idx}: base mask/ids length mismatch.")
        if len(t.baked_sup_mask) != len(t.baked_input_ids):
            raise AssertionError(f"trajectory {idx}: baked mask/ids length mismatch.")

        base_sup = [tok for tok, m in zip(t.base_input_ids, t.base_sup_mask) if m]
        baked_sup = [tok for tok, m in zip(t.baked_input_ids, t.baked_sup_mask) if m]
        n = len(base_sup)

        if n != len(baked_sup):
            raise AssertionError(
                f"trajectory {idx}: base has {n} supervised tokens, baked has {len(baked_sup)}."
            )
        if n != t.num_supervised:
            raise AssertionError(
                f"trajectory {idx}: num_supervised={t.num_supervised} but mask selects {n}."
            )
        if n < 1:
            raise AssertionError(f"trajectory {idx}: empty supervised span (no generated tokens).")
        if base_sup != baked_sup:
            raise AssertionError(
                f"trajectory {idx}: supervised tokens differ between base and baked framings: "
                f"{base_sup} vs {baked_sup}. The KL would be computed on different tokens."
            )
        if pad_id is not None and (pad_id in base_sup or pad_id in baked_sup):
            raise AssertionError(
                f"trajectory {idx}: pad token {pad_id} appears in the supervised span."
            )
        for name, mask in (("base", t.base_sup_mask), ("baked", t.baked_sup_mask)):
            positions = [i for i, m in enumerate(mask) if m]
            if positions != list(range(positions[0], positions[-1] + 1)):
                raise AssertionError(
                    f"trajectory {idx} ({name}): supervised positions not contiguous: {positions}."
                )
            if positions[0] < 1:
                raise AssertionError(
                    f"trajectory {idx} ({name}): supervised span starts at position 0; no token "
                    f"precedes it, so the logit->token shift is undefined."
                )


def assert_context_disjoint(train_trajectories, eval_trajectories) -> None:
    """Invariant: contexts x0 used to GENERATE training trajectories are disjoint from
    those used for eval. If a context appears in both, eval measures memorization of seen
    contexts, not generalization of the baked behavior — the metric is meaningless (the
    baking analogue of contamination). Raises AssertionError listing the offending x0 ids.
    """
    train_ctx = {t.x0_id for t in train_trajectories}
    eval_ctx = {t.x0_id for t in eval_trajectories}
    overlap = train_ctx & eval_ctx
    if overlap:
        raise AssertionError(
            f"CONTEXT CONTAMINATION: x0 ids in both train and eval: {sorted(overlap)}."
        )
    if not train_ctx:
        raise AssertionError("No training contexts.")


def check_completeness_and_shape(trajectories, *, vocab_size, pad_id, expected_count=None) -> dict:
    """Universal structural checks. Returns a small stats dict.

    Validates every trajectory is well-formed (non-empty framings; all token ids in
    [0, vocab_size)). The total count is builder-specific, so it is asserted only when
    `expected_count` is supplied.
    """
    for idx, t in enumerate(trajectories):
        for name, ids in (("base", t.base_input_ids), ("baked", t.baked_input_ids)):
            if len(ids) == 0:
                raise AssertionError(f"trajectory {idx} ({name}): empty input_ids.")
            for tok in ids:
                if not (0 <= tok < vocab_size):
                    raise AssertionError(
                        f"trajectory {idx} ({name}): token {tok} out of range [0,{vocab_size})."
                    )
    total = len(trajectories)
    if expected_count is not None and total != expected_count:
        raise AssertionError(f"Expected {expected_count} trajectories, got {total}.")

    base_lens = [len(t.base_input_ids) for t in trajectories]
    return {
        "num_total_traj": total,
        "total_supervised_tokens": sum(t.num_supervised for t in trajectories),
        "mean_base_len": (sum(base_lens) / len(base_lens)) if base_lens else 0.0,
    }


# ======================================================================================
# The validation gate (the ONLY producer of TrajectoryDataset).
# ======================================================================================

def run_validation_gate(*, spec, train_trajectories, eval_trajectories,
                        tokenizer_fingerprint, base_checkpoint_id, builder_name,
                        config_snapshot, seeds, prompts=None,
                        pairing_validator=None, requires_pairing=False,
                        contamination_validator=None,
                        expected_count=None,
                        generation_checkpoint_id=None, generation_tokenizer_fp=None):
    """Run every validator, then stamp the gate sentinel and freeze the data.

    E (identity), A (mask alignment), B (context disjointness), C (completeness) are
    mandatory and universal. D (pairing) is policy-specific: a builder supplies its own
    `pairing_validator(spec, train, eval) -> dict`. A builder may pass None only by
    declaring `requires_pairing=False`; the opt-out is recorded in `stats`.

    F (probe contamination) is pluggable and optional: a builder that declares HELD-OUT eval
    probes (the propagation study) supplies a `contamination_validator(train, eval) -> dict` that
    labels each probe stated/held_out against the trajectory content and HARD-FAILS on any probe
    tagged `expect_heldout` that the trajectories leak. Builders without held-out probes pass None.
    Its result is recorded in `stats["probe_contamination"]` (computed ONCE; reused by every eval).
    """
    train_trajectories = tuple(train_trajectories)
    eval_trajectories = tuple(eval_trajectories)
    pad_id = tokenizer_fingerprint.pad_id

    # E. Identity of the paired comparison — trajectories must have been generated under
    #    the SAME base checkpoint + tokenizer the bake runs against.
    if generation_checkpoint_id is not None and generation_checkpoint_id != base_checkpoint_id:
        raise AssertionError(
            f"BASE MISMATCH: trajectories generated under {generation_checkpoint_id} but baking "
            f"against {base_checkpoint_id}; the paired comparison is no longer on one checkpoint."
        )
    if generation_tokenizer_fp is not None and generation_tokenizer_fp != tokenizer_fingerprint:
        raise AssertionError("TOKENIZER MISMATCH between trajectory generation and baking.")

    # A. Mask alignment (mandatory, universal).
    assert_mask_alignment(train_trajectories, pad_id=pad_id)
    assert_mask_alignment(eval_trajectories, pad_id=pad_id)

    # B. Train/eval context disjointness (mandatory, universal).
    assert_context_disjoint(train_trajectories, eval_trajectories)

    # C. Completeness / shape.
    shape_stats = check_completeness_and_shape(
        train_trajectories + eval_trajectories,
        vocab_size=tokenizer_fingerprint.vocab_size, pad_id=pad_id,
        expected_count=expected_count,
    )

    # D. Pairing / policy (pluggable).
    pairing_stats: dict = {}
    opted_out = False
    if pairing_validator is not None:
        pairing_stats = pairing_validator(spec, train_trajectories, eval_trajectories) or {}
    elif requires_pairing:
        raise AssertionError(
            f"Builder {builder_name!r} requires a policy validator but supplied none. "
            f"Set requires_pairing=False to opt out explicitly (and document why)."
        )
    else:
        opted_out = True

    # F. Probe contamination (pluggable, optional). The validator labels each held-out probe and
    #    raises if an `expect_heldout` probe is leaked by the trajectories.
    contamination_stats: dict = {}
    if contamination_validator is not None:
        contamination_stats = contamination_validator(train_trajectories, eval_trajectories) or {}

    # Visibility (not a guarantee): a filtering source (e.g. the atomic-link control) keeps "up to"
    # N per context, silently shrinking + biasing the surviving set. Surface requested-vs-kept and the
    # per-context yield so a shrunk/biased dataset is never silent. We do NOT enforce a count here —
    # dropping is intended; this just makes the shortfall legible in the manifest.
    train_per_ctx = Counter(t.x0_id for t in train_trajectories)
    yields = sorted(train_per_ctx.values())
    requested = (spec.num_contexts * spec.trajectories_per_context
                 if getattr(spec, "num_contexts", None) is not None else None)
    coverage = {
        "requested_train_traj": requested,
        "kept_train_traj": len(train_trajectories),
        "train_contexts_requested": getattr(spec, "num_contexts", None),
        "train_contexts_realized": len(train_per_ctx),
        "traj_per_ctx_min": (yields[0] if yields else 0),
        "traj_per_ctx_max": (yields[-1] if yields else 0),
        "traj_per_ctx_mean": (round(sum(yields) / len(yields), 2) if yields else 0.0),
    }

    stats = {
        **shape_stats,
        "n_train_traj": len(train_trajectories),
        "n_eval_traj": len(eval_trajectories),
        "n_train_ctx": len({t.x0_id for t in train_trajectories}),
        "n_eval_ctx": len({t.x0_id for t in eval_trajectories}),
        "coverage": coverage,
        "pairing": pairing_stats,
        "pairing_opted_out": opted_out,
        "probe_contamination": contamination_stats,
        "sampler": spec.sampler,
    }

    return TrajectoryDataset(
        spec=spec,
        train_trajectories=train_trajectories,
        eval_trajectories=eval_trajectories,
        tokenizer_fingerprint=tokenizer_fingerprint,
        base_checkpoint_id=base_checkpoint_id,
        builder_name=builder_name,
        config_snapshot=dict(config_snapshot),
        seeds=seeds,
        prompts=dict(prompts or {}),
        stats=stats,
        _gate_token=_GATE_SENTINEL,
    )


# ======================================================================================
# Builder base class + registry. The gate is the hard guarantee; this is the ergonomics.
# ======================================================================================

def _config_to_dict(cfg) -> dict:
    if isinstance(cfg, dict):
        return dict(cfg)
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(vars(cfg))


class DatasetBuilder(abc.ABC):
    """Subclass this to add a new trajectory source.

    Implement the pure hooks (which return raw pieces, never a TrajectoryDataset) and
    optionally `pairing_validator` / `expected_count`. Do NOT override `build()`: it is
    final and is what forces everything through the gate. `register_builder` rejects any
    subclass that overrides `build`.
    """

    name: str = ""
    requires_pairing: bool = False

    # --- hooks an agent implements -----------------------------------------------------
    # `bundle` is the ModelBundle (carries .tokenizer, .peft_model, and the .base()/.baked()
    # adapter-toggle context managers). The kernel treats it as opaque.
    @abc.abstractmethod
    def fingerprints(self, cfg, *, bundle):
        """Return (TokenizerFingerprint, CheckpointId) for the model the bake runs against."""

    @abc.abstractmethod
    def build_generation_spec(self, cfg) -> GenerationSpec:
        ...

    @abc.abstractmethod
    def build_trajectories(self, cfg, *, bundle):
        """Return (train_trajectories, eval_trajectories, prompts_dict).

        Trajectories are either GENERATED (via a TrajectoryGenerator under the right
        adapter state) or LOADED from a cache and re-validated. The split into train/eval
        contexts happens here and is what makes assert_context_disjoint meaningful.
        """

    def pairing_validator(self) -> Optional[Callable]:
        return None

    def contamination_validator(self) -> Optional[Callable]:
        """Optional probe<->trajectory contamination check (criterion F). A builder that declares
        HELD-OUT eval probes returns a closure `validator(train, eval) -> dict` (typically capturing
        the tokenizer + probe bank + chain stashed during build_trajectories); the default opts out."""
        return None

    def expected_count(self, cfg) -> Optional[int]:
        return None

    # --- FINAL: do not override --------------------------------------------------------
    def build(self, cfg, *, bundle, data_seed, gen_seed=None, model_seed=None):
        seed_everything(data_seed)
        self.data_seed = data_seed
        self.gen_seed = data_seed if gen_seed is None else gen_seed
        tok_fp, ckpt_id = self.fingerprints(cfg, bundle=bundle)
        spec = self.build_generation_spec(cfg)
        train_t, eval_t, prompts = self.build_trajectories(cfg, bundle=bundle)
        seeds = SeedBundle(
            data_seed=data_seed,
            model_seed=data_seed if model_seed is None else model_seed,
            gen_seed=self.gen_seed,
        )
        # For trajectories LOADED from cache, the builder records the checkpoint that actually
        # GENERATED them (self._generation_checkpoint) so criterion E compares stored-vs-current
        # rather than trivially comparing current-vs-current. Fresh generation leaves it None ->
        # falls back to the current checkpoint (which did generate them).
        gen_ckpt = getattr(self, "_generation_checkpoint", None) or ckpt_id
        return run_validation_gate(
            spec=spec, train_trajectories=train_t, eval_trajectories=eval_t,
            tokenizer_fingerprint=tok_fp, base_checkpoint_id=ckpt_id,
            builder_name=self.name, config_snapshot=_config_to_dict(cfg), seeds=seeds,
            prompts=prompts, pairing_validator=self.pairing_validator(),
            requires_pairing=self.requires_pairing,
            contamination_validator=self.contamination_validator(),
            expected_count=self.expected_count(cfg),
            generation_checkpoint_id=gen_ckpt, generation_tokenizer_fp=tok_fp,
        )


_BUILDER_REGISTRY: dict = {}


def register_builder(cls):
    """Class decorator. Registers a DatasetBuilder and forbids bypassing the gate."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty `name`.")
    if cls.name in _BUILDER_REGISTRY:
        raise ValueError(f"Duplicate builder name {cls.name!r}.")
    if cls.build is not DatasetBuilder.build:
        raise TypeError(
            f"{cls.__name__} overrides build(); the validation gate must not be bypassed. "
            f"Implement fingerprints/build_generation_spec/build_trajectories instead."
        )
    _BUILDER_REGISTRY[cls.name] = cls
    return cls


def get_builder(name: str):
    if name not in _BUILDER_REGISTRY:
        raise KeyError(f"Unknown builder {name!r}. Known: {sorted(_BUILDER_REGISTRY)}")
    return _BUILDER_REGISTRY[name]


def list_builders() -> list:
    return sorted(_BUILDER_REGISTRY)


def build_dataset(name: str, cfg, *, bundle, data_seed, gen_seed=None, model_seed=None):
    """Resolve a registered builder and run it through the gate."""
    return get_builder(name)().build(
        cfg, bundle=bundle, data_seed=data_seed, gen_seed=gen_seed, model_seed=model_seed,
    )
