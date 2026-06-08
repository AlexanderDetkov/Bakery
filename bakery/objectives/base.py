"""Baking objectives — the variant seam (bake / pursue / knowledge / half-bake).

Each objective is a small registered unit the generic runner calls without branching. The
ONE place the KL is computed is `supervised_kl_terms` / `aligned_kl`: it obtains the base
(teacher) logits via the SAME peft model with the adapter toggled, the baked (student)
logits with the adapter on, and computes the divergence on EXACTLY the gate-validated
supervised tokens. Teacher/student roles are NAMED (not positional), so the KL direction
cannot be flipped; the full vocab is used (no top-k); the logit->token shift is applied
once here. An objective that misaligns base vs baked would have to defeat the gate first.
"""

from __future__ import annotations

import abc

import torch
import torch.nn.functional as F


# ======================================================================================
# Batching: FramedTrajectory -> right-padded tensors (training forward passes).
# ======================================================================================

def _pad_ids(seqs, pad_id):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)     # RIGHT pad
        attn[i, : len(s)] = 1
    return ids, attn


def _pad_mask(masks):
    L = max(len(m) for m in masks)
    out = torch.zeros((len(masks), L), dtype=torch.bool)
    for i, m in enumerate(masks):
        out[i, : len(m)] = torch.tensor(m, dtype=torch.bool)
    return out


def collate_framings(trajs, pad_id, traj_keys=None) -> dict:
    """Collate FramedTrajectory objects into base/baked padded tensors + supervised masks.

    `traj_keys` (optional) are stable per-trajectory identities used ONLY by the opt-in teacher-logit
    cache to key its memoized teacher log-probs; when None (the default, and always at eval) nothing is
    cached and the dict is functionally the historical one. `num_sup` is recorded so the cache can split
    the flattened [N_sup, V] teacher block back into per-trajectory blocks (it is unused otherwise)."""
    base_ids, base_attn = _pad_ids([t.base_input_ids for t in trajs], pad_id)
    baked_ids, baked_attn = _pad_ids([t.baked_input_ids for t in trajs], pad_id)
    out = {
        "base_ids": base_ids, "base_attn": base_attn,
        "base_sup": _pad_mask([t.base_sup_mask for t in trajs]),
        "baked_ids": baked_ids, "baked_attn": baked_attn,
        "baked_sup": _pad_mask([t.baked_sup_mask for t in trajs]),
        "num_sup": [int(t.num_supervised) for t in trajs],
    }
    if traj_keys is not None:
        out["traj_keys"] = list(traj_keys)
    return out


# ======================================================================================
# The ONE KL primitive.
# ======================================================================================

def _sup_pred_logprobs(logits, sup_mask):
    """Log-softmax of the logits that PREDICT each supervised token.

    logit at position p-1 predicts the token at position p; a supervised token at p is thus
    scored by logits[:, p-1]. We select logits[:, :-1] at positions where sup_mask[:, 1:] is
    True, flatten in (row, position) order, and log-softmax over the FULL vocab in float32.
    """
    pred = logits[:, :-1, :]            # [B, L-1, V] — position j predicts token j+1
    sel = sup_mask[:, 1:]               # [B, L-1] bool, aligned to `pred`
    chosen = pred[sel]                  # [N_sup, V]
    return F.log_softmax(chosen.float(), dim=-1)


def _sup_target_ids(input_ids, sup_mask):
    """Target token ids aligned 1:1 (same order) with `_sup_pred_logprobs(_, sup_mask)`.

    A supervised token at position p is predicted by the logit at p-1; `_sup_pred_logprobs`
    selects those logits with `sup_mask[:, 1:]`. The matching *target* is the token at p, i.e.
    `input_ids[:, 1:]` under the SAME mask. Defining it here (next to the log-prob selector)
    keeps the one logit->token shift in a single place, so SFT cross-entropy reuses it instead
    of re-rolling the shift.
    """
    sel = sup_mask[:, 1:]               # [B, L-1] bool, aligned to input_ids[:, 1:]
    return input_ids[:, 1:][sel]        # [N_sup] target token ids, same row-major order


class TeacherLogitCache:
    """Opt-in memoization of the teacher's full-vocab float32 log-prob block per trajectory.

    The teacher (frozen base weights, adapter DISABLED, fixed prompt, fixed trajectory tokens) is
    MATHEMATICALLY a pure function of fixed inputs, so caching its log-probs once and reusing them avoids
    recomputing the (expensive, long-prompt) teacher forward every epoch — the bulk of the per-step cost.

    NOT bit-exact to the cache-off baseline (a PRAGMATIC speedup, default off): the baseline recomputes
    the teacher in a PADDED batch each epoch, and right-padding (hence the float-level result) varies with
    the epoch's shuffle/batchmates — so even the baseline's own teacher targets drift epoch-to-epoch at
    float scale. The cache pins one value, perturbing training by that SAME kind/scale of float-noise the
    baseline already carries. `verify_every>0` recomputes the teacher at intervals and HARD-FAILS if the
    cached block deviates beyond `verify_atol` (catches gross cache bugs / real non-determinism while
    tolerating padding float-noise). `put` asserts `last-dim == vocab_size`, so the full-vocab / no-top-k
    guarantee is preserved structurally. Refused for objectives whose teacher changes per epoch (pursue).
    Whether the baked model it learns matches the baseline is an EMPIRICAL question — see
    research/decisions/teacher-logit-cache.md.
    """

    def __init__(self, backend="cpu", verify_every=0, dtype="float32", verify_atol=1e-2):
        if dtype != "float32":
            raise ValueError("cache_teacher_dtype must be 'float32' (matches the live float32 log_softmax).")
        if backend not in ("cpu", "gpu"):
            raise NotImplementedError(
                f"teacher cache backend {backend!r} not implemented; use 'cpu' or 'gpu' (or 'off')."
            )
        self.backend = backend
        self.verify_every = int(verify_every)
        self.verify_atol = float(verify_atol)
        self._store: dict = {}
        self._reads = 0

    def has(self, key) -> bool:
        return key in self._store

    def put(self, key, block, vocab_size) -> None:
        if int(block.shape[-1]) != int(vocab_size):
            raise RuntimeError(
                f"teacher cache block last-dim {block.shape[-1]} != vocab_size {vocab_size}; the cache "
                f"must store the FULL vocab (the no-top-k guarantee)."
            )
        dest = "cpu" if self.backend == "cpu" else block.device
        self._store[key] = block.detach().to(device=dest, dtype=torch.float32).contiguous()

    def get(self, key, device):
        return self._store[key].to(device=device, dtype=torch.float32)

    def should_verify(self) -> bool:
        if self.verify_every <= 0:
            return False
        self._reads += 1
        return (self._reads % self.verify_every) == 0


def supervised_kl_terms(bundle, batch, *, base_with_adapter=False, device="cpu", teacher_cache=None):
    """Per-supervised-token KL( teacher ‖ student ), returned as a 1-D tensor [N_sup].

    teacher = base-model-with-prompt   (adapter DISABLED, unless base_with_adapter -> pursuit)
    student = baked-model-without-prompt (adapter ENABLED)

    The teacher forward is always under no_grad (the target is detached / stop-gradient). The
    student forward carries grad when called outside a no_grad context (training); eval wraps
    the whole call in no_grad. Because the batch came from a gate-validated TrajectoryDataset,
    base_sup and baked_sup select the SAME tokens in the same order, so the two [N_sup, V]
    tensors align element-wise (a count mismatch would raise loudly).

    `teacher_cache` (opt-in, default None) memoizes the teacher log-probs across epochs — see
    `TeacherLogitCache`. It is used ONLY when not base_with_adapter (the pursuit teacher changes each
    epoch) and only for batches carrying `traj_keys`; the student is ALWAYS recomputed live. With it None
    the function is byte-identical to the historical recompute-every-step path.
    """
    base_ids = batch["base_ids"].to(device)
    base_attn = batch["base_attn"].to(device)
    base_sup = batch["base_sup"].to(device)
    baked_ids = batch["baked_ids"].to(device)
    baked_attn = batch["baked_attn"].to(device)
    baked_sup = batch["baked_sup"].to(device)

    vocab_size = int(bundle.tokenizer_fingerprint.vocab_size)

    def _compute_teacher_t():
        teacher_ctx = bundle.baked if base_with_adapter else bundle.base
        with torch.no_grad():
            with teacher_ctx() as m:
                # The teacher is a FIXED target. Force eval mode for its forward so a non-zero
                # lora_dropout (or any train-mode stochasticity) cannot inject noise into the target;
                # restore the prior mode so the student trains normally.
                was_training = m.training
                m.eval()
                teacher_logits = m(input_ids=base_ids, attention_mask=base_attn).logits
                if was_training:
                    m.train()
        return _sup_pred_logprobs(teacher_logits, base_sup)   # [N, V] log-probs (detached, float32)

    cacheable = (teacher_cache is not None) and (not base_with_adapter)
    keys = batch.get("traj_keys")
    num_sup = batch.get("num_sup")

    if cacheable and keys is not None and all(teacher_cache.has(k) for k in keys):
        t = torch.cat([teacher_cache.get(k, device) for k in keys], dim=0)   # per-traj blocks in order
        if int(t.shape[-1]) != vocab_size:
            raise RuntimeError("cached teacher block is not full-vocab (no-top-k guarantee violated).")
        if teacher_cache.should_verify():
            live = _compute_teacher_t()
            atol = teacher_cache.verify_atol
            if not torch.allclose(t, live, atol=atol, rtol=0.0):
                maxdiff = float((t - live).abs().max())
                raise RuntimeError(
                    f"teacher-logit cache VERIFY FAILED: cached teacher log-probs deviate from a fresh "
                    f"recompute by {maxdiff:.3g} > atol {atol:.3g} (gross cache bug or genuine "
                    f"non-determinism beyond padding float-noise). Set train.cache_teacher_logits=off "
                    f"or raise train.cache_teacher_verify_atol if this is expected float-noise."
                )
    else:
        t = _compute_teacher_t()
        if cacheable and keys is not None and num_sup is not None:
            off = 0                                          # split [N,V] into per-traj blocks (row order)
            for k, n in zip(keys, num_sup):
                teacher_cache.put(k, t[off:off + n], vocab_size)
                off += n
            if off != t.shape[0]:
                raise RuntimeError(
                    f"teacher cache split mismatch: sum(num_sup)={off} != N_sup={t.shape[0]}."
                )

    with bundle.baked() as m:
        student_logits = m(input_ids=baked_ids, attention_mask=baked_attn).logits

    s = _sup_pred_logprobs(student_logits, baked_sup)    # [N, V] log-probs
    if t.shape != s.shape:
        raise RuntimeError(
            f"Supervised-token mismatch between base ({t.shape}) and baked ({s.shape}) framings; "
            f"the gate should have prevented this."
        )
    return (t.exp() * (t - s)).sum(dim=-1)               # [N] KL(teacher ‖ student) per token


def aligned_kl(bundle, batch, *, base_with_adapter=False, device="cpu", teacher_cache=None):
    """Mean supervised-token KL — the training loss / the eval-KL numerator."""
    return supervised_kl_terms(
        bundle, batch, base_with_adapter=base_with_adapter, device=device, teacher_cache=teacher_cache
    ).mean()


# ======================================================================================
# Objective ABC + registry.
# ======================================================================================

class Objective(abc.ABC):
    name: str = ""
    sampler: str = "base_disable_adapter"      # which GenerationSpec.sampler this objective requires
    needs_per_epoch_trajectories: bool = False  # pursuit regenerates each epoch

    @abc.abstractmethod
    def compute_loss(self, *, bundle, batch, cfg, teacher_cache=None):
        """Return the scalar loss tensor (grad on the ENABLED adapter only). The runner
        owns backward / grad-accum / clip / step. `teacher_cache` (opt-in, default None) is the
        TeacherLogitCache; objectives that distill from the base teacher thread it into aligned_kl,
        others ignore it."""


_OBJECTIVES: dict = {}


def register_objective(cls):
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty `name`.")
    if cls.name in _OBJECTIVES:
        raise ValueError(f"Duplicate objective {cls.name!r}.")
    _OBJECTIVES[cls.name] = cls
    return cls


def get_objective(name) -> Objective:
    if name not in _OBJECTIVES:
        raise KeyError(f"Unknown objective {name!r}. Known: {sorted(_OBJECTIVES)}")
    return _OBJECTIVES[name]()


def list_objectives() -> list:
    return sorted(_OBJECTIVES)
