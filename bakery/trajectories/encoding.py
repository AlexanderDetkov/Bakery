"""The atomic trajectory unit and the single source of truth for its supervised span.

A baking trajectory is one sampled continuation y over a context x0, carried in BOTH
framings the KL needs:
  * base framing  = base-model-WITH-prompt    : system+u + x0 + y
  * baked framing = baked-model-WITHOUT-prompt : (empty) + x0 + y   (usually)

We store ONLY token ids + a boolean "supervised" mask marking the generated tokens y in
each framing. We deliberately store NO logits: base and baked logits are recomputed at
train time from the SAME peft model by toggling the adapter, which (a) is cheap on disk
and (b) makes "full-vocab, not top-k" un-violatable — there is nowhere to put a truncated
distribution.

The gate (trajectories/base.py) guarantees the tokens under `base_sup_mask` are identical
to those under `baked_sup_mask`: the SAME y in both framings. `iter_supervised_ids` is the
ONLY place code reads the supervised tokens; never re-derive mask offsets by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

# logit-at-position-t predicts token-at-position-(t+1). A supervised token at position p is
# therefore predicted by the logit at position p-1; the gate guarantees p >= 1 so this shift
# is always well-defined.
ASSISTANT_SHIFT = 1


@dataclass(frozen=True)
class FramedTrajectory:
    base_input_ids: tuple          # full base-framing sequence (system+u + x0 + y)
    base_sup_mask: tuple           # bool tuple, len == len(base_input_ids); True on generated y
    baked_input_ids: tuple         # full baked-framing sequence ((empty) + x0 + y)
    baked_sup_mask: tuple          # bool tuple, len == len(baked_input_ids); True on generated y
    x0_id: int                     # id of the source context (for the train/eval split check)
    num_supervised: int            # == sum(base_sup_mask) == sum(baked_sup_mask)


def iter_supervised_ids(traj: FramedTrajectory):
    """Return (base_supervised_ids, baked_supervised_ids), aligned.

    The single source of truth for "which generated tokens does this trajectory supervise."
    """
    base = [tok for tok, m in zip(traj.base_input_ids, traj.base_sup_mask) if m]
    baked = [tok for tok, m in zip(traj.baked_input_ids, traj.baked_sup_mask) if m]
    return base, baked
