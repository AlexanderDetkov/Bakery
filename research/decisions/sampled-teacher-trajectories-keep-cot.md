---
title: theorem_qa bakes on-policy SAMPLED teacher trajectories (CoT tail kept); contamination check F is recorded, not enforced, when sampling
date: 2026-06-07
status: accepted
decided_by: user (adetkov)
affects: bakery/trajectories/theorem_qa.py (sample_trajectories), tests/test_theorem_qa.py
---

## Decision
`theorem_qa` now produces the supervised answer `y` for the BAKE arm by **sampling from the prompted
teacher** (`base + u`, adapter OFF) on each relation's question — canonical on-policy prompt baking,
matching `squad_qa` and the methodology in CLAUDE.md (sample trajectories from the prompted base; KL on
the generated tokens). Teacher-forcing the ground-truth answer remains available behind
`data.sample_trajectories=False` (and is always used for `objective=sft`, which needs one-hot targets).

The sampled teacher **chains in its free generation** — with all axioms in context it recites the
axiom path ("...every Nove is a Timuva, every Timuva is a Doka..."), so the training answers STATE
held-out transitive consequences (measured: ~40 held-out probes stated at 1B). The user's explicit
call: **keep it simple, keep the CoT tail, don't engineer truncation/filtering around the leaks.**

Consequently theorem_qa's own **criterion F** (probe-contamination) is **relaxed for the sampled path**:
it RECORDS the contamination in `stats["probe_contamination"]` (`enforced: false`, `sampled_cot_leak:
true`, `n_heldout_stated`) instead of HARD-FAILING. Teacher-forced still hard-fails on any leak.

## How trajectories are generated (teacher-GENERATED, never hand-written)
Both trajectory types are produced by the MODEL's `generate()` under `bundle.base()` (adapter OFF) —
**nothing is hand-authored.** The only difference is the prompt prefix the teacher conditions on.

- **Main (bake) trajectories** — `theorem_qa._sample_frames`: for each enumerated relation build the
  question `Q = phrasing.question(world, X, Z)`, condition the base model on the PROMPTED prefix
  `build_prefix_ids(tok, u, Q)`, and SAMPLE the answer — `with bundle.base(): y =
  generator.generate(prompted_prefix, gen_cfg)`. Teacher = **base + u**. base framing = `[sys=u]+Q+y`,
  baked framing = `[sys=∅]+Q+y`, supervised span = the sampled `y`.
- **Regularized (anchor) trajectories** — `regularization.build_anchor_trajectories`: for each
  irrelevant (SQuAD) context condition the base model on the EMPTY prefix `build_prefix_ids(tok, "",
  ctx)` and SAMPLE — `with bundle.base(): y = generator.generate(...)`. Teacher = **base, NO prompt**.
  base framing == baked framing == `[sys=∅]+ctx+y`, supervised span = `y`.

In both, `y` is the model's OWN sampled tokens; provenance (`sampler_kind`, `do_sample`, temperature,
seed, `trajectories_per_context`) is recorded in the manifest. The ONLY hand-written path is the
teacher-forced FALLBACK (`data.sample_trajectories=False`), kept solely for the SFT arm and for
reproducing the early runs — the default bake path never hand-writes answers.

## Why this is allowed (integrity)
- Criterion F is a **builder-specific, pluggable** check (`contamination_validator`), NOT one of the 5
  universal validity criteria and NOT the universal gate. The universal gate (E identical checkpoint,
  A mask alignment, B context disjointness, C completeness) and the KL primitive are **untouched**.
- Validity criterion 4 (held-out eval CONTEXTS disjoint from trajectory-generation contexts) still
  holds: trajectories are generated on the TRAINED relations' questions; the held-out probe questions
  are never used as generation contexts. F is a stronger *content*-level guard; only it is relaxed.
- The invariant test `test_theorem_qa_contamination_validator_hard_fails_on_leak` is unchanged and
  still passes (teacher-forced path); a sibling test pins the sampled record-only behavior. No test was
  weakened to pass.
- Per the "if you think an invariant is wrong, write a decisions note" rule, this note IS that record.

## Scientific caveat (must carry into any finding)
Under sampling, held-out `d′` at proof-depth ≥ 2 is **partly recall-of-recited**, not pure propagation,
because the teacher's CoT recitation states many held-out consequences in training. Interpret the
sampled-arm held-out cells as "did baking internalize the consequences the teacher recited," not "did
an unseen consequence emerge." For a clean propagation measurement, use the teacher-forced arm
(`data.sample_trajectories=False`). `n_heldout_stated` in the manifest quantifies the contamination per
run. The chaining itself is a real (re)observation that the prompted model propagates in generation
(cf. [[cot-chains-baked-rules-but-not-the-converse]]).

## Reversibility
`data.sample_trajectories=False` restores the teacher-forced, hard-fail, clean-held-out behavior with
no code change.
