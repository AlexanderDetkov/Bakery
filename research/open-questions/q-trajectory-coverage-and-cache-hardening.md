---
title: Trajectory coverage of the axiom set + framework hardening (code review 2026-06-07)
status: parked
priority: high
created: 2026-06-07
hypothesis: >
  A code review surfaced one validity issue that affects the propagation HEADLINE and four framework
  bugs that do not affect the current separate-process logic-world runs but matter for correctness /
  sweeps / reproducibility. The user chose to DEFER all of these ("keep it simple, just remember") and
  proceed with the current setup, caveated. This note is the durable record so nothing is lost.
acceptance_criteria: >
  When revisited: (COVERAGE) every taught edge appears in >=1 train trajectory, enforced by a gate
  criterion that hard-fails otherwise; re-run the headline and confirm baked d'-vs-depth changes vs the
  partial-coverage runs. (HARDENING) a sweep of >=2 points in one process yields independent models
  (P0 fixed); a cache generated under a different revision/dtype is NOT reused (P1 fixed); negatives no
  longer carry a `proof_depth` field (P2a); truncated-collision contexts cannot leak train<->eval (P2b).
experiment: >
  Deferred. Fix sketches below.
links: [[q-graph-structure-diamonds-multipremise]], [[q-grokking-converse-via-longer-training]], [[q-sft-vs-bake-reversal-curse]]
---

## The issue that affects the headline (deferred, caveated)
**Coverage — baking trains on a PARTIAL axiom set.** The atomic source samples the prompted model over
"complete the rule: Every X is a ___" contexts; multi-parent atoms get only one parent elicited.
Measured on `lw_alpha`: **27/38 edges covered, 11 never stated** (e.g. `Timuva→Doka` ✓×4 but
`Timuva→{Vofupa,Tuvu}` ✗; `Sokezo`, `Fesa` wholly missing). Prompting has all 38 rules in context, so:
- **prior + prompted d′ arms are UNAFFECTED** (no trajectories involved) — prompt-propagation results valid.
- **baked + SFT arms are coverage-confounded** — a baked miss may be "edge never trained," not "didn't
  propagate." Any bake-vs-prompt headline MUST carry this caveat until coverage is fixed.

Fix options (when revisited): list-ALL-parents contexts + a multi-edge-aware atomic filter (keep several
atomic edges from one source; still drop composed/reverse/cross) + oversample-until-covered + teacher-force
stragglers, and/or a teacher-forced **canonical** mode (one trajectory per rule, supervised y = "Every X is
a Y."). Enforce with a **coverage gate criterion** (every edge stated >=1× in train, else hard-fail).

## UPDATE 2026-06-07: the four framework bugs are now FIXED (coverage still deferred)
Per "fix the bugs but don't guarantee baking sees every rule": P0/P1/P1b/P2a/P2b are implemented +
tested (98 fast tests, smoke green); the COVERAGE guarantee is intentionally NOT added (still parked).
- P0 — `peft_factory` now caches the tokenizer only and loads a FRESH base model per `build_bundle`
  (test_framework_hardening: model loader not memoized).
- P1 — `CacheIdentity` gained a `checkpoint` field (+ schema_version 2, invalidating old caches);
  `checkpoint_key()` folds in revision/dtype/adapter; the cache stores `generation_checkpoint` and the
  gate re-checks it (criterion E now meaningful for cached data).
- P1b — the gate records `stats["coverage"]` (requested-vs-kept + per-context yield) — visibility, not
  enforcement (confirmed in smoke).
- P2a — negatives now carry `proof_depth=null` + `match_depth`; `dprime._depth` reads `match_depth`
  (backward-compatible fallback to proof_depth/hop).
- P2b — contexts are truncated BEFORE dedup.
NOTE: the live (coverage-limited) runs hold the pre-edit code in memory, so they finish on old behavior
(unaffected; none of these bugs changed their numbers). On-disk `lw_*` probe banks keep the old
proof_depth schema until the next regeneration — deliberately NOT regenerated now to avoid a torn read
crashing the live runs (the dprime fallback reads old banks correctly). Only COVERAGE remains open below.

## Framework bugs (FIXED 2026-06-07 — see update above; originally flagged here)
- **P0 — sweep model-cache contamination.** `peft_factory._load_base` is `@lru_cache`; `get_peft_model` /
  `merge_and_unload` mutate that base object in place; `runner.main` runs sweep points in ONE process
  (`run()` loop), so later `--sweep` runs inherit an earlier run's mutated/trained model → sweep diffs can
  look like signal. Separate `python run.py` invocations are unaffected (fresh process → fresh cache).
  Fix: load fresh per run (or deepcopy a pristine copy / clear the cache); cache tokenizer+config only.
- **P1 — cache provenance vs the "same checkpoint" invariant.** `CacheIdentity` keys on model NAME (via
  tokenizer_id) but omits revision/dtype/adapter; cached rows store no generation checkpoint; the gate is
  passed the CURRENT checkpoint as `generation_checkpoint_id`, so criterion E is trivially satisfied for
  cached data. Fix: put full `CheckpointId` in the cache key, store it in the cache header, and pass the
  STORED id to the gate so E actually compares.
- **P2a — `proof_depth` on negatives.** `make_logic_world._probe` writes `proof_depth=d` for non-provable
  probes, where d is a matched CONTROL depth, not a proof depth. Statistically fine (the d′ pairing is
  intentional) but the schema overclaims. Fix: `proof_depth=null` for negatives + a separate `match_depth`;
  `dprime._depth` reads `match_depth` for non-provable.
- **P2b — context dedup before truncation.** `_load_contexts` dedups raw text then truncates; disjointness
  is by `x0_id` not text, so two contexts that collide after truncation could leak train<->eval. Doesn't
  bite the short logic contexts now. Fix: truncate→dedup, and also disjoint by normalized text in
  `assert_context_disjoint`.

## Decision (2026-06-07)
User: "for now lets just remember this aspect of training and keep it simple and not add anything."
→ No code change now. Current runs proceed with partial coverage; prompt-propagation is the clean
headline, bake/SFT reported with the coverage caveat. Revisit this note before any strong bake-vs-prompt
claim. See harness memory `trajectory-coverage-deferred`.
