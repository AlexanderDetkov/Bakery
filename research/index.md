# Research index

The catalog. Orient from this file (and the agenda) rather than scanning directories. Keep current
(or run `/lint-research`).

## Theme: knowledge propagation under prompting vs baking
How far does an injected fact propagate to its n-hop consequences, and does baking (fact → LoRA)
propagate as far as prompting (fact in context)? Instrument: the `bake_fact` experiment +
`fact_propagation` builder + `propagation` metric (no-CoT forced-choice belief shift, per hop, for
prior vs prompted vs baked, all on ONE checkpoint). Assets: `data/prompts/tsunami_u.md`,
`data/contexts/tsunami_contexts.json`, `data/probes/tsunami_probes.json`.

## Open questions
- [[q-propagation-trajectory-type]] — open (high) — does context TYPE shape the curve? (cycle-1: on/off-topic GATE confirmed; finer ordering needs count+convergence match)
- [[q-propagation-hardening]] — open (high) — is C1 (gap grows with hops) real with CIs across facts/seeds? (≥2 facts, ≥3 seeds, ≥12 probes/hop)
- [[q-propagation-trajectory-size]] — open (high) — does propagation depth scale with #trajectories?
- [[q-propagation-model-scale]] — open (medium) — controlled one-family size sweep (matched LoRA params, normalized retention)
- [[q-propagation-cot-confound]] — open (medium) — does CoT inflate apparent propagation vs the no-CoT readout?

## Resolved questions
- [[q-propagation-prompt-vs-bake]] — resolved (cycle 1) → [[propagation-bounded-by-trajectory-coverage]]

## Findings
- [[propagation-bounded-by-trajectory-coverage]] — **baking only injects what the trajectories exercise; eval_kl ⟂ propagation** (C3 confirmed high; on/off-topic gate robust; C1/C4 suggestive single-run). Runs: prop-tsunami-{1b,8b}-mixed, prop-tsunami-1b-{restate,consequence,neutral}-m12.

## Decisions
_(none yet)_
