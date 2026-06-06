# Research index

The catalog. Orient from this file (and the agenda) rather than scanning directories. Keep current
(or run `/lint-research`).

## Theme: knowledge propagation under prompting vs baking
How far does an injected fact propagate to its n-hop consequences, and does baking (fact → LoRA)
propagate as far as prompting (fact in context)? Instrument: the `bake_fact` experiment +
`fact_propagation` builder + `propagation` metric (no-CoT forced-choice belief shift, per hop, for
prior vs prompted vs baked, all on ONE checkpoint). Assets: `data/prompts/tsunami_u.md`,
`data/contexts/tsunami_contexts.json`, `data/probes/tsunami_probes.json`.

## Open questions (enqueued for future sessions; this session hit its 6-cycle budget)
- [[q-propagation-hardening]] — open (high) — are the per-hop/converse magnitudes real with CIs across facts/seeds? (≥2 facts, ≥3 seeds, ≥12 probes/hop, fix degenerate h3 probes)
- [[q-propagation-trajectory-size]] — open (medium) — cycle-2 inconclusive/confounded; needs matched-steps + multi-seed re-run
- [[q-propagation-model-scale]] — open (medium) — controlled one-family size sweep (matched LoRA params, normalized retention)

## Resolved questions
- [[q-propagation-prompt-vs-bake]] — resolved (cycle 1) → [[propagation-bounded-by-trajectory-coverage]]
- [[q-propagation-deductive-chain]] — resolved (cycle 3) → [[baking-is-associative-prompting-is-directional]]
- [[q-propagation-cot-confound]] — resolved (cycle 5) → [[cot-chains-baked-rules-but-not-the-converse]] (corrected free-gen readout)
- [[q-propagation-trajectory-type]] — resolved (cycle 6) → [[trajectory-type-is-a-binary-coverage-gate]]

## ⭐ Capstone
- [[SYNTHESIS-baking-vs-prompting-propagation]] — the full answer (cycles 1–6): baking transfers an ASSOCIATIVE SHADOW of the prompted model over the trajectory distribution.

## Findings
- [[propagation-bounded-by-trajectory-coverage]] — **baking only injects what the trajectories exercise; eval_kl ⟂ propagation** (C3 confirmed high; on/off-topic gate robust; C1/C4 suggestive single-run). Runs: prop-tsunami-{1b,8b}-mixed, prop-tsunami-1b-{restate,consequence,neutral}-m12.
- [[size-helps-fidelity-not-the-propagation-gap]] — **more trajectories lower eval_kl but don't close the prompting–baking gap; eval_kl converges BEFORE belief does** (size→propagation scaling inconclusive: steps-confound + n=4 noise + propagation under-converged at 20 ep). Runs: prop-size-tpc{1,4,8,16}-1b. Figs: results/bake_fact/_fig_by_size.png.
- [[baking-is-associative-prompting-is-directional]] — **baking installs an UNDIRECTED/associative chain (affirms the false converse, −4.75); prompting preserves logical direction (−0.01); 1B propagation weak even prompted** (zero-prior Veld syllogism, no-CoT). Runs: prop-veld-{8b,1b}-mixed. Fig: results/bake_fact/_fig_veld_depth.png.
- [[cot-cue-scoring-is-artifactual]] — **the cycle-4 CoT readout was artifactual** (a cue-only ablation reproduced the "deflation"; rule-verbatim probe +14.3→+2.5 with zero reasoning) → CoT comparison inconclusive; no-CoT results unaffected. Methodological lesson: parse Yes/No from free generation, always run a cue-only ablation.
- [[cot-chains-baked-rules-but-not-the-converse]] — **single-pass baking = all-"Yes" saturation (no discrimination); CoT forward-chains the installed rules (0.50→~0.75) but hallucinates REVERSED rules on the converse** (corrected free-gen readout). Runs: prop-veld-{8b,1b}-mixed. NB: the verification panel itself made a converse sign error — caught by per-probe inspection.
- [[trajectory-type-is-a-binary-coverage-gate]] — **type is a BINARY coverage gate** (on-topic restate/consequence/mixed inject ~+1.35 CIs exclude 0; neutral +0.21 CI includes 0) — NOT graded by reasoning-richness (restate ≈ consequence). Matched count+convergence+CIs. Runs: prop-type2-{restate,consequence,neutral,mixed}-1b.

## Decisions
_(none yet)_
