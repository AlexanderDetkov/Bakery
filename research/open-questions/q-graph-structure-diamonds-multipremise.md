---
title: Beyond chains — convergent (diamond) and multi-premise (conjunctive) proof structures
status: open
priority: high
created: 2026-06-06
hypothesis: >
  The current logic worlds (data/worlds/lw_*.json) are forests of SINGLE-INHERITANCE TREES: every concept
  has ≤1 parent (verified: max in-degree 1), all rules are single-premise (body size 1), and 0 concepts are
  reachable by more than one path. So every positive entailment "is every X a Z?" is a single PATH up one
  tree — chained modus ponens, one step per edge. Branching (out-degree) and the 4 disjoint components are
  used only to build hard NEGATIVES (sibling/missing-edge, cross-component), not to make the positive
  reasoning more than a chain. Two structural generalizations test reasoning a chain cannot:

  (1) CONVERGENT DAGs ("diamonds") — a conclusion reachable by ≥2 distinct paths of DIFFERENT lengths. Tests
      whether single-pass propagation tracks the SHORTEST proof (the engine's minimal depth) and whether a
      redundant short alternate path EXTENDS effective propagation — a lever distinct from trajectory coverage.
  (2) MULTI-PREMISE rules ("everything that is both an X and a Y is a Z") — proofs become TREES, depth = tree
      HEIGHT, and the conclusion needs BOTH premises held at once. Tests genuine 2-fact composition, not chaining.

  Predictions: baking's single-pass propagation is SHALLOWER for multi-premise conclusions than for
  equal-depth chains (holding two facts simultaneously in one forward pass is harder than chaining one), with
  the gap widening with depth; diamonds with a short alternate path RAISE the reachable depth toward the
  shortest-path length (redundancy helps), separating "depth of the fact" from "depth of the available proof".
acceptance_criteria: >
  d′-vs-proof-depth (the existing bias-immune metric, generality-matched negatives) on worlds containing
  diamonds and multi-premise rules, for prior/prompted/soft-bake/one-hot-SFT.
  DECISIVE-DIAMOND: for a target reachable by a long path (len L) and a short path (len S<L), propagation
  distance tracks S (minimal proof), not L — i.e. baked/prompted d′ at that target matches depth-S targets,
  not depth-L ones; adding the short alternate edge to a previously-deep target RAISES its d′.
  DECISIVE-MULTIPREMISE: multi-premise conclusions show LOWER d′ than single-premise chain conclusions at the
  SAME proof depth, gap widening with depth (CIs across worlds exclude 0). NULL: multi-premise ≈ chain at
  equal depth → the model composes premises as cheaply as it chains.
experiment: >
  Extend scripts/make_logic_world.py and bakery/logic/, keeping the gate + d′ metric unchanged:
  (a) DIAMONDS — cheap, engine ALREADY supports it. ProofEngine.query is shortest-path BFS over a general DAG
      and the branching_shortest_path self-test already covers a 2-path target. So this is a GENERATOR-ONLY
      change: let some concepts have ≥2 parents such that a target is reachable by paths of unequal length;
      add a 5th world (e.g. lw_epsilon) so it does not disturb runs in flight. Add a probe tag (e.g.
      alt_path_len) so analysis can compare "minimal-depth d vs longest-path d" at the same target.
  (b) MULTI-PREMISE — medium effort. The data model is ready (Rule(body, head) allows a multi-atom body), but
      ProofEngine.__init__/query currently build adjacency from single-body rules only and BFS single edges;
      extend to a forward-chaining SATURATION that fires a rule when ALL body atoms are derived, with
      depth(head) = max(depth(premise_i)) + 1 (proof-tree height). Update brute_force_reachability + the
      self-test to an independent saturation oracle so the cross-check still holds on every pair. The
      contamination guard's states_beyond_atomic needs a multi-premise notion of "one atomic step", and a new
      negative type "withhold-one-premise" (state X and the rule but not Y, so "is the X-and-Y a Z?" is a
      genuine non-theorem). Render multi-premise rules in the prompt as
      "Everything that is both an X and a Y is a Z." Keep single-premise worlds as the chain baseline.
links: [[q-grokking-converse-via-longer-training]], [[q-sft-vs-bake-reversal-curse]]
---

## Why this matters
The headline result so far (single-pass prompt propagation is shallow; baking vs SFT vs prompting on
proof-depth) is measured on pure chains. "Knowledge propagation" in real models surely involves convergent
evidence (many routes to a conclusion) and multi-premise composition (combining two known facts). These two
extensions are the minimal, controlled way to test those without leaving the verified-proof-engine framework:
proof depth stays exactly defined (shortest path / proof-tree height), labels stay engine-verified, and the
d′ instrument is unchanged.

## Effort / sequencing
- **Diamonds first** (low cost, no engine change, no disruption to current runs): generator + a 5th world +
  an `alt_path_len` probe tag + an analysis slice. Good immediate next round.
- **Multi-premise second** (the deferred Tier-2 work): engine saturation forward-chainer + oracle + guard +
  generator + new negative type. Schedule after the chain headline + diamonds land.

## Guardrails (unchanged)
Build data through the gate; never hand-construct TrajectoryDataset; the proof-engine self-test (engine vs an
INDEPENDENT oracle, every pair) must be extended to cover diamonds and multi-premise, or the labels are not
trustworthy. d′ negatives stay generality-matched (see [[q-grokking-converse-via-longer-training]] log S3c).
