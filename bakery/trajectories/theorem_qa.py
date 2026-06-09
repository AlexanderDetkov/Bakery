"""Deterministic teacher-forced QA builder for the knowledge-propagation curriculum.

Each trajectory is the prompted teacher ANSWERING a forced-choice question with a full DECLARATIVE
sentence:

    base framing  (system = u):     "In {w}, is every X a Z?"  ->  "Yes, every X is a Z."
    baked framing (system = empty):  "In {w}, is every X a Z?"  ->  "Yes, every X is a Z."

The supervised span is the WHOLE answer — many content tokens, which is where soft-KL baking
installs the relation (a one-token Yes/No is too thin a KL signal; that is the whole reason baking
can beat one-hot SFT). The leading Yes/No keeps it scorable by the existing yes/no `dprime` metric,
which shares the question phrasing via `bakery.logic.phrasing`.

Trajectory source (bake): by DEFAULT the answer y is SAMPLED from the prompted teacher (base+u,
adapter OFF) on each relation's question — canonical on-policy baking (mirrors `squad_qa`). Set
`data.sample_trajectories=False` to TEACHER-FORCE the engine-verified ground-truth answer instead; the
SFT objective always teacher-forces (it needs one-hot ground truth). Teacher-forcing is closer to
SFT-via-KL — it distills along a path the teacher may never take, and for a relation the teacher gets
wrong it injects ground truth the teacher doesn't hold; sampling faithfully measures what the prompted
model's behavior, once baked, transfers. Either way the KL is teacher = base+u vs student = baked, and
the relation ENUMERATION (coverage/curriculum/held-out) is identical — only how y is obtained differs.

Relations are enumerated from the WORLD via the `ProofEngine` (the probe bank only samples a few per
depth, so it does NOT contain every axiom — enumerating from the world is what guarantees depth-1 ==
EVERY axiom). The curriculum knob `train_max_depth` (=n) is swept: depth-1 always fully trained (+
the recall baseline); depths 2..n contribute a balanced sampled subset; the rest + depths > n are
held out (the propagation test). Trained relations are recorded in
`stats["pairing"]["trained_relations"]` so the d′ metric holds them out of the depth>=2 cells.
Negatives (the "No, not..." answers) carry the closed-world closure and are drawn from the SAME
pools as the eval negatives (`bakery.logic.relations`) so a held-out negative type was also seen in
training. Any pair that is an `expect_heldout` eval probe is excluded from training (no leakage;
criterion F is the backstop).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from bakery.config import build_data_config
from bakery.logic import phrasing, relations
from bakery.logic.proof_engine import ProofEngine
from bakery.logic.world import World
from bakery.prompts import build_prefix_ids, load_prompt, sha256
from bakery.trajectories.base import DatasetBuilder, GenerationSpec, register_builder
from bakery.trajectories.contamination import (
    assert_probe_schema_and_balance,
    assert_probes_heldout,
)
from bakery.trajectories.contamination_dag import label_probes_dag
from bakery.trajectories.encoding import FramedTrajectory, iter_supervised_ids

_NEG_TYPES = ("converse", "cross", "missing_edge")


@dataclass
class TheoremQADataConfig:
    world_spec: str = "data/worlds/lw_alpha.json"
    probe_bank: str = "data/probes/lw_alpha_qa_probes.json"   # the d′ eval set (unified question)
    train_max_depth: int = 1            # n: max proof depth trained (depth-1 ALWAYS fully trained)
    per_depth_train_cap: int = 16       # max trained TRUE theorems per depth at depths 2..n (negatives matched)
    split_seed: int = 0                 # train/held-out split is a pure function of (split_seed, depth)
    eval_relations_cap: int = 48        # bounded held-out QA used for eval-KL (bake fidelity)
    sample_trajectories: bool = True    # bake: sample y from base+u (canonical); False -> teacher-force ground truth
    relation_mode: str = "directed"     # "directed" (reachability) | "equivalence" (same-component, symmetric)


# --------------------------------------------------------------------------- relation selection

def _negative_candidates(pools_d) -> list:
    """Flatten a depth's negative pools into (subj, obj, neg_type), in PROBE orientation, deduped by
    pair (a pair can appear under two types, e.g. converse AND cross — keep it once so it can't be
    trained twice). `converse` holds the underlying true pair (x, z); its non-theorem probe is (z, x).
    """
    tagged = ([(z, x, "converse") for (x, z) in pools_d["converse"]]
              + [(x, w, "cross") for (x, w) in pools_d["cross"]]
              + [(x, w, "missing_edge") for (x, w) in pools_d["missing"]])
    out, seen = [], set()
    for x, w, t in tagged:
        if (x, w) not in seen:
            seen.add((x, w))
            out.append((x, w, t))
    return out


def _round_robin(items, count) -> list:
    """Pick up to `count` (subj, obj, neg_type) round-robin across the three negative types, so the
    trained negatives include a MIX of types (matched to the eval negative distribution)."""
    by_t = {t: [r for r in items if r[2] == t] for t in _NEG_TYPES}
    order = [t for t in _NEG_TYPES if by_t[t]]
    out, i = [], 0
    while len(out) < count and order:
        t = order[i % len(order)]
        if by_t[t]:
            out.append(by_t[t].pop())
        else:
            order.remove(t)
            i -= 1
        i += 1
    return out


def _build_relations(world, engine, bank, n, per_depth_cap, split_seed, mode="directed") -> list:
    """Deterministic, (split_seed, depth)-keyed selection of trained relations.

    depth-1: ALL edges (Yes) + matched non-edges (No) -> full coverage of the prompt's axioms.
    depths 2..n: up to `per_depth_cap` Yes (sampled) + matched No. Everything excludes any pair that
    is an `expect_heldout` eval probe (no leakage). Stable across n (per-depth seed), so n=2's depth-2
    training is identical to n=3's.

    `mode` selects directed vs equivalence candidate pools (passed straight to `candidate_pools`); in
    equivalence mode converse/missing pools are empty so the trained negatives are all cross-component.
    """
    heldout = {(p["subj"], p["obj"]) for p in bank if p.get("expect_heldout")}
    # A forward affirm ("Yes, every A is a B") states ONLY (A,B); a deny ("No, not every A is a B")
    # mentions BOTH entities + a reversal cue, so it states the relation in EITHER orientation. So a
    # trained NEGATIVE must avoid a held-out probe with the same UNORDERED pair (both directions); a
    # trained POSITIVE only needs the exact direction (and must NOT be over-excluded, or depth-1
    # coverage of an edge whose REVERSE is a held-out converse probe would break).
    heldout_both = heldout | {(b, a) for (a, b) in heldout}
    max_pool_depth = max(
        int(n),
        *(int(p.get("match_depth") or p.get("proof_depth") or p.get("hop") or 1) for p in bank),
        1,
    )
    pools = relations.candidate_pools(world, engine, max_pool_depth, mode=mode)
    used: set = set()
    trained: list = []
    for d in range(1, int(n) + 1):
        rng = random.Random(split_seed * 1000 + d)            # stable across n
        pos = [(x, z) for (x, z) in pools[d]["true"] if (x, z) not in heldout and (x, z) not in used]
        if d == 1:
            sel_pos = sorted(pos)                              # ALL edges -> coverage by construction
        else:
            rng.shuffle(pos)
            sel_pos = pos[:per_depth_cap]
        for (x, z) in sel_pos:
            used.add((x, z))
            trained.append({"subj": x, "obj": z, "depth": d, "provable": True, "neg_type": None})
        negs = [r for r in _negative_candidates(pools[d])
                if (r[0], r[1]) not in heldout_both and (r[0], r[1]) not in used]
        rng.shuffle(negs)
        for (x, w, t) in _round_robin(negs, len(sel_pos)):
            used.add((x, w))
            trained.append({"subj": x, "obj": w, "depth": d, "provable": False, "neg_type": t})
    return trained


def _eval_relations(bank, cap, seed) -> list:
    """A bounded, balanced sample of HELD-OUT eval probes (expect_heldout) for the eval-KL set —
    disjoint from the trained relations by construction (training excludes expect_heldout pairs)."""
    held = [p for p in bank if p.get("expect_heldout")]
    rng = random.Random(seed)
    true_h = [p for p in held if p.get("provable")]
    false_h = [p for p in held if not p.get("provable")]
    rng.shuffle(true_h)
    rng.shuffle(false_h)
    k = max(cap // 2, 1)
    out = []
    for p in true_h[:k] + false_h[:k]:
        d = p.get("proof_depth") or p.get("match_depth") or p.get("hop") or 2
        out.append({"subj": p["subj"], "obj": p["obj"], "depth": int(d),
                    "provable": bool(p.get("provable")), "neg_type": p.get("neg_type")})
    return out


# --------------------------------------------------------------------------- builder

@register_builder
class TheoremQABuilder(DatasetBuilder):
    name = "theorem_qa"
    requires_pairing = True                   # the coverage/balance criterion is mandatory here

    def _plan(self, cfg) -> dict:
        if getattr(self, "_plan_cache", None) is not None:
            return self._plan_cache
        data_cfg = build_data_config(cfg, TheoremQADataConfig)
        wpath = Path(data_cfg.world_spec)
        if not wpath.exists():
            raise ValueError(f"world_spec {wpath!r} not found.")
        world = World.from_spec(json.loads(wpath.read_text()))
        mode = str(getattr(data_cfg, "relation_mode", "directed"))
        engine = ProofEngine(world, relation_mode=mode)
        bpath = Path(data_cfg.probe_bank)
        bank = json.loads(bpath.read_text()).get("probes", []) if bpath.exists() else []
        n = int(data_cfg.train_max_depth)
        trained = _build_relations(world, engine, bank, n,
                                   int(data_cfg.per_depth_train_cap), int(data_cfg.split_seed),
                                   mode=mode)
        eval_rel = _eval_relations(bank, int(data_cfg.eval_relations_cap), int(data_cfg.split_seed) + 7)
        self._plan_cache = {
            "world": world, "engine": engine, "bank": bank, "n": n, "mode": mode,
            "trained": trained, "eval_rel": eval_rel, "data_cfg": data_cfg,
            "u_text": load_prompt(cfg.generation.base_prompt),
            "baked_text": load_prompt(cfg.generation.baked_prompt),
        }
        return self._plan_cache

    def _use_sampling(self, cfg) -> bool:
        """Bake samples on-policy from the prompted teacher (canonical baking); SFT must teacher-force
        ground truth. `data.sample_trajectories=False` forces teacher-forcing (reproduce the early
        teacher-forced runs, or feed the SFT arm)."""
        if (cfg.train.objective or "bake").lower() == "sft":
            return False
        return bool(getattr(self._plan(cfg)["data_cfg"], "sample_trajectories", True))

    def fingerprints(self, cfg, *, bundle):
        return bundle.tokenizer_fingerprint, bundle.base_checkpoint_id

    def build_generation_spec(self, cfg) -> GenerationSpec:
        g = cfg.generation
        plan = self._plan(cfg)
        base_text, baked_text = plan["u_text"], plan["baked_text"]
        sampling = self._use_sampling(cfg)
        extra = {"sampler_kind": "sampled_qa" if sampling else "teacher_forced_qa",
                 "probe_bank": plan["data_cfg"].probe_bank,
                 "train_max_depth": plan["n"],
                 "relation_mode": plan["mode"],     # criterion 5 provenance: directed vs equivalence
                 "split_seed": plan["data_cfg"].split_seed,
                 "n_trained_relations": len(plan["trained"])}
        reg = getattr(cfg, "regularization", None)
        if reg is not None and int(getattr(reg, "num_train_contexts", 0)) > 0:
            # Provenance for criterion 5 — only recorded when regularization is ON (OFF leaves the
            # spec byte-identical to a non-regularized run).
            extra["regularization"] = {
                "num_train_contexts": reg.num_train_contexts,
                "eval_num_contexts": reg.eval_num_contexts,
                "source": reg.source, "context_split": reg.context_split,
                "max_new_tokens": reg.max_new_tokens, "do_sample": reg.do_sample,
                "seed": reg.seed, "x0_start": 20_000,
            }
        return GenerationSpec(
            sampler="base_disable_adapter",       # base+u (adapter OFF) is the teacher, sampled or forced
            base_prompt_sha256=sha256(base_text),
            baked_prompt_sha256=sha256(baked_text),
            template_sha256=None,
            dataset_id=f"{plan['data_cfg'].world_spec}#qa@n={plan['n']}",
            num_contexts=len(plan["trained"]),
            trajectories_per_context=(int(g.trajectories_per_context) if sampling else 1),
            eval_num_contexts=len(plan["eval_rel"]),
            max_new_tokens=g.max_new_tokens,
            min_new_tokens=g.min_new_tokens,
            temperature=(g.temperature if sampling else 0.0),
            top_p=(g.top_p if sampling else 1.0),
            top_k=(g.top_k if sampling else 0),
            do_sample=(bool(g.do_sample) if sampling else False),
            seed=self.gen_seed,
            extra=extra,
        )

    def _frame(self, r, x0_id, u_text, baked_text, tok) -> FramedTrajectory:
        world_name = self._plan_cache["world"].name
        mode = self._plan_cache["mode"]
        q = phrasing.question(world_name, r["subj"], r["obj"], mode=mode)
        ans = phrasing.answer(r["subj"], r["obj"], r["provable"], mode=mode)
        ans_ids = tuple(tok(ans, add_special_tokens=False).input_ids)
        if not ans_ids:
            raise ValueError(f"empty answer tokenization for relation {r}")
        base_prefix = tuple(build_prefix_ids(tok, u_text, q))
        baked_prefix = tuple(build_prefix_ids(tok, baked_text, q))
        return FramedTrajectory(
            base_input_ids=base_prefix + ans_ids,
            base_sup_mask=tuple([False] * len(base_prefix) + [True] * len(ans_ids)),
            baked_input_ids=baked_prefix + ans_ids,
            baked_sup_mask=tuple([False] * len(baked_prefix) + [True] * len(ans_ids)),
            x0_id=x0_id,
            num_supervised=len(ans_ids),
        )

    def _sample_frames(self, relations, x0_start, u_text, baked_text, bundle, generator, g):
        """Sample continuations from the PROMPTED base (adapter OFF) on each relation's question and
        frame them base(system=u) vs baked(system=empty) — on-policy baking (mirrors squad_qa._frame).
        `g.trajectories_per_context` samples per relation, all sharing its x0_id (the context id the
        gate disjoints on). Empty (all-stop) continuations are dropped."""
        tok = bundle.tokenizer
        world_name = self._plan_cache["world"].name
        mode = self._plan_cache["mode"]
        questions = [phrasing.question(world_name, r["subj"], r["obj"], mode=mode) for r in relations]
        base_prefixes = [tuple(build_prefix_ids(tok, u_text, q)) for q in questions]
        baked_prefixes = [tuple(build_prefix_ids(tok, baked_text, q)) for q in questions]
        n_per = max(int(g.trajectories_per_context), 1)
        gen_inputs, meta = [], []
        for i in range(len(relations)):
            for _ in range(n_per):
                gen_inputs.append(list(base_prefixes[i]))
                meta.append(i)
        with bundle.base():                       # adapter OFF -> sample the PROMPTED base (teacher)
            ys = generator.generate(gen_inputs, g)
        trajs = []
        for i, y in zip(meta, ys):
            if not y:                             # empty continuation (all stop tokens) -> drop
                continue
            bp, kp = base_prefixes[i], baked_prefixes[i]
            trajs.append(FramedTrajectory(
                base_input_ids=bp + tuple(y),
                base_sup_mask=tuple([False] * len(bp) + [True] * len(y)),
                baked_input_ids=kp + tuple(y),
                baked_sup_mask=tuple([False] * len(kp) + [True] * len(y)),
                x0_id=x0_start + i,
                num_supervised=len(y),
            ))
        return trajs

    def build_trajectories(self, cfg, *, bundle):
        plan = self._plan(cfg)
        tok = bundle.tokenizer
        u_text, baked_text = plan["u_text"], plan["baked_text"]
        self._tokenizer = tok
        sampling = self._use_sampling(cfg)
        self._sampling = sampling                  # the contamination validator reads this (record vs hard-fail)
        if sampling:
            # Canonical baking: sample y from the PROMPTED teacher (base+u) on each relation's question.
            from bakery.seeding import seed_everything
            from bakery.trajectories.generator import make_generator
            seed_everything(self.gen_seed)
            generator = make_generator(cfg.generation.backend, bundle)
            train = self._sample_frames(plan["trained"], 0, u_text, baked_text, bundle, generator, cfg.generation)
            eval_ = self._sample_frames(plan["eval_rel"], 10_000, u_text, baked_text, bundle, generator, cfg.generation)
        else:
            train = [self._frame(r, i, u_text, baked_text, tok) for i, r in enumerate(plan["trained"])]
            # held-out eval-KL set in a disjoint x0_id range (criterion B disjoints on x0_id)
            eval_ = [self._frame(r, 10_000 + j, u_text, baked_text, tok)
                     for j, r in enumerate(plan["eval_rel"])]
        # Optional regularization anchors (base==baked, no prompt; x0_id in the reserved 20000+ band)
        # mixed into training — OFF by default. The coverage validator keys on plan["trained"], not
        # the trajectory list, so anchors never affect depth-1 coverage.
        from bakery.trajectories.regularization import append_train_anchors
        train = append_train_anchors(train, cfg=cfg, bundle=bundle, tokenizer=tok)
        return train, eval_, {"base_u": u_text, "baked": baked_text}

    def pairing_validator(self):
        """Criterion D (here, the COVERAGE criterion): every depth-1 axiom is trained, training carries
        both Yes and No at each trained depth (closure + no all-Yes collapse), and the trained relations
        are recorded for the d′ metric's held-out filter."""
        plan = self._plan_cache
        world, trained = plan["world"], plan["trained"]
        edges = set(world.edges())

        def _validate(spec, train_trajectories, eval_trajectories):
            trained_d1 = {(r["subj"], r["obj"]) for r in trained if r["depth"] == 1 and r["provable"]}
            missing = edges - trained_d1
            if missing:
                raise AssertionError(
                    f"COVERAGE: {len(missing)} depth-1 axiom(s) never trained (the baked model can't "
                    f"have all the prompt's information): {sorted(missing)[:8]}{'…' if len(missing) > 8 else ''}"
                )
            by_depth: dict = {}
            for r in trained:
                c = by_depth.setdefault(r["depth"], {"yes": 0, "no": 0})
                c["yes" if r["provable"] else "no"] += 1
            one_sided = {f"d{d}": c for d, c in by_depth.items() if c["yes"] == 0 or c["no"] == 0}
            if one_sided:
                raise AssertionError(
                    f"TRAIN BALANCE: every trained depth needs BOTH Yes and No answers (else the model "
                    f"learns a constant) — one-sided: {one_sided}")
            if not any(not r["provable"] for r in trained):
                raise AssertionError("CLOSURE: no 'No' training answers — the closed-world info isn't carried.")
            return {
                "trained_relations": [[r["subj"], r["obj"], r["depth"], r["provable"]] for r in trained],
                "n_trained": len(trained),
                "n_trained_true": sum(1 for r in trained if r["provable"]),
                "n_trained_neg": sum(1 for r in trained if not r["provable"]),
                "coverage_depth1": f"{len(trained_d1)}/{len(edges)}",
                "trained_by_depth": {f"d{d}": dict(c) for d, c in sorted(by_depth.items())},
            }

        return _validate

    def contamination_validator(self):
        """Criterion F (theorem_qa-specific, pluggable): decode the train answers and check whether any
        `expect_heldout` probe is STATED by them.

        TEACHER-FORCED: the answer states ONLY the asked relation, so held-out content must be clean —
        HARD-FAIL on any leak (the un-constructable backstop; training already excludes expect_heldout
        pairs). SAMPLED: the prompted teacher's free generation CHAINS (recites the axiom path), so
        held-out consequences get stated. Per an explicit research decision
        (`research/decisions/sampled-teacher-trajectories-keep-cot.md`) we KEEP these on-policy
        trajectories and RECORD the contamination instead of blocking it — NB held-out d′ at depth>=2 is
        then partly recall-of-recited, not pure propagation. This relaxes only this builder's own
        pluggable F-check; the universal gate (E/A/B/C) and the KL primitive are untouched. Always runs
        the d′ schema/balance check on the eval bank."""
        plan = getattr(self, "_plan_cache", None)
        if plan is None:
            return None
        world, probes = plan["world"], plan["bank"]
        mode = plan.get("mode", "directed")
        tokenizer = getattr(self, "_tokenizer", None)
        if not probes or tokenizer is None:
            return None
        sampling = bool(getattr(self, "_sampling", False))

        def _validate(train_trajectories, eval_trajectories):
            continuations = [
                tokenizer.decode(iter_supervised_ids(t)[1], skip_special_tokens=True)
                for t in train_trajectories
            ]
            labels, summary = label_probes_dag(probes, continuations, world, mode=mode)
            balance = assert_probe_schema_and_balance(probes, require_balanced_for_dprime=True)
            if not sampling:
                assert_probes_heldout(probes, labels)     # un-constructable if a held-out probe is leaked
                return {"labels": labels, "enforced": True, "balance": balance, **summary}
            n_leaked = sum(1 for p, lab in zip(probes, labels)
                           if p.get("expect_heldout") and lab.get("label") == "stated")
            return {"labels": labels, "enforced": False, "sampled_cot_leak": True,
                    "n_heldout_stated": n_leaked, "balance": balance, **summary}

        return _validate
