"""Post-hoc CoT-vs-no-CoT propagation eval on a SAVED adapter (the internal-vs-chaining control).

The headline propagation metric reads belief with NO chain-of-thought (a single-forward-pass forced
choice) — by design it measures INTERNAL propagation. This script asks the complementary question the
user posed: does letting the model REASON ALOUD (CoT) recover consequences it fails to answer internally?
If CoT rescues a baked model's / a small model's failures, those failures were about single-pass
internalization, not missing knowledge.

It reuses the cycle-3 Veld adapters (no re-bake). For each probe and each model state on ONE checkpoint
(prior = base+empty, prompted = base+u, baked = adapter+empty) it computes:
  * noCoT belief: logP(pos)-logP(neg) of the immediate answer (same readout as the propagation metric);
  * CoT belief:   generate a short greedy rationale, then score logP(pos)-logP(neg) after a final-answer cue.
Reuses the audited _seq_logprob / build_prefix_ids primitives. Writes a JSON for analysis. This is an EVAL
(loads the model stack) — deliberately NOT a per-epoch metric (generation is too slow for that) and NOT in
analysis/ (which is JSON-only).

    python -m bakery.eval.cot_probe --model meta-llama/Llama-3.1-8B-Instruct \
        --adapter results/bake_fact/prop-veld-8b-mixed/checkpoints/final \
        --u data/prompts/veld_chain_u.md --probes data/probes/veld_probes.json \
        --out results/bake_fact/prop-veld-8b-mixed/cot_probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bakery.eval.metrics.propagation import _seq_logprob
from bakery.prompts import build_prefix_ids, load_prompt

CUE = "\nFinal answer (Yes or No):"
COT_INSTR = " Reason briefly step by step, then end with your final answer as a single word, Yes or No."


def _strip_forcedchoice(q: str) -> str:
    return q.replace("Answer with only Yes or No. ", "").strip()


def _generate_rationale(model, tok, prefix_ids, device, max_new):
    if max_new <= 0:                          # cue-only ablation: no rationale, isolates the cue's effect
        return []
    inp = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(inp, attention_mask=torch.ones_like(inp), max_new_tokens=max_new,
                             do_sample=False, pad_token_id=tok.pad_token_id)
    return out[0].tolist()[len(prefix_ids):]


def _beliefs_for_state(model, tok, probes, system_text, device, max_new):
    """Return per-probe {noCoT, cot} belief for one (adapter-state, system_text)."""
    cue_ids = tok(CUE, add_special_tokens=False).input_ids
    out = []
    for pr in probes:
        # no-CoT: forced-choice question as-is, immediate answer.
        nc_prefix = build_prefix_ids(tok, system_text, pr["question"])
        noCoT = (_seq_logprob(model, tok, nc_prefix, pr["pos"], device)
                 - _seq_logprob(model, tok, nc_prefix, pr["neg"], device))
        # CoT: reasoning-allowed question, generate rationale, then score after a cue.
        q_core = _strip_forcedchoice(pr["question"])
        cot_prefix = build_prefix_ids(tok, system_text, q_core + COT_INSTR)
        with torch.no_grad():
            rationale = _generate_rationale(model, tok, cot_prefix, device, max_new)
        # drop a trailing eos so the cue continues the same assistant turn
        while rationale and rationale[-1] in (tok.eos_token_id, tok.pad_token_id):
            rationale.pop()
        scored_prefix = cot_prefix + rationale + cue_ids
        cot = (_seq_logprob(model, tok, scored_prefix, pr["pos"], device)
               - _seq_logprob(model, tok, scored_prefix, pr["neg"], device))
        out.append({"hop": pr["hop"], "pos": pr["pos"], "question": pr["question"],
                    "noCoT": noCoT, "cot": cot,
                    "rationale": tok.decode(rationale)[:400]})   # persisted for auditability
    return out


def run(model_name, adapter, u_path, probes_path, out_path, device="cuda", max_new=100):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()
    model = PeftModel.from_pretrained(base, adapter).eval()

    u_text = load_prompt(u_path)
    probes = json.loads(Path(probes_path).read_text())["probes"]

    results = {}
    # prior + prompted: adapter DISABLED (base model); baked: adapter ENABLED.
    with model.disable_adapter():
        results["prior"] = _beliefs_for_state(model, tok, probes, "", device, max_new)
        results["prompted"] = _beliefs_for_state(model, tok, probes, u_text, device, max_new)
    results["baked"] = _beliefs_for_state(model, tok, probes, "", device, max_new)

    payload = {"model": model_name, "adapter": adapter, "max_cot_tokens": max_new, "states": results}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=1))
    print(out_path)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--u", required=True, help="the injected fact prompt (for the prompted state)")
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_cot_tokens", type=int, default=100)
    a = ap.parse_args(argv)
    run(a.model, a.adapter, a.u, a.probes, a.out, a.device, a.max_cot_tokens)


if __name__ == "__main__":
    main()
