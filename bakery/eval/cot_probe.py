"""Post-hoc CoT-vs-no-CoT propagation eval on a SAVED adapter (the internal-vs-chaining control).

The headline propagation metric reads belief with NO chain-of-thought (a single-forward-pass forced
choice) — by design it measures INTERNAL propagation. This script asks the complementary question the
user posed: does letting the model REASON ALOUD (CoT) recover consequences it fails to answer internally?

Two modes (reusing the cycle-3 Veld adapters; no re-bake). For each probe and each model state on ONE
checkpoint (prior = base+empty, prompted = base+u, baked = adapter+empty):
  * mode=belief  — noCoT belief logP(pos)-logP(neg); plus a CoT belief scored after a generated rationale +
    a "Final answer:" cue. With --max_cot_tokens 0 this is the CUE-ONLY ablation (no rationale) that exposes
    the cue's intrinsic bias. NOTE (cycle 4): the cue+single-token scoring is artifactual — the cue alone
    shifts the forced choice ~8-12 nats — so DO NOT compare absolute noCoT-vs-CoT belief. Kept for the ablation.
  * mode=freegen — the CORRECTED readout: free-generate the answer (k sampled rationales), PARSE Yes/No from
    the text, majority-vote (self-consistency). Reports per-probe noCoT belief + the CoT majority answer +
    fraction-correct, so CoT accuracy can be compared to no-CoT accuracy (sign of belief) without any cue.

Reuses the audited _seq_logprob / build_prefix_ids primitives. Writes a JSON for analysis. This is an EVAL
(loads the model stack) — deliberately NOT a per-epoch metric and NOT in analysis/ (which is JSON-only).

    python -m bakery.eval.cot_probe --mode freegen --model meta-llama/Llama-3.1-8B-Instruct \
        --adapter results/bake_fact/prop-veld-8b-mixed/checkpoints/final \
        --u data/prompts/veld_chain_u.md --probes data/probes/veld_probes.json \
        --out results/bake_fact/prop-veld-8b-mixed/cot_freegen.json --k 5
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch

from bakery.eval.metrics.propagation import _seq_logprob
from bakery.prompts import build_prefix_ids, load_prompt

CUE = "\nFinal answer (Yes or No):"
COT_INSTR = " Reason briefly step by step, then end with your final answer as a single word, Yes or No."
_YN = re.compile(r"\b(yes|no)\b", re.I)


def _strip_forcedchoice(q: str) -> str:
    return q.replace("Answer with only Yes or No. ", "").strip()


def _parse_yesno(text: str):
    """The LAST yes/no token in the generation (closest to the conclusion); None if absent."""
    found = _YN.findall(text.lower())
    return found[-1] if found else None


def _noCoT_belief(model, tok, system_text, pr, device) -> float:
    prefix = build_prefix_ids(tok, system_text, pr["question"])
    return (_seq_logprob(model, tok, prefix, pr["pos"], device)
            - _seq_logprob(model, tok, prefix, pr["neg"], device))


# ----- mode=belief (CoT belief after a cue; --max_cot_tokens 0 = cue-only ablation) ----------------

def _generate_greedy(model, tok, prefix_ids, device, max_new):
    if max_new <= 0:                          # cue-only ablation: no rationale, isolates the cue's effect
        return []
    inp = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(inp, attention_mask=torch.ones_like(inp), max_new_tokens=max_new,
                             do_sample=False, pad_token_id=tok.pad_token_id)
    return out[0].tolist()[len(prefix_ids):]


def _beliefs_for_state(model, tok, probes, system_text, device, max_new):
    cue_ids = tok(CUE, add_special_tokens=False).input_ids
    out = []
    for pr in probes:
        noCoT = _noCoT_belief(model, tok, system_text, pr, device)
        cot_prefix = build_prefix_ids(tok, system_text, _strip_forcedchoice(pr["question"]) + COT_INSTR)
        with torch.no_grad():
            rationale = _generate_greedy(model, tok, cot_prefix, device, max_new)
        while rationale and rationale[-1] in (tok.eos_token_id, tok.pad_token_id):
            rationale.pop()
        scored = cot_prefix + rationale + cue_ids
        cot = (_seq_logprob(model, tok, scored, pr["pos"], device)
               - _seq_logprob(model, tok, scored, pr["neg"], device))
        out.append({"hop": pr["hop"], "pos": pr["pos"], "question": pr["question"],
                    "noCoT": noCoT, "cot": cot, "rationale": tok.decode(rationale)[:400]})
    return out


# ----- mode=freegen (parse Yes/No from sampled generations; self-consistency) ----------------------

def _freegen_for_state(model, tok, probes, system_text, device, k, max_new, temperature, top_p):
    out = []
    for i, pr in enumerate(probes):
        noCoT = _noCoT_belief(model, tok, system_text, pr, device)
        prefix = build_prefix_ids(tok, system_text, _strip_forcedchoice(pr["question"]) + COT_INSTR)
        inp = torch.tensor([prefix], dtype=torch.long, device=device)
        attn = torch.ones_like(inp)
        answers, first_sample = [], ""
        for s in range(k):
            torch.manual_seed(1000 * i + s)               # reproducible sampling
            with torch.no_grad():
                gen = model.generate(inp, attention_mask=attn, max_new_tokens=max_new, do_sample=True,
                                     temperature=temperature, top_p=top_p, pad_token_id=tok.pad_token_id)
            text = tok.decode(gen[0].tolist()[len(prefix):], skip_special_tokens=True)
            answers.append(_parse_yesno(text))
            if s == 0:
                first_sample = text[:400]
        valid = [a for a in answers if a]
        correct = pr["pos"].strip().lower()              # 'yes' or 'no'
        out.append({
            "hop": pr["hop"], "pos": pr["pos"], "question": pr["question"], "noCoT_belief": noCoT,
            "cot_answers": answers,
            "cot_majority": Counter(valid).most_common(1)[0][0] if valid else None,
            "cot_frac_correct": (sum(a == correct for a in valid) / len(valid)) if valid else None,
            "sample": first_sample,
        })
    return out


def run(model_name, adapter, u_path, probes_path, out_path, *, mode="freegen", device="cuda",
        max_new=160, k=5, temperature=0.7, top_p=0.95):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()
    model = PeftModel.from_pretrained(base, adapter).eval()

    u_text = load_prompt(u_path)
    probes = json.loads(Path(probes_path).read_text())["probes"]

    def for_state(system_text):
        if mode == "belief":
            return _beliefs_for_state(model, tok, probes, system_text, device, max_new)
        return _freegen_for_state(model, tok, probes, system_text, device, k, max_new, temperature, top_p)

    results = {}
    with model.disable_adapter():                       # prior + prompted = base model
        results["prior"] = for_state("")
        results["prompted"] = for_state(u_text)
    results["baked"] = for_state("")                    # adapter enabled

    payload = {"model": model_name, "adapter": adapter, "mode": mode,
               "k": k, "max_new": max_new, "temperature": temperature, "states": results}
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
    ap.add_argument("--mode", choices=["belief", "freegen"], default="freegen")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_cot_tokens", type=int, default=160, dest="max_new")
    ap.add_argument("--k", type=int, default=5, help="self-consistency samples (freegen mode)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    a = ap.parse_args(argv)
    run(a.model, a.adapter, a.u, a.probes, a.out, mode=a.mode, device=a.device,
        max_new=a.max_new, k=a.k, temperature=a.temperature, top_p=a.top_p)


if __name__ == "__main__":
    main()
