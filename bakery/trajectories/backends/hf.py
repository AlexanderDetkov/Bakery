"""HuggingFace transformers generation backend (default).

Batched, left-padded, sampling-based generation. Returns the generated continuation y as
content token ids only — interior/trailing stop tokens (eos, and Llama-3 <|eot_id|>) are
truncated, so the supervised span the gate validates never contains a stop/pad token (this
also sidesteps the pad==eos aliasing that the reference repos tripped over).
"""

from __future__ import annotations

import torch

from bakery.trajectories.generator import TrajectoryGenerator


def _stop_ids(tokenizer) -> list[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    # Llama-3 instruct end-of-turn marker, if present.
    try:
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot >= 0 and eot != tokenizer.unk_token_id:
            ids.append(eot)
    except Exception:
        pass
    return sorted(set(ids))


def _truncate_at_stop(row: list[int], stop_ids: set) -> list[int]:
    out = []
    for tok in row:
        if tok in stop_ids:
            break
        out.append(tok)
    return out


class HFGenerator(TrajectoryGenerator):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @property
    def backend_name(self) -> str:
        return "hf"

    @torch.no_grad()
    def generate(self, prefixes: list[list[int]], gen_cfg) -> list[list[int]]:
        tok = self.tokenizer
        stop_ids = _stop_ids(tok)
        stop_set = set(stop_ids)
        pad_id = tok.pad_token_id
        out: list[list[int]] = []

        for start in range(0, len(prefixes), gen_cfg.batch_size):
            batch = prefixes[start:start + gen_cfg.batch_size]
            maxlen = max(len(p) for p in batch)
            input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for i, p in enumerate(batch):            # LEFT padding (correct for causal gen)
                input_ids[i, maxlen - len(p):] = torch.tensor(p, dtype=torch.long)
                attn[i, maxlen - len(p):] = 1
            input_ids = input_ids.to(self.device)
            attn = attn.to(self.device)

            gen = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                do_sample=gen_cfg.do_sample,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                top_k=(gen_cfg.top_k or 0),
                max_new_tokens=gen_cfg.max_new_tokens,
                min_new_tokens=gen_cfg.min_new_tokens,
                repetition_penalty=(gen_cfg.repetition_penalty or 1.0),
                no_repeat_ngram_size=(gen_cfg.no_repeat_ngram_size or 0),
                pad_token_id=pad_id,
                eos_token_id=stop_ids or None,
            )
            for row in gen[:, maxlen:].tolist():     # the generated portion only
                out.append(_truncate_at_stop(row, stop_set))
        return out
