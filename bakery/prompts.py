"""Prompt loading + chat-template framing — the SINGLE source of truth for how a
(prompt, context, generated y) is assembled into token ids.

Both framings (base = system+u, baked = system+baked_prompt) build their prefix here, then
the SAME generated token ids y are appended to each. Because y is appended as TOKEN IDS
(never re-encoded from text), the supervised tokens are identical across framings by
construction — the gate's mask-alignment invariant cannot be violated by a tokenization
boundary effect.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(text: str | None) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


def load_prompt(path_or_text: str | None) -> str:
    """Load a prompt. If it names an existing file, read it; otherwise treat the string
    itself as the literal prompt text (handy for inline prompts in tests/configs). Empty
    or missing -> "" (the canonical 'no prompt' used by the baked framing)."""
    if not path_or_text:
        return ""
    p = Path(path_or_text)
    if p.exists() and p.is_file():
        return p.read_text()
    return path_or_text


def build_prefix_ids(tokenizer, system_text: str, user_text: str) -> list[int]:
    """Token ids for the chat prefix up to the assistant generation point.

    Uses the tokenizer's chat template when present (instruct models); falls back to a
    minimal manual framing for base models without one. This is the only place framing
    happens — generation and KL both consume prefixes built here.
    """
    if getattr(tokenizer, "chat_template", None):
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        return list(ids)

    # Fallback: no chat template (e.g. a tiny stub or a base model).
    text = (f"{system_text}\n\n" if system_text else "") + f"{user_text}\n"
    return list(tokenizer(text, add_special_tokens=True).input_ids)
