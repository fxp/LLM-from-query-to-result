"""Minimal tokenizer wrapper.

GPT-2 uses a Byte-Pair Encoding (BPE) over bytes with a 50257-entry vocab
(50000 merges + 256 byte fallbacks + 1 <|endoftext|>). We don't reimplement
BPE here — it's conceptually simple but full of edge cases (whitespace, UTF-8,
regex pre-tokenizer). The `tiktoken` library ships a drop-in tokenizer that
matches HuggingFace's exactly, so we use that.

If you'd like to read a from-scratch BPE, nanoGPT and minbpe are good.
"""
from __future__ import annotations

import tiktoken

_enc = tiktoken.get_encoding("gpt2")


def encode(text: str) -> list[int]:
    return _enc.encode(text)


def decode(ids: list[int]) -> str:
    return _enc.decode(ids)


def decode_one(token_id: int) -> str:
    return _enc.decode([token_id])
