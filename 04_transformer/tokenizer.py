"""Tokenizer facade — thin shim over our hand-written BPE.

GPT-2 uses Byte-Pair Encoding over bytes with a 50,257-entry vocab. The
algorithm + vocab loading + UTF-8 byte mapping all live in `bpe.py`. This
file just exposes the three call sites the rest of the repo uses:

    encode(text)     -> list[int]      # used by 00_train, 03_model, 02_agent
    decode(ids)      -> str            # used by 03_model
    decode_one(id)   -> str            # used by 04_transformer/inference

We could call `bpe.encode` directly everywhere, but keeping this facade
means swapping in a different tokenizer (a hand-trained BPE, char-level,
SentencePiece, etc.) is a one-file change.
"""
from __future__ import annotations

from bpe import decode, decode_one, encode  # noqa: F401  (re-export)

__all__ = ["encode", "decode", "decode_one"]
