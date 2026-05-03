"""GPT-2 BPE tokenizer, hand-written.

This is a from-scratch reimplementation of OpenAI's BPE encoding for GPT-2,
matching tiktoken bit-for-bit (we verify this against tiktoken in the test
section at the bottom). It uses OpenAI's published `encoder.json` and
`vocab.bpe` files for the merge rules / vocab — those are 1.2 MB of *data*,
not code, downloaded once into `data/`.

Why bother? The repo's promise is "every token comes out of code we wrote".
tiktoken is a fast Rust implementation; this Python version is ~50× slower
but ~30× shorter and lets you actually see the algorithm. For training and
inference at this scale (Shakespeare, 1MB) the speed difference doesn't
matter — encoding the whole corpus takes <2 s.

The four moving parts:
  1. `bytes_to_unicode()`  — map each byte 0..255 to a printable unicode
     char so BPE can operate on str (instead of bytes). This is OpenAI's
     trick for keeping the BPE strings displayable while still being
     byte-level (handles all unicode without OOV).
  2. Regex pre-tokenizer  — splits text into "words" (English-ish chunks).
     BPE merges only happen within a chunk, never across.
  3. `bpe(token)`           — given one chunk, repeatedly apply the
     highest-priority merge rule until none apply.
  4. `encode/decode`        — string → byte-strs → bpe → ids; reverse it.

References:
  - https://github.com/openai/gpt-2/blob/master/src/encoder.py
  - https://github.com/karpathy/minbpe (educational from-scratch BPE)
"""
from __future__ import annotations

import json
import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import regex as re  # third-party: needs \p{L} \p{N} which stdlib re lacks


_DATA_DIR = Path(__file__).resolve().parent / "data"
_ENCODER_URL = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json"
_VOCAB_URL   = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"

# GPT-2's pre-tokenization regex: pulls "words" out of arbitrary text. The
# leading-space variants (` ?\p{L}+` etc.) are why GPT-2 tokenizes " hello"
# and "hello" differently — the space is part of the token.
PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Reversible mapping from byte (0..255) to a unicode char.

    OpenAI maps the printable ASCII range to itself, then maps the rest
    (control chars, non-ASCII bytes) to characters in the unicode private
    use area starting at U+0100. This way every byte sequence becomes a
    string of "safe" chars that can be displayed and handled like text.
    """
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _ensure_files() -> tuple[Path, Path]:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    enc_path = _DATA_DIR / "encoder.json"
    bpe_path = _DATA_DIR / "vocab.bpe"
    if not enc_path.exists():
        print(f"BPE: downloading {_ENCODER_URL} -> {enc_path}")
        urllib.request.urlretrieve(_ENCODER_URL, enc_path)
    if not bpe_path.exists():
        print(f"BPE: downloading {_VOCAB_URL} -> {bpe_path}")
        urllib.request.urlretrieve(_VOCAB_URL, bpe_path)
    return enc_path, bpe_path


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """All adjacent symbol pairs in a word, e.g. ('h','i','!') -> {('h','i'), ('i','!')}."""
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class BPE:
    """Stateful BPE encoder. One instance reused across encode/decode calls."""

    def __init__(self) -> None:
        enc_path, bpe_path = _ensure_files()
        self.encoder: dict[str, int] = json.loads(enc_path.read_text())
        self.decoder: dict[int, str] = {v: k for k, v in self.encoder.items()}

        # Parse vocab.bpe: a header line + 50000 merge rules of the form "a b".
        merges = bpe_path.read_text(encoding="utf-8").split("\n")[1:-1]
        self.bpe_ranks: dict[tuple[str, str], int] = {
            tuple(merge.split()): i for i, merge in enumerate(merges)
        }

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Memoize per-token BPE results — same chunk appears thousands of times
        # in real text. This is what makes Python BPE survivable.
        self._cache: dict[str, str] = {}

    def _bpe(self, token: str) -> str:
        """Apply BPE merges to one pre-tokenized chunk.

        Algorithm: maintain a sequence of "symbols" (initially individual
        chars). Repeatedly find the symbol pair with the lowest rank in
        bpe_ranks (i.e. the highest-priority merge), and merge it. Stop
        when no pair has a known rank. Return the symbols joined by spaces.
        """
        if token in self._cache:
            return self._cache[token]

        word = tuple(token)
        pairs = _get_pairs(word)
        if not pairs:
            self._cache[token] = token
            return token

        while True:
            # Pick the best (lowest-rank) mergeable pair, or stop.
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best not in self.bpe_ranks:
                break
            first, second = best
            new_word: list[str] = []
            i = 0
            while i < len(word):
                # Skip ahead to the next occurrence of `first` (no further merges
                # possible before that point).
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        out = " ".join(word)
        self._cache[token] = out
        return out

    def encode(self, text: str) -> list[int]:
        """text -> list of token ids."""
        ids: list[int] = []
        for chunk in re.findall(PAT, text):
            # 1. encode chunk's bytes via the byte->unicode map (so BPE works on str)
            byte_str = "".join(self.byte_encoder[b] for b in chunk.encode("utf-8"))
            # 2. apply BPE merges
            merged = self._bpe(byte_str).split(" ")
            # 3. look up token ids
            ids.extend(self.encoder[piece] for piece in merged)
        return ids

    def decode(self, ids: list[int]) -> str:
        """list of token ids -> text. Inverts encode() exactly."""
        text = "".join(self.decoder[i] for i in ids)
        # Map each unicode char back to its byte and decode UTF-8.
        # `errors="replace"` so a partial multi-byte token (mid-stream) shows
        # a Unicode replacement char instead of crashing.
        return bytes(self.byte_decoder[c] for c in text).decode("utf-8", errors="replace")


# Module-level singleton + simple functional API.
_bpe: BPE | None = None


def _get() -> BPE:
    global _bpe
    if _bpe is None:
        _bpe = BPE()
    return _bpe


def encode(text: str) -> list[int]:
    return _get().encode(text)


def decode(ids: list[int]) -> str:
    return _get().decode(ids)


def decode_one(token_id: int) -> str:
    return _get().decode([token_id])


# ---------------------------------------------------------------------------
# Self-test: verify our BPE matches tiktoken bit-for-bit, when tiktoken is
# available. Run with `python bpe.py`.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "ROMEO:\nO Juliet, wherefore art thou?",
        "Question: What is the capital of France?\nAnswer:",
        "  multiple   spaces   and\ttabs\n\nnewlines",
        "中文 日本語 🚀 emoji",
        "1234567890 + - * / = (test)",
    ]
    print("BPE self-test")
    print("=" * 60)
    try:
        import tiktoken
        ref = tiktoken.get_encoding("gpt2")
        all_ok = True
        for s in samples:
            ours = encode(s)
            theirs = ref.encode(s)
            ok = ours == theirs
            mark = "✓" if ok else "✗"
            all_ok &= ok
            print(f"{mark} {s[:40]!r}  ours={len(ours)}  ref={len(theirs)}  match={ok}")
            if not ok:
                print(f"  ours  : {ours}")
                print(f"  ref   : {theirs}")
        print("=" * 60)
        print("ALL MATCH" if all_ok else "MISMATCH — bug in our BPE")
    except ImportError:
        print("tiktoken not available — skipping equivalence test, just printing samples:")
        for s in samples:
            ids = encode(s)
            print(f"  {s[:40]!r:42}  -> {ids[:8]}{'...' if len(ids)>8 else ''}  ({len(ids)} tokens)")
            roundtrip = decode(ids)
            assert roundtrip == s, f"decode mismatch: {roundtrip!r} != {s!r}"
        print("encode-decode roundtrip OK on all samples")
