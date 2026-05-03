"""L0 data prep: download Tiny Shakespeare, BPE-tokenize, save train/val.

Output:
    data/input.txt          ~1.1 MB raw Shakespeare
    data/train.bin          uint16 token ids, 90% of the corpus
    data/val.bin            uint16 token ids, 10% of the corpus

The vocab is GPT-2's BPE (50,257 entries), the same one L3/L4/L2 use —
so a model trained here can be served by L3 with no tokenizer mismatch.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

# Reuse the same tiktoken-backed encoder as L4.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_transformer"))
import tokenizer  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    src = DATA_DIR / "input.txt"

    if not src.exists():
        print(f"downloading -> {src}  (~1.1 MB)")
        urllib.request.urlretrieve(URL, src)
    else:
        print(f"already present: {src}")

    text = src.read_text(encoding="utf-8")
    print(f"corpus length: {len(text):,} chars")

    ids = tokenizer.encode(text)
    print(f"tokenized:     {len(ids):,} BPE tokens  (vocab=50257)")
    print(f"unique tokens used: {len(set(ids)):,}")

    # 90/10 split. Shakespeare is small enough that 10% val is plenty.
    n = len(ids)
    split = int(n * 0.9)
    train_ids = np.array(ids[:split], dtype=np.uint16)
    val_ids = np.array(ids[split:], dtype=np.uint16)
    train_ids.tofile(DATA_DIR / "train.bin")
    val_ids.tofile(DATA_DIR / "val.bin")

    print(f"\nwrote train.bin: {train_ids.size:>8,} tokens  ({train_ids.nbytes/1e6:.2f} MB)")
    print(f"wrote val.bin:   {val_ids.size:>8,} tokens  ({val_ids.nbytes/1e6:.2f} MB)")
    print("\ndone. now: python train.py")


if __name__ == "__main__":
    main()
