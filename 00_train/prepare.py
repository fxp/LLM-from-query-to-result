"""L0 data prep: BPE-tokenize Tiny Shakespeare into train/val.bin.

Output:
    data/input.txt          ~1.1 MB raw Shakespeare (bundled in repo)
    data/train.bin          uint16 token ids, 90% of the corpus
    data/val.bin            uint16 token ids, 10% of the corpus

The vocab is GPT-2's BPE (50,257 entries), the same one L3/L4/L2 use —
so a model trained here can be served by L3 with no tokenizer mismatch.

`input.txt` is committed to the repo so this script doesn't need network
(github.com raw is intermittently blocked from some networks). If for
some reason it's missing, we fall back to downloading from karpathy/
char-rnn — but with a 10s timeout so we fail fast in restricted regions.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np

# Reuse our hand-written BPE in L4.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_transformer"))
import tokenizer  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
DOWNLOAD_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    src = DATA_DIR / "input.txt"

    if not src.exists():
        # Should be rare since we bundle input.txt — but handle the case
        # someone deleted it. Use a short timeout so we don't hang.
        print(f"input.txt missing; downloading from {DOWNLOAD_URL}")
        try:
            with urllib.request.urlopen(DOWNLOAD_URL, timeout=10) as resp:
                src.write_bytes(resp.read())
            print(f"  saved to {src}")
        except Exception as e:
            sys.exit(
                f"\nFailed to download tinyshakespeare ({type(e).__name__}: {e}).\n"
                f"Workaround: manually save the file to {src} and re-run."
            )
    else:
        print(f"using bundled: {src}")

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
