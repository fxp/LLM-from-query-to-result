"""L0 trainer: train a small GPT from scratch on Tiny Shakespeare.

This is the "from-scratch" half of the repo: we use *exactly the same*
GPT class L4 defines (`04_transformer/model.py`), only smaller, and
optimize it on the token stream prepared by `prepare.py`.

Targets a 5-10 minute CPU run that takes loss from ~10 (random init)
down to ~3, where the model starts producing recognizable Shakespeare-
like text. Not competitive with the OpenAI-pretrained gpt2 (which saw
40 GB of WebText) — just enough to demonstrate that the loop works
and the loss really does descend.

Output: out/ckpt.pt — a state_dict + config; loadable by L3 via
        MODEL_PATH=00_train/out/ckpt.pt python server.py
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Reuse L4's GPT (architecture is identical; we only shrink the config).
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "04_transformer"))
from model import GPT, GPTConfig  # noqa: E402

# ───── Hyperparameters ───────────────────────────────────────────────
# Tuned for "trains in 5-10 min on M1 CPU and shows loss descending".
# Bigger model + more steps = better Shakespeare; same loop.
CFG = GPTConfig(
    vocab_size=50257,   # GPT-2 BPE; we don't shrink — most rows will stay near init
    n_layer=4,
    n_head=4,
    n_embd=128,
    block_size=128,
    dropout=0.0,
)
BATCH_SIZE = 32
MAX_STEPS = 1000
LR_MAX = 3e-4
LR_MIN = 3e-5
WARMUP = 100
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EVAL_INTERVAL = 100
EVAL_ITERS = 20
LOG_INTERVAL = 25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "out"


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random [B, T] window from train/val.bin.

    The trick: instead of reading file→tokenize on the fly, we mmap the
    pre-tokenized .bin file and slice it. That makes every batch O(B*T)
    memory, no Python loop, no tokenizer in the hot path.
    """
    path = DATA_DIR / f"{split}.bin"
    data = np.memmap(path, dtype=np.uint16, mode="r")
    # Random starts; each window is block_size tokens for inputs, +1 for targets.
    ix = torch.randint(len(data) - CFG.block_size - 1, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i : i + CFG.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + CFG.block_size].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model: GPT) -> dict[str, float]:
    """Average loss over EVAL_ITERS random batches, both splits."""
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def lr_at(step: int) -> float:
    """Linear warmup then cosine decay to LR_MIN."""
    if step < WARMUP:
        return LR_MAX * (step + 1) / WARMUP
    if step >= MAX_STEPS:
        return LR_MIN
    progress = (step - WARMUP) / (MAX_STEPS - WARMUP)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not (DATA_DIR / "train.bin").exists():
        print("data/train.bin not found. run: python prepare.py", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(1337)
    model = GPT(CFG).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_non_emb = n_params - model.wte.weight.numel() - model.wpe.weight.numel()
    print(f"model: {n_params/1e6:.2f}M params total  "
          f"({n_non_emb/1e6:.2f}M non-embedding)  device={DEVICE}")
    print(f"config: n_layer={CFG.n_layer} n_head={CFG.n_head} n_embd={CFG.n_embd} "
          f"block_size={CFG.block_size}")

    # AdamW with weight decay only on 2D params (matrix weights), not on
    # 1D bias / LN — same recipe as nanoGPT, GPT-2, etc.
    decay, nodecay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else nodecay).append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=LR_MAX, betas=(0.9, 0.95),
    )

    print(f"\ntraining for {MAX_STEPS} steps, batch_size={BATCH_SIZE}, "
          f"block_size={CFG.block_size}\n")

    t0 = time.time()
    last_log = t0
    for step in range(MAX_STEPS):
        # LR schedule
        lr = lr_at(step)
        for g in optim.param_groups:
            g["lr"] = lr

        # ---- one optimization step ----
        x, y = get_batch("train")
        _, loss = model(x, targets=y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optim.step()

        if step % LOG_INTERVAL == 0:
            now = time.time()
            tps = (BATCH_SIZE * CFG.block_size * LOG_INTERVAL) / max(now - last_log, 1e-9)
            print(f"step {step:>4} | loss {loss.item():6.3f} | lr {lr:.2e} | "
                  f"{tps:>6,.0f} tok/s | {now-t0:5.1f}s")
            last_log = now

        if step > 0 and step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            print(f"          eval | train {losses['train']:.3f}  val {losses['val']:.3f}")

    # Final eval + save.
    losses = estimate_loss(model)
    print(f"\nfinal | train {losses['train']:.3f}  val {losses['val']:.3f}  "
          f"({time.time()-t0:.1f}s total)")

    ckpt_path = OUT_DIR / "ckpt.pt"
    torch.save({
        "model": model.state_dict(),
        "config": CFG.__dict__,
        "final_loss": losses,
        "steps": MAX_STEPS,
    }, ckpt_path)
    print(f"\nsaved checkpoint -> {ckpt_path}  ({ckpt_path.stat().st_size/1e6:.1f} MB)")
    print("\ngenerate samples:  python sample.py 'ROMEO:'")
    print("serve via L3:      MODEL_PATH=00_train/out/ckpt.pt python ../03_model/server.py")


if __name__ == "__main__":
    main()
