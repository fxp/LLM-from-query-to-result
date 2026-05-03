"""L0.5b — SFT starting from OpenAI's pretrained GPT-2 124M (instead of L0).

Why this exists alongside `train.py`:

  - `train.py` SFTs our 7M from-scratch base. That base saw only 0.3M
    Shakespeare tokens — it's barely literate. SFT teaches it the Q/A
    format and a handful of facts, but it can't generalize ("What is the
    capital of Norway?" → empty if Norway isn't in the SFT data even
    though Norway / Oslo were probably in WebText).

  - This script SFTs the same Q/A data on top of OpenAI's GPT-2 124M
    (`GPT.from_pretrained("gpt2")`). That model saw 8 GB of WebText —
    it has a real prior on geography, arithmetic, the names of authors,
    etc. The SFT loop's job here is just to teach it the Q/A *format*;
    the *facts* mostly come for free from the pretraining.

Identical loop as train.py — we just point it at a different base via
`BASE_CKPT_HF=gpt2` semantics. The trained weights save to a different
output path so the two SFTs don't clobber each other.

Recommended hardware:
  - CPU: too slow. 124M with batch 16 × seq 32 ≈ 30 sec per epoch on
    M1, so 100 epochs ≈ 50 min. Doable but tedious.
  - 1× T4 / V100 / A100: ~10× faster. 100 epochs in 5-10 min.

Output: 00b_sft/out/sft_from_gpt2.pt — drop-in for L3:
    MODEL_PATH=00b_sft/out/sft_from_gpt2.pt python ../03_model/server.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "04_transformer"))
from model import GPT, GPTConfig  # noqa: E402
import tokenizer  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_CKPT = Path(os.environ.get("OUT_CKPT") or (HERE / "out" / "sft_from_gpt2.pt"))
DATA = Path(os.environ.get("SFT_DATA") or (HERE / "data.json"))
HF_NAME = os.environ.get("HF_BASE", "gpt2")  # "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"

EOT_ID = 50256
IGNORE = -1

# Defaults tuned for the 124M model. Lower LR than the L0 SFT — pretrained
# features are valuable and we don't want to nuke them with a high lr.
LR = float(os.environ.get("SFT_LR", "5e-5"))
EPOCHS = int(os.environ.get("SFT_EPOCHS", "30"))
BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "8"))   # smaller — 124M is bigger
GRAD_CLIP = float(os.environ.get("SFT_GRAD_CLIP", "1.0"))
WEIGHT_DECAY = float(os.environ.get("SFT_WEIGHT_DECAY", "0.01"))
WARMUP_STEPS = int(os.environ.get("SFT_WARMUP", "30"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_examples() -> list[tuple[list[int], list[int]]]:
    raw = json.loads(DATA.read_text())
    examples = []
    for ex in raw:
        prompt = f"Q: {ex['q']}\nA:"
        answer = f" {ex['a']}"
        prompt_ids = tokenizer.encode(prompt)
        answer_ids = tokenizer.encode(answer) + [EOT_ID]
        full = prompt_ids + answer_ids
        input_ids = full[:-1]
        target_ids = [IGNORE] * (len(prompt_ids) - 1) + answer_ids[:]
        assert len(input_ids) == len(target_ids)
        examples.append((input_ids, target_ids))
    return examples


def make_batch(examples, idx):
    selected = [examples[i] for i in idx]
    max_len = max(len(x[0]) for x in selected)
    B = len(selected)
    inp = torch.full((B, max_len), EOT_ID, dtype=torch.long)
    tgt = torch.full((B, max_len), IGNORE, dtype=torch.long)
    for i, (x, y) in enumerate(selected):
        inp[i, :len(x)] = torch.tensor(x)
        tgt[i, :len(y)] = torch.tensor(y)
    return inp.to(DEVICE), tgt.to(DEVICE)


@torch.no_grad()
def sample_after(model: GPT, q: str) -> str:
    prompt = f"Q: {q}\nA:"
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    logits, kvs = model.step(x)
    out_ids = []
    for _ in range(30):
        next_id = int(logits[0, -1].argmax().item())
        if next_id == EOT_ID:
            break
        out_ids.append(next_id)
        nx = torch.tensor([[next_id]], device=DEVICE, dtype=torch.long)
        logits, kvs = model.step(nx, kvs)
    return tokenizer.decode(out_ids)


def main() -> None:
    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading pretrained {HF_NAME} via L4 GPT.from_pretrained() on {DEVICE}...")
    model = GPT.from_pretrained(HF_NAME).to(DEVICE).train()
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded base: {n/1e6:.2f}M params  device={DEVICE}")

    examples = build_examples()
    print(f"SFT data: {len(examples)} Q/A pairs  "
          f"avg len={sum(len(x[0]) for x in examples)/len(examples):.1f} tokens")

    # Sample questions: mix of in-data (recall) and out-of-data (generalization).
    # OpenAI gpt2 has seen way more than the 7M from-scratch base, so the
    # out-of-data ones should still get plausible answers.
    sample_qs = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is 2 plus 2?",
        "Who wrote Hamlet?",
        # Held out from data.json:
        "What is the capital of Iceland?",         # tests fact generalization
        "What is the largest river in Africa?",    # similar pattern, different fact
        "Who wrote Pride and Prejudice?",          # author fact not in our SFT set
    ]

    print("\n--- BEFORE SFT (pretrained gpt2, no instruction tuning) ---")
    model.eval()
    for q in sample_qs:
        print(f"  Q: {q}")
        print(f"  A:{sample_after(model, q)}")
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                              weight_decay=WEIGHT_DECAY)

    steps_per_epoch = max(1, math.ceil(len(examples) / BATCH_SIZE))
    total_steps = EPOCHS * steps_per_epoch
    print(f"\nSFT for {EPOCHS} epochs × {steps_per_epoch} steps = {total_steps} updates")
    print(f"  base={HF_NAME} ({n/1e6:.0f}M)  batch_size={BATCH_SIZE}  "
          f"lr={LR}  warmup={WARMUP_STEPS}  weight_decay={WEIGHT_DECAY}\n")

    def lr_at(step: int) -> float:
        if step < WARMUP_STEPS:
            return LR * (step + 1) / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return (LR / 10) + 0.5 * (LR - LR / 10) * (1 + math.cos(math.pi * progress))

    t0 = time.time()
    rng = np.random.default_rng(0)
    global_step = 0
    log_every = max(1, EPOCHS // 15)
    for epoch in range(EPOCHS):
        order = rng.permutation(len(examples))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(examples), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE].tolist()
            x, y = make_batch(examples, idx)
            cur_lr = lr_at(global_step)
            for g in optim.param_groups:
                g["lr"] = cur_lr
            _, loss = model(x, targets=y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
        if epoch % log_every == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:>3}/{EPOCHS} | avg loss {epoch_loss/n_batches:6.3f} "
                  f"| lr {cur_lr:.2e} | {time.time()-t0:6.1f}s")

    print("\n--- AFTER SFT ---")
    model.eval()
    for q in sample_qs:
        print(f"  Q: {q}")
        print(f"  A:{sample_after(model, q)}")

    cfg = model.cfg
    torch.save({
        "model": model.state_dict(),
        "config": cfg.__dict__,
        "final_loss": {"sft_epochs": EPOCHS, "lr": LR},
        "steps": EPOCHS * steps_per_epoch,
        "sft_from": f"hf:{HF_NAME}",
    }, OUT_CKPT)
    print(f"\nsaved SFT checkpoint -> {OUT_CKPT}  ({OUT_CKPT.stat().st_size/1e6:.1f} MB)")
    print(f"\nserve via L3:  MODEL_PATH={OUT_CKPT} python ../03_model/server.py")


if __name__ == "__main__":
    main()
