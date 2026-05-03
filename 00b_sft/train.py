"""L0.5 SFT — supervised fine-tuning on instruction (Q/A) data.

Takes a base checkpoint from 00_train and continues training it on a small
Q/A dataset, with a key change: the loss is masked so it only counts on
the *answer* tokens. The model isn't penalized for failing to predict the
question (it's just there to condition on).

Format we train on:
    Q: <question>\nA: <answer><|endoftext|>

After SFT the model learns to:
  - Recognize the "Q: ... A:" pattern and emit something after "A:"
  - Stop with <|endoftext|> after the answer (in the ideal world)
  - For knowledge it's actually seen, parrot the answer

Honest expectation: the L0 base (7M params, trained on 0.3M Shakespeare
tokens) is too undertrained for SFT to produce factual answers. After
SFT it'll mostly learn the FORMAT — emit "A: <something>." style replies.
For real factual QA you'd start from a larger base (gpt2-small 124M+) and
a real SFT corpus (Alpaca etc.).
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
BASE_CKPT = Path(os.environ.get("BASE_CKPT") or (HERE.parent / "00_train" / "out" / "ckpt.pt"))
OUT_CKPT = Path(os.environ.get("OUT_CKPT") or (HERE / "out" / "sft.pt"))
DATA = Path(os.environ.get("SFT_DATA") or (HERE / "data.json"))

EOT_ID = 50256  # GPT-2's <|endoftext|>
IGNORE = -1     # tells F.cross_entropy(ignore_index=-1) to skip these positions

# SFT hyperparameters. All overridable via env so this same script works
# for the small from-scratch base (defaults below) AND for the OpenAI
# 124M base via 00b_sft/train_from_gpt2.py without code duplication.
#   LR, EPOCHS, BATCH_SIZE, MAX_GRAD_NORM, WEIGHT_DECAY, WARMUP
LR = float(os.environ.get("SFT_LR", "3e-4"))            # higher than commit 2 default 1e-4
EPOCHS = int(os.environ.get("SFT_EPOCHS", "100"))       # was 50; with 4× more data we still want more passes
BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "16"))  # was 8
GRAD_CLIP = float(os.environ.get("SFT_GRAD_CLIP", "1.0"))
WEIGHT_DECAY = float(os.environ.get("SFT_WEIGHT_DECAY", "0.01"))
WARMUP_STEPS = int(os.environ.get("SFT_WARMUP", "50"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_examples() -> list[tuple[list[int], list[int]]]:
    """Return list of (input_ids, target_ids) where target_ids has IGNORE
    on the prompt portion so loss is only computed on the answer."""
    raw = json.loads(DATA.read_text())
    examples = []
    for ex in raw:
        prompt = f"Q: {ex['q']}\nA:"
        answer = f" {ex['a']}"
        prompt_ids = tokenizer.encode(prompt)
        answer_ids = tokenizer.encode(answer) + [EOT_ID]
        full = prompt_ids + answer_ids
        # input is full[:-1], target is full[1:] (next-token prediction).
        # On the prompt portion we set target to IGNORE so loss is masked.
        input_ids = full[:-1]
        target_ids = [IGNORE] * (len(prompt_ids) - 1) + answer_ids[:]
        # Length sanity: target_ids should equal full[1:] except IGNORE on prompt
        assert len(input_ids) == len(target_ids), (len(input_ids), len(target_ids))
        examples.append((input_ids, target_ids))
    return examples


def make_batch(examples: list[tuple[list[int], list[int]]],
               idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad selected examples to the same length and stack."""
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
    """Generate an answer for a sample question (greedy, max 30 tokens)."""
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
    if not BASE_CKPT.exists():
        print(f"base checkpoint not found: {BASE_CKPT}\n"
              f"run: cd ../00_train && python train.py", file=sys.stderr)
        sys.exit(1)
    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)

    blob = torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False)
    cfg = GPTConfig(**blob["config"])
    model = GPT(cfg).to(DEVICE).train()
    model.load_state_dict(blob["model"])
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded base: {n/1e6:.2f}M params  pre-SFT loss={blob.get('final_loss')}")

    examples = build_examples()
    print(f"SFT data: {len(examples)} Q/A pairs  "
          f"avg len={sum(len(x[0]) for x in examples)/len(examples):.1f} tokens")

    sample_qs = [
        # In-data (exact match): should be answered after SFT
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is 2 plus 2?",
        "Who wrote Hamlet?",
        "How many continents are there?",
        # In-data (paraphrased): tests question-form generalization
        "Capital of France?",
        # Out-of-data (held out): tests fact generalization. The 7M base
        # has no prior on these; expect SFT to fail. Use a bigger base to
        # see real generalization (cf 00b_sft/train_from_gpt2.py).
        "What is the capital of Norway?",  # IN data actually — but rephrased forms not all included
        "What is 100 plus 100?",            # arithmetic the model has never seen
        "What is the capital of Mars?",     # nonsensical: should hallucinate or refuse
    ]

    print("\n--- BEFORE SFT (base model) ---")
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
    print(f"  batch_size={BATCH_SIZE}  lr={LR}  warmup={WARMUP_STEPS}  "
          f"weight_decay={WEIGHT_DECAY}  device={DEVICE}\n")

    def lr_at(step: int) -> float:
        # Linear warmup then cosine decay to lr/10.
        if step < WARMUP_STEPS:
            return LR * (step + 1) / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return (LR / 10) + 0.5 * (LR - LR / 10) * (1 + math.cos(math.pi * progress))

    t0 = time.time()
    rng = np.random.default_rng(0)
    global_step = 0
    log_every = max(1, EPOCHS // 20)  # ~20 log lines total
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

    # Save SFT'd checkpoint with same format as L0 so L3 can load it.
    torch.save({
        "model": model.state_dict(),
        "config": blob["config"],
        "final_loss": {"sft_epochs": EPOCHS, "lr": LR},
        "steps": blob.get("steps", 0),
        "sft_from": str(BASE_CKPT),
    }, OUT_CKPT)
    print(f"\nsaved SFT checkpoint -> {OUT_CKPT}  ({OUT_CKPT.stat().st_size/1e6:.1f} MB)")
    print(f"\nserve via L3:  MODEL_PATH={OUT_CKPT} python ../03_model/server.py")


if __name__ == "__main__":
    main()
