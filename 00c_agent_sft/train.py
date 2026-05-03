"""L0.6 — Agent SFT.

Continue SFT on the L0.5 path-B checkpoint (sft_from_gpt2.pt) with
ReAct-style traces. Loss masked so the model only learns to generate
THOUGHT / ACTION / ANSWER tokens — not the user's Q (which it conditions
on) or the OBSERVATION (which comes from the tool, not the model).

Output: out/agent.pt — drop-in for L3:
    MODEL_PATH=00c_agent_sft/out/agent.pt python ../03_model/server.py
And then L2 agent_loop.py can drive Plan → Act → Observe.

Why this works on a 124M model: SFT on 258 traces forces the model to
emit the exact format (`THOUGHT: ...\\nACTION: tool(args)\\nOBSERVATION:`)
reliably. Combined with constrained generation in the agent loop (stop
at "OBSERVATION:" so we can inject real tool output), even a small base
can drive a 1-2 step agent reliably for in-distribution questions.
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
BASE_CKPT = Path(os.environ.get("BASE_CKPT") or
                 (HERE.parent / "00b_sft" / "out" / "sft_from_gpt2.pt"))
OUT_CKPT = Path(os.environ.get("OUT_CKPT") or (HERE / "out" / "agent.pt"))
DATA = Path(os.environ.get("AGENT_DATA") or (HERE / "data.json"))

EOT_ID = 50256
IGNORE = -1

LR = float(os.environ.get("AGENT_LR", "5e-5"))
EPOCHS = int(os.environ.get("AGENT_EPOCHS", "20"))
BATCH_SIZE = int(os.environ.get("AGENT_BATCH_SIZE", "8"))
GRAD_CLIP = float(os.environ.get("AGENT_GRAD_CLIP", "1.0"))
WEIGHT_DECAY = float(os.environ.get("AGENT_WEIGHT_DECAY", "0.01"))
WARMUP_STEPS = int(os.environ.get("AGENT_WARMUP", "30"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_example(ex: dict) -> tuple[list[int], list[int]]:
    """Tokenize one trace into (input_ids, target_ids) for next-token LM
    with masking. learn=True portions get loss; learn=False portions are
    IGNORE."""
    parts: list[tuple[str, bool]] = []
    parts.append((f"Q: {ex['q']}\n", False))           # prompt: don't learn
    for step in ex["trace"]:
        parts.append((f"THOUGHT: {step['thought']}\n", True))   # model emits
        parts.append((f"ACTION: {step['action']}\n", True))     # model emits
        parts.append((f"OBSERVATION: {step['observation']}\n", False))  # from tool
    parts.append((f"ANSWER: {ex['answer']}", True))    # model emits
    # Plus EOT — model learns to stop
    seq: list[int] = []
    learn: list[bool] = []
    for text, l in parts:
        toks = tokenizer.encode(text)
        seq.extend(toks)
        learn.extend([l] * len(toks))
    seq.append(EOT_ID)
    learn.append(True)

    # Standard next-token: input = seq[:-1], target = seq[1:]
    # Predicting position i+1 from input[:i+1]; learn it iff learn[i+1] is True.
    input_ids = seq[:-1]
    target_ids = [seq[i + 1] if learn[i + 1] else IGNORE
                  for i in range(len(seq) - 1)]
    assert len(input_ids) == len(target_ids)
    return input_ids, target_ids


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
def sample_trace(model: GPT, q: str, max_new: int = 200) -> str:
    """Generate the model's full trace (best-effort — no tool execution).
    For real agent loop see 02_agent/agent_loop.py.

    Greedy. Stops at EOT or when length runs out.
    """
    prompt = f"Q: {q}\nTHOUGHT:"
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    logits, kvs = model.step(x)
    out_ids = []
    for _ in range(max_new):
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
              f"run: cd ../00b_sft && python train_from_gpt2.py", file=sys.stderr)
        sys.exit(1)
    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)

    blob = torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False)
    cfg = GPTConfig(**blob["config"])
    model = GPT(cfg).to(DEVICE).train()
    model.load_state_dict(blob["model"])
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded base: {n/1e6:.2f}M params  device={DEVICE}")
    print(f"  base: {BASE_CKPT.name}  (continuing from L0.5 SFT)")

    raw = json.loads(DATA.read_text())
    examples = [build_example(ex) for ex in raw]
    print(f"agent traces: {len(examples)}  "
          f"avg len={sum(len(x[0]) for x in examples)/len(examples):.1f} tokens")

    sample_qs = [
        "What is the capital of France?",
        "What is 17 plus 25?",
        "Who wrote Hamlet?",
        "What is 8 times 7?",
    ]

    print("\n--- BEFORE agent SFT (just L0.5 chat-SFT) ---")
    model.eval()
    for q in sample_qs:
        out = sample_trace(model, q, max_new=120)
        print(f"  Q: {q}")
        # Show first ~150 chars of trace for readability
        print(f"  →  {out[:150]!r}{'...' if len(out) > 150 else ''}")
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                              weight_decay=WEIGHT_DECAY)

    steps_per_epoch = max(1, math.ceil(len(examples) / BATCH_SIZE))
    total_steps = EPOCHS * steps_per_epoch
    print(f"\nAgent SFT for {EPOCHS} epochs × {steps_per_epoch} steps "
          f"= {total_steps} updates")
    print(f"  batch_size={BATCH_SIZE}  lr={LR}  warmup={WARMUP_STEPS}\n")

    def lr_at(step: int) -> float:
        if step < WARMUP_STEPS:
            return LR * (step + 1) / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return (LR / 10) + 0.5 * (LR - LR / 10) * (1 + math.cos(math.pi * progress))

    t0 = time.time()
    rng = np.random.default_rng(0)
    global_step = 0
    log_every = max(1, EPOCHS // 10)
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

    print("\n--- AFTER agent SFT ---")
    model.eval()
    for q in sample_qs:
        out = sample_trace(model, q, max_new=120)
        print(f"  Q: {q}")
        print(f"  →  {out[:200]!r}{'...' if len(out) > 200 else ''}")

    torch.save({
        "model": model.state_dict(),
        "config": cfg.__dict__,
        "final_loss": {"agent_epochs": EPOCHS, "lr": LR},
        "steps": EPOCHS * steps_per_epoch,
        "agent_from": str(BASE_CKPT),
    }, OUT_CKPT)
    print(f"\nsaved agent checkpoint -> {OUT_CKPT}  ({OUT_CKPT.stat().st_size/1e6:.1f} MB)")
    print(f"\nrun agent loop:  cd ../02_agent && AGENT_MODE=1 python agent.py 'What is the capital of France?'")


if __name__ == "__main__":
    main()
