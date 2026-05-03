"""L3 inference server (from-scratch).

POST /generate {prompt, max_tokens, temperature} -> SSE token stream.

This server uses **our own** GPT class (`04_transformer/model.py`) and our
own KV cache logic. No transformers.GPT2LMHeadModel runtime dependency.

Two modes, picked by env var:
  MODEL_PATH=path/to/ckpt.pt  -> load a checkpoint trained by 00_train
  (unset)                     -> load OpenAI's pretrained gpt2 weights
                                  via L4's GPT.from_pretrained() (one-shot
                                  HF download, then run on our own model)

What it shows:
  - tokens the prompt became (BPE)
  - per-step prefill vs decode timings
  - KV cache length growing one token at a time

What it deliberately skips:
  - continuous batching, paged attention, quantization, multi-GPU.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Reuse L4's GPT and tokenizer.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "04_transformer"))
from model import GPT, GPTConfig  # noqa: E402
import tokenizer  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Engine:
    model: GPT
    eos_id: int = 50256  # GPT-2's <|endoftext|>


def load() -> Engine:
    if MODEL_PATH:
        print(f"Loading local checkpoint -> {MODEL_PATH} on {DEVICE}...")
        blob = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        cfg = GPTConfig(**blob["config"])
        model = GPT(cfg).to(DEVICE).eval()
        model.load_state_dict(blob["model"])
        n = sum(p.numel() for p in model.parameters())
        print(f"Loaded: {n/1e6:.2f}M params  "
              f"(local: n_layer={cfg.n_layer} n_head={cfg.n_head} "
              f"n_embd={cfg.n_embd} block_size={cfg.block_size})")
        if "final_loss" in blob:
            print(f"  trained {blob.get('steps','?')} steps, final loss: {blob['final_loss']}")
        return Engine(model=model)

    print(f"Loading pretrained gpt2 on {DEVICE} (via L4 GPT.from_pretrained)...")
    model = GPT.from_pretrained("gpt2").to(DEVICE).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {n/1e6:.0f}M params  (HuggingFace pretrained gpt2)")
    return Engine(model=model)


ENGINE: Engine | None = None
app = FastAPI()


class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = 32
    temperature: float = 0.8


@app.on_event("startup")
def _startup() -> None:
    global ENGINE
    ENGINE = load()


@app.post("/generate")
async def generate(req: GenRequest) -> StreamingResponse:
    assert ENGINE is not None
    model = ENGINE.model
    eos_id = ENGINE.eos_id

    # === Tokenize ===
    # The string "Once upon a time" becomes e.g. [7454, 2402, 257, 640].
    # This is L3's boundary with the outside world: from here down it's tensors.
    ids = tokenizer.encode(req.prompt)
    input_ids = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    print(f"[prompt] {req.prompt!r} -> tokens {ids}")

    async def stream():
        # === Prefill: feed the full prompt, get logits at every position +
        # initial KV cache. Cache lives in CPU/GPU memory between steps.
        t0 = time.perf_counter()
        logits, kvs = model.step(input_ids)
        prefill_ms = (time.perf_counter() - t0) * 1000

        for step_i in range(req.max_tokens):
            # Sample from the LAST position. Real servers also do top-p/top-k.
            last = logits[0, -1]
            if req.temperature <= 0:
                next_id_int = int(last.argmax().item())
            else:
                probs = torch.softmax(last / req.temperature, dim=-1)
                next_id_int = int(torch.multinomial(probs, num_samples=1).item())
            next_id = torch.tensor([[next_id_int]], device=DEVICE, dtype=torch.long)

            piece = tokenizer.decode([next_id_int])
            kv_len = kvs[0][0].size(2)  # K cache seq length
            kind = "prefill" if step_i == 0 else "decode"
            dt = prefill_ms if step_i == 0 else (time.perf_counter() - t1) * 1000
            print(f"[step {step_i:>2}] {kind:<7} {dt:6.1f} ms  "
                  f"kv_len={kv_len:<4}  -> {piece!r}")

            yield f"data: {json.dumps({'token': piece})}\n\n"

            if next_id_int == eos_id:
                break

            # === Decode: 1 new token, attended against cached K/V ===
            t1 = time.perf_counter()
            logits, kvs = model.step(next_id, kvs)
            await asyncio.sleep(0)  # let SSE flush

        yield 'data: {"done": true}\n\n'

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
