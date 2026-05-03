"""L3 inference server (minimal).

POST /generate {prompt, max_tokens, temperature} -> SSE token stream.

Runs GPT-2 small on CPU or CUDA. Shows in the server log:
  - the tokens the prompt became
  - how long prefill vs decode steps take
  - how the KV cache grows step by step

What it deliberately skips: continuous batching, paged attention, quantization,
multi-GPU. Those matter for throughput in production but would obscure the
core "request -> token stream" story.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Engine:
    model: GPT2LMHeadModel
    tok: GPT2TokenizerFast


def load() -> Engine:
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    tok = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {n/1e6:.0f}M params")
    return Engine(model=model, tok=tok)


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
    tok, model = ENGINE.tok, ENGINE.model

    # === Tokenize ===
    # The string "Once upon a time" becomes e.g. [7454, 2402, 257, 640].
    # This is L3's boundary with the outside world: from here down it's tensors.
    input_ids = tok(req.prompt, return_tensors="pt").input_ids.to(DEVICE)
    print(f"[prompt] {req.prompt!r} -> tokens {input_ids.tolist()[0]}")

    async def stream():
        past = None  # the KV cache
        cur_ids = input_ids
        for step in range(req.max_tokens):
            t0 = time.perf_counter()
            with torch.no_grad():
                # model(...) ultimately calls forward() on each transformer
                # block — see 04_transformer for what that actually does.
                out = model(input_ids=cur_ids, past_key_values=past, use_cache=True)
            dt = (time.perf_counter() - t0) * 1000
            logits = out.logits[:, -1, :]             # last position only
            past = out.past_key_values                 # <- grown by cur_ids tokens
            # transformers 5.x returns a DynamicCache object (no longer a tuple);
            # keep the legacy tuple path as a fallback for older versions.
            kv_len = past.get_seq_length() if hasattr(past, "get_seq_length") else past[0][0].shape[-2]

            # Temperature sampling. Real servers also support top_p / top_k.
            if req.temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / req.temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            piece = tok.decode(next_id[0])
            kind = "prefill" if step == 0 else "decode"
            print(f"[step {step:>2}] {kind:<7} {dt:6.1f} ms  kv_len={kv_len:<4}  -> {piece!r}")

            yield f"data: {json.dumps({'token': piece})}\n\n"

            if next_id.item() == tok.eos_token_id:
                break
            cur_ids = next_id  # <-- decode step: feed only the 1 new token
            # Yield to the event loop so SSE actually flushes.
            await asyncio.sleep(0)

        yield "data: {\"done\": true}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
