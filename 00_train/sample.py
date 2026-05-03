"""L0 sample: load the trained checkpoint and generate text.

Quick way to eyeball whether training actually worked, without going
through L3/L2/L1. Mirrors what 04_transformer/inference.py does, but
with our small from-scratch checkpoint instead of the OpenAI gpt2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "04_transformer"))
from model import GPT, GPTConfig  # noqa: E402
import tokenizer  # noqa: E402

CKPT = Path(__file__).resolve().parent / "out" / "ckpt.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def generate(model: GPT, prompt: str, max_new: int = 200, temperature: float = 0.8) -> str:
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    for _ in range(max_new):
        # Crop context to block_size — no KV cache here, just the simple loop.
        x_cond = x if x.size(1) <= model.cfg.block_size else x[:, -model.cfg.block_size :]
        logits = model(x_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
    return tokenizer.decode(x[0].tolist())


def main() -> None:
    if not CKPT.exists():
        print(f"checkpoint not found: {CKPT}\nrun: python train.py", file=sys.stderr)
        sys.exit(1)

    blob = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = GPTConfig(**blob["config"])
    model = GPT(cfg).to(DEVICE).eval()
    model.load_state_dict(blob["model"])
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded {CKPT.name}: {n/1e6:.2f}M params  step={blob.get('steps','?')}  "
          f"final_loss={blob.get('final_loss')}")

    prompt = sys.argv[1] if len(sys.argv) > 1 else "ROMEO:"
    print(f"\nprompt: {prompt!r}\n" + "─" * 60)
    print(generate(model, prompt, max_new=200, temperature=0.8))
    print("─" * 60)


if __name__ == "__main__":
    main()
