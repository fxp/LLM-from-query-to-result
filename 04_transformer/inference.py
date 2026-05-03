"""L4 inference demo: load GPT-2, run one forward, print shape of every layer,
pick the argmax of the last position as the "next token".
"""
from __future__ import annotations

import sys

import torch

import tokenizer
from model import GPT


def main(prompt: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT.from_pretrained("gpt2").to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded gpt2: {n/1e6:.2f}M params  (device={device})")

    ids = tokenizer.encode(prompt)
    print(f"tokens: {ids}  ({len(ids)} tokens)")

    x = torch.tensor([ids], device=device)

    print("\nforward pass, one step:")
    with torch.no_grad():
        logits = model(x, verbose=True)

    next_id = int(logits[0, -1].argmax().item())
    piece = tokenizer.decode_one(next_id)
    print(f"\nargmax at last pos -> {next_id}  ({piece!r})")
    print(f"\npredicted next token: {piece!r}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Hello, I am")
