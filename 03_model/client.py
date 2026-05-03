"""L3 client: send a prompt, print tokens as they stream back."""
from __future__ import annotations

import json
import sys
import urllib.request


def stream(prompt: str, max_tokens: int = 32) -> None:
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(
        "http://localhost:9000/generate",
        data=body,
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        print(prompt, end="", flush=True)
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            if payload.get("done"):
                print()
                return
            print(payload["token"], end="", flush=True)


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    stream(prompt, max_tokens=int(sys.argv[2]) if len(sys.argv) > 2 else 32)
