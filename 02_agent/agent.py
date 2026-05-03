"""L2 Agent: a thin streaming client to the L3 inference server.

Why no LLM API call? This whole repo is "LLM from scratch". L4 implements
GPT-2 by hand; L3 hosts it as an HTTP service. So L2 here just does what a
real chat client does: build a prompt, stream tokens back. No external
provider, no tool use — every token you see was produced by the GPT-2
forward pass in L4 running on top of L5's matmul.

Trade-off this makes explicit: GPT-2 small (124M, 2019) is too weak to act
as a coding agent. It can't reliably emit tool_use JSON, and even if it
could, it can't write a Todo website. The "agent loop" (Plan → Act →
Observe with write_file / run_shell tools) is a real architecture, but it
needs an instruction-tuned, tool-use-capable model — that belongs to a
separate exercise once you swap GPT-2 for, say, a local Qwen-2.5-Coder.

So in this repo L2 demonstrates one specific thing: how a streaming chat
client wraps an inference server. That's it. The interesting "from
scratch" content is in L3/L4/L5.

Emits the same event shape L1 expects:
    {type: 'token', v: str}
    {type: 'done'}
    {type: 'error', message: str}
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

L3_URL = os.environ.get("L3_URL", "http://localhost:9000/generate")
MAX_TOKENS = int(os.environ.get("L2_MAX_TOKENS", "64"))
TEMPERATURE = float(os.environ.get("L2_TEMPERATURE", "0.8"))


def build_prompt(query: str) -> str:
    """Frame the user's query for a base LM (no chat template, no tools).

    GPT-2 is a base model — it has no notion of "user" / "assistant", it
    just continues text. So we frame the query as the start of a written
    answer and let it continue. This is the same trick OpenAI used in the
    GPT-2/GPT-3 era before instruction-tuning existed.
    """
    return f"Question: {query}\nAnswer:"


def run_agent(query: str, work_dir: Path | None = None) -> Iterator[dict]:
    """Stream tokens from L3 back to the caller. `work_dir` is unused
    (kept for L1 ABI compatibility — once tool use is added back it'll
    matter again)."""
    payload = json.dumps({
        "prompt": build_prompt(query),
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }).encode()
    req = urllib.request.Request(
        L3_URL,
        data=payload,
        headers={"content-type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            # L3 streams SSE: "data: {...}\n\n" frames. We parse line by
            # line — each `token` event is one piece of text emitted by
            # one decode step in the transformer.
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("done"):
                    yield {"type": "done"}
                    return
                if "token" in event:
                    yield {"type": "token", "v": event["token"]}
    except urllib.error.URLError as exc:
        yield {
            "type": "error",
            "message": (
                f"can't reach L3 at {L3_URL}: {exc.reason}. "
                f"Start it first: cd 03_model && python server.py"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python agent.py '<your query>'", file=sys.stderr)
        sys.exit(2)
    print(f"[L2] prompt: {build_prompt(sys.argv[1])!r}")
    print(f"[L2] streaming from {L3_URL} ...\n")
    for ev in run_agent(sys.argv[1]):
        if ev["type"] == "token":
            sys.stdout.write(ev["v"]); sys.stdout.flush()
        elif ev["type"] == "done":
            print("\n\n[done]")
        elif ev["type"] == "error":
            print(f"\n[error] {ev['message']}", file=sys.stderr)
            sys.exit(1)
