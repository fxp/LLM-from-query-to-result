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

# AGENT_MODE=1 routes through the ReAct agent loop in agent_loop.py
# (requires L3 to be serving an agent-SFT'd checkpoint from 00c_agent_sft).
AGENT_MODE = os.environ.get("AGENT_MODE", "").lower() in ("1", "true", "yes")


def build_prompt(query: str) -> str:
    """Frame the user's query for a base LM (no chat template, no tools).

    GPT-2 is a base model — it has no notion of "user" / "assistant", it
    just continues text. So we frame the query as the start of a written
    answer and let it continue. This is the same trick OpenAI used in the
    GPT-2/GPT-3 era before instruction-tuning existed.

    Format matches `00b_sft/data.json` so that an SFT'd checkpoint
    trained on the same template behaves predictably.
    """
    return f"Q: {query}\nA:"


def run_agent(query: str, work_dir: Path | None = None) -> Iterator[dict]:
    """Stream events from L3 back to the caller.

    Two modes:
      - default (chat completion): build prompt → POST → relay token events
      - AGENT_MODE=1: route through agent_loop.run_agent (Plan→Act→Observe
        with calc / lookup tools, requires agent-SFT'd checkpoint)
    """
    if AGENT_MODE:
        from agent_loop import run_agent as run_loop  # local import — avoids tools.py side effects
        yield from run_loop(query, work_dir=work_dir)
        return

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
        print("  set AGENT_MODE=1 to use the ReAct agent loop "
              "(needs agent-SFT'd ckpt in L3)", file=sys.stderr)
        sys.exit(2)
    if AGENT_MODE:
        print(f"[L2] AGENT_MODE — running ReAct loop on {L3_URL}\n")
    else:
        print(f"[L2] prompt: {build_prompt(sys.argv[1])!r}")
        print(f"[L2] streaming from {L3_URL} ...\n")
    for ev in run_agent(sys.argv[1]):
        t = ev["type"]
        if t == "token":
            sys.stdout.write(ev["v"]); sys.stdout.flush()
        elif t == "thought":
            print(f"💭 {ev['v']}")
        elif t == "action":
            print(f"🔧 {ev['v']}")
        elif t == "observation":
            print(f"   ↳ {ev['v']}")
        elif t == "done":
            print("\n\n[done]")
        elif t == "error":
            print(f"\n[error] {ev['message']}", file=sys.stderr)
            sys.exit(1)
