"""L2 — real agent loop (Plan → Act → Observe).

Drives a model that's been SFT'd on ReAct traces (00c_agent_sft/agent.pt).
Generation happens through L3's /generate endpoint, but we control the
loop here: we stop generation at "OBSERVATION:", parse the model's
ACTION, run the actual tool, inject the real observation, and resume
generation. The model emits THOUGHT/ACTION/ANSWER, we run the tool.

Format the model is trained to emit:
    Q: <question>
    THOUGHT: <reasoning>
    ACTION: tool_name(args)
    OBSERVATION: <real tool output goes here>
    THOUGHT: ...                ← optionally another step
    ACTION: ...
    OBSERVATION: ...
    ANSWER: <final><|endoftext|>

Why a loop instead of one-shot generation: at SFT time we put real tool
outputs in OBSERVATION lines. If we just generate continuously, the model
will *hallucinate* OBSERVATION text (it learned "after ACTION there's an
OBSERVATION"). To prevent hallucination, we stop generation right after
"OBSERVATION:" and inject the real tool output. This is the standard
ReAct trick.

Yields events for L1:
    {type: 'thought', v: str}
    {type: 'action', v: str}
    {type: 'observation', v: str}
    {type: 'token', v: str}      (during ANSWER)
    {type: 'done'}
    {type: 'error', message: str}
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

# Reuse 00c_agent_sft/tools.py for tool execution.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "00c_agent_sft"))
import tools  # noqa: E402

L3_URL = os.environ.get("L3_URL", "http://localhost:9000/generate")
MAX_LOOP_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "5"))
MAX_NEW_PER_STEP = int(os.environ.get("AGENT_MAX_NEW", "100"))


def _request_generate(prompt: str, max_tokens: int) -> Iterator[str]:
    """Stream tokens from L3 with greedy decoding (temperature=0).

    Yields token strings one at a time.
    """
    payload = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        L3_URL,
        data=payload,
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev.get("done"):
                return
            if "token" in ev:
                yield ev["token"]


def _generate_until_stop(prompt: str, stop_strings: list[str],
                         max_tokens: int) -> tuple[str, str | None]:
    """Generate from L3, accumulating tokens, until any of `stop_strings`
    appears in the accumulated text — or we hit max_tokens.

    Returns (generated_text_excluding_stop_string, matched_stop_or_None).
    The matched_stop_string is consumed from the stream; the caller
    knows the model output up to that point.
    """
    accumulated = ""
    for tok in _request_generate(prompt, max_tokens):
        accumulated += tok
        for s in stop_strings:
            idx = accumulated.find(s)
            if idx >= 0:
                return accumulated[:idx], s
    return accumulated, None


def run_agent(query: str, work_dir: Path | None = None) -> Iterator[dict]:
    """Run the agent loop. Yields events for L1 to render."""
    prompt = f"Q: {query}\n"
    try:
        for step in range(MAX_LOOP_STEPS):
            # Generate the next THOUGHT + ACTION (or ANSWER if model decides
            # to terminate). Stop when we hit "\nOBSERVATION:" — that means
            # model wants a tool result. Or stop at <|endoftext|> in the
            # ANSWER (which the L3 server emits as a special token string).
            chunk, stop = _generate_until_stop(
                prompt,
                stop_strings=["\nOBSERVATION:", "<|endoftext|>"],
                max_tokens=MAX_NEW_PER_STEP,
            )

            if stop is None:
                # Hit max_tokens without finding either stop — emit what
                # we have as a fallback "answer" event.
                yield {"type": "token", "v": chunk}
                yield {"type": "done"}
                return

            # Accumulated chunk is everything the model wrote before stop.
            # Surface its THOUGHT/ACTION/ANSWER as separate events.
            for piece in _split_emitted(chunk):
                yield piece

            if stop == "<|endoftext|>":
                yield {"type": "done"}
                return

            # stop == "\nOBSERVATION:" → model just emitted an ACTION and
            # is waiting for tool output. Find the action, run it.
            action_line = _last_action(chunk)
            if action_line is None:
                yield {"type": "error",
                       "message": "model wanted observation but no ACTION found"}
                return
            obs = tools.call(action_line)
            yield {"type": "observation", "v": obs}

            # Append the chunk + the OBSERVATION line to prompt and continue.
            # Note: the stop string "\nOBSERVATION:" was consumed; we add it
            # back here, plus the real observation, plus the trailing newline
            # so the next generation starts with `THOUGHT:` or `ANSWER:`.
            prompt = prompt + chunk + f"\nOBSERVATION: {obs}\n"

        yield {"type": "error",
               "message": f"hit MAX_LOOP_STEPS={MAX_LOOP_STEPS}"}

    except urllib.error.URLError as exc:
        yield {"type": "error",
               "message": f"can't reach L3 at {L3_URL}: {exc.reason}. "
                          f"Start it first: cd 03_model && python server.py"}
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}


# --- helpers --------------------------------------------------------------

_THOUGHT_RE = re.compile(r"^THOUGHT:\s*(.+)$", re.MULTILINE)
_ACTION_RE = re.compile(r"^ACTION:\s*(.+)$", re.MULTILINE)
_ANSWER_RE = re.compile(r"^ANSWER:\s*(.+)$", re.MULTILINE | re.DOTALL)


def _split_emitted(chunk: str) -> Iterator[dict]:
    """Surface THOUGHT / ACTION / ANSWER lines as separate events."""
    for m in _THOUGHT_RE.finditer(chunk):
        yield {"type": "thought", "v": m.group(1).strip()}
    for m in _ACTION_RE.finditer(chunk):
        yield {"type": "action", "v": m.group(1).strip()}
    answers = _ANSWER_RE.findall(chunk)
    for ans in answers:
        for char in ans.strip():
            yield {"type": "token", "v": char}


def _last_action(chunk: str) -> str | None:
    """Get the last ACTION line in chunk."""
    matches = _ACTION_RE.findall(chunk)
    return matches[-1].strip() if matches else None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python agent_loop.py '<query>'", file=sys.stderr)
        sys.exit(2)
    print(f"[agent_loop] {sys.argv[1]}")
    print(f"[agent_loop] streaming from {L3_URL} ...\n")
    for ev in run_agent(sys.argv[1]):
        t = ev["type"]
        if t == "thought":
            print(f"💭 {ev['v']}")
        elif t == "action":
            print(f"🔧 {ev['v']}")
        elif t == "observation":
            print(f"   ↳ {ev['v']}")
        elif t == "token":
            sys.stdout.write(ev["v"])
            sys.stdout.flush()
        elif t == "done":
            print("\n[done]")
        elif t == "error":
            print(f"\n[error] {ev['message']}", file=sys.stderr)
            sys.exit(1)
