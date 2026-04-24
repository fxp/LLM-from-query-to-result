"""L2 Agent: Plan → Act → Observe loop, backed by Claude's tool use API.

This file is deliberately small — all the intelligence lives inside the
model call. The loop does only three things:

    1. Send the running conversation to the model.
    2. If the model returned tool_use blocks, execute them and append
       the results as a user turn.
    3. If the model returned only text (no tool_use), we're done.

Emits a stream of dict events so L1 can render progress.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

import anthropic

import tools

MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 20

SYSTEM_PROMPT = """You are a coding agent. The user will ask for a small website or
script. Use the `write_file` and `run_shell` tools to produce the files in the
working directory. Keep it small and self-contained: a single index.html plus
at most one backend file. Do NOT run long-lived servers. When the task is
complete, respond with a short summary (no more tool_use calls).
"""


def run_agent(query: str, work_dir: Path) -> Iterator[dict]:
    """Yield events: {type: 'token'|'tool'|'tool_result'|'done'|'error', ...}."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": query}]

    for _ in range(MAX_ITERATIONS):
        # === One trip down to L3 ===
        # Under the hood this is an HTTPS request; the response body is
        # L3 streaming tokens one by one (L3 runs L4 runs L5).
        #
        # We iterate raw events just for text deltas (so L1 can render
        # tokens live); at the end we ask the SDK for the fully-parsed
        # message so we don't have to reassemble tool_use JSON ourselves.
        with client.messages.stream(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools.TOOL_SCHEMAS,
            messages=messages,
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield {"type": "token", "v": event.delta.text}
            final = stream.get_final_message()

        # Convert the SDK's typed blocks into the dict form the API wants
        # when we send the conversation back next turn.
        assistant_content: list[dict] = []
        tool_uses: list[dict] = []
        for block in final.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                tool_uses.append({"id": block.id, "name": block.name, "input": block.input})
        messages.append({"role": "assistant", "content": assistant_content})

        if not tool_uses:
            yield {"type": "done"}
            return

        # === Execute tools, one per block; build a single user turn back ===
        tool_results: list[dict] = []
        for tu in tool_uses:
            yield {"type": "tool", "name": tu["name"], "summary": _summarize_input(tu["name"], tu["input"])}
            result = tools.call(tu["name"], tu["input"], work_dir=work_dir)
            yield {"type": "tool_result", "summary": result.summary}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result.output,
                "is_error": not result.ok,
            })
        messages.append({"role": "user", "content": tool_results})

    yield {"type": "error", "message": f"hit MAX_ITERATIONS={MAX_ITERATIONS}"}


def _summarize_input(name: str, inp: dict) -> str:
    if name == "write_file":
        return f'path="{inp.get("path","")}"'
    if name == "run_shell":
        cmd = inp.get("cmd", "")
        return f'cmd="{cmd[:60]}{"..." if len(cmd) > 60 else ""}"'
    return ""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python agent.py '<your query>'", file=sys.stderr)
        sys.exit(2)
    work_dir = Path(__file__).resolve().parents[1] / "generated"
    for ev in run_agent(sys.argv[1], work_dir=work_dir):
        t = ev["type"]
        if t == "token":
            sys.stdout.write(ev["v"]); sys.stdout.flush()
        elif t == "tool":
            print(f"\n[tool_use] {ev['name']}({ev['summary']})")
        elif t == "tool_result":
            print(f"[tool_result] {ev['summary']}")
        elif t == "done":
            print("\n[done]")
        elif t == "error":
            print(f"\n[error] {ev['message']}", file=sys.stderr)
