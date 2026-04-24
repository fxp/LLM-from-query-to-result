"""L2 tools: the minimal toolbelt the agent is allowed to use.

Two tools, chosen because together they can build any static + simple
backend website:

    write_file(path, content)   # create/overwrite a file
    run_shell(cmd)              # run a shell command, capture stdout/stderr

Both are sandboxed to a `work_dir` the caller supplies.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


TOOL_SCHEMAS = [
    {
        "name": "write_file",
        "description": "Create or overwrite a file inside the work directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path inside work_dir."},
                "content": {"type": "string", "description": "Full file contents."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_shell",
        "description": "Run a shell command inside work_dir. 30s timeout.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command to run."},
            },
            "required": ["cmd"],
        },
    },
]


@dataclass
class ToolResult:
    ok: bool
    summary: str   # one-liner for the UI
    output: str    # full text returned to the model


def _safe_path(work_dir: Path, rel: str) -> Path:
    """Resolve `rel` against `work_dir` and refuse paths that escape it."""
    target = (work_dir / rel).resolve()
    if not str(target).startswith(str(work_dir.resolve())):
        raise ValueError(f"path escapes work_dir: {rel}")
    return target


def write_file(work_dir: Path, path: str, content: str) -> ToolResult:
    target = _safe_path(work_dir, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return ToolResult(
        ok=True,
        summary=f"wrote {len(content)} bytes to {path}",
        output=f"OK. {len(content)} bytes written to {path}.",
    )


def run_shell(work_dir: Path, cmd: str) -> ToolResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=work_dir, capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, summary="timeout (30s)", output="TIMEOUT after 30s")
    out = (proc.stdout + proc.stderr).strip()
    # Truncate to keep the context small — in a real agent you'd page this.
    if len(out) > 4000:
        out = out[:4000] + "\n...[truncated]"
    return ToolResult(
        ok=proc.returncode == 0,
        summary=f"exit {proc.returncode}",
        output=f"exit={proc.returncode}\n{out}",
    )


DISPATCH = {"write_file": write_file, "run_shell": run_shell}


def call(name: str, args: dict, work_dir: Path) -> ToolResult:
    fn = DISPATCH.get(name)
    if fn is None:
        return ToolResult(ok=False, summary=f"unknown tool {name}", output=f"no such tool: {name}")
    return fn(work_dir=work_dir, **args)
