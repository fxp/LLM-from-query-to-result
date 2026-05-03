"""L0.6 agent tools — calc + lookup.

Both tools take a string argument (parsed from the model's `ACTION:` line)
and return a string (rendered into the next `OBSERVATION:` line).

Kept minimal so a 124M model can plausibly learn to call them after ~150
training traces.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# calc — eval arithmetic expressions in a safe sandbox
# ---------------------------------------------------------------------------

_CALC_PAT = re.compile(r"^[\d+\-*/().\s]+$")


def calc(expr: str) -> str:
    """Eval a simple arithmetic expression. Returns the result as a string,
    or an error message. Safe: only digits, + - * / ( ) and whitespace.

    >>> calc("2 + 2")
    '4'
    >>> calc("23 + 47")
    '70'
    >>> calc("3 * (4 + 5)")
    '27'
    >>> calc("import os")
    'invalid expression'
    """
    expr = expr.strip()
    if not _CALC_PAT.match(expr):
        return "invalid expression"
    try:
        # `eval` with empty globals + locals — no builtins, no names accessible.
        result = eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return "error"
    # Normalize: 70.0 → "70", 0.5 → "0.5"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# ---------------------------------------------------------------------------
# lookup — query a small bundled knowledge base
# ---------------------------------------------------------------------------

_KB_PATH = Path(__file__).resolve().parent / "kb.json"
_kb_cache: dict[str, str] | None = None


def _load_kb() -> dict[str, str]:
    global _kb_cache
    if _kb_cache is None:
        _kb_cache = {k.lower(): v for k, v in json.loads(_KB_PATH.read_text()).items()}
    return _kb_cache


def lookup(key: str) -> str:
    """Look up a fact in the bundled KB. Case-insensitive exact-key match.

    >>> lookup("capital of France")
    'Paris'
    >>> lookup("capital of Japan")
    'Tokyo'
    >>> lookup("does not exist")
    'not found'
    """
    kb = _load_kb()
    return kb.get(key.strip().lower(), "not found")


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

TOOLS: dict[str, Callable[[str], str]] = {
    "calc": calc,
    "lookup": lookup,
}

# Pattern for parsing `tool_name(args)` from a model's ACTION: line.
# Allows nested parens and spaces in args. Captures (tool_name, args).
_ACTION_RE = re.compile(r"^\s*([a-z_]+)\s*\((.*)\)\s*$", re.DOTALL)


def parse_action(action_line: str) -> tuple[str, str] | None:
    """Parse `tool_name(args)` from a single line. Returns (name, args) or
    None if malformed."""
    m = _ACTION_RE.match(action_line)
    if not m:
        return None
    return m.group(1), m.group(2)


def call(action_line: str) -> str:
    """Parse and execute one ACTION line. Returns the OBSERVATION string."""
    parsed = parse_action(action_line)
    if parsed is None:
        return "malformed action"
    name, args = parsed
    fn = TOOLS.get(name)
    if fn is None:
        return f"unknown tool {name!r}"
    try:
        return fn(args)
    except Exception as exc:
        return f"tool error: {type(exc).__name__}"


if __name__ == "__main__":
    # Tiny self-test
    cases = [
        ("calc(2 + 2)", "4"),
        ("calc(23 + 47)", "70"),
        ("calc(3 * (4 + 5))", "27"),
        ("calc(100 / 4)", "25"),
        ("lookup(capital of France)", "Paris"),
        ("lookup(capital of Japan)", "Tokyo"),
        ("lookup(does not exist)", "not found"),
        ("calc(import os)", "invalid expression"),
        ("badtool(x)", "unknown tool 'badtool'"),
        ("not a function", "malformed action"),
    ]
    for action, expected in cases:
        got = call(action)
        ok = got == expected
        mark = "✓" if ok else "✗"
        print(f"  {mark} call({action!r}) = {got!r}  (expected {expected!r})")
