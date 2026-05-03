"""L1 backend: FastAPI + SSE.

Receives a chat query, forwards it to the L2 agent, and streams agent
events (tokens, tool calls, done) back to the browser as SSE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Make L2 importable without packaging.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "02_agent"))
from agent import run_agent  # noqa: E402

app = FastAPI()
FRONTEND = REPO_ROOT / "01_app" / "frontend" / "index.html"


class ChatRequest(BaseModel):
    query: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND)


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    """Stream agent events as SSE.

    Each event is a JSON object on its own `data:` line, with a blank line
    separator — the SSE format the browser's EventSource / fetch-stream
    expects.
    """
    def sse() -> "Iterator[str]":
        try:
            for event in run_agent(req.query, work_dir=REPO_ROOT / "generated"):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # surface errors to the browser
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
