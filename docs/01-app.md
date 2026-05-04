# L7 · App / Web UI

**一句话**：把用户在浏览器里敲的字，变成一条 HTTP 请求送到 L8 客户端手上；再把回来的 token 一个一个流式画回浏览器。

源码：[`01_app/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/01_app)

## 为什么是这样

ChatGPT 给人"一个字一个字蹦出来"的感觉，不是界面动画，而是**真的**一个字一个字从服务端到客户端传过来的。实现这件事最简单的技术是 **SSE (Server-Sent Events)**：一条长连接的 HTTP 响应，服务端 flush 一行，浏览器就收到一行。

```
浏览器                              后端
  │                                  │
  │  POST /chat  {query: "..."}      │
  │ ───────────────────────────────▶ │
  │                                  │  run_agent(query) 开始
  │  data: {"type":"token","v":"好"} │
  │ ◀─────────────────────────────── │  yield 一个 token
  │  data: {"type":"token","v":"的"} │
  │ ◀─────────────────────────────── │  yield 一个 token
  │   ...                            │
  │  data: {"type":"done"}           │
  │ ◀─────────────────────────────── │
```

所以 L7 其实是两件事：

1. **一个 HTML 页面**，显示消息气泡，用 `fetch` + `ReadableStream` 读 SSE，每收到一个 token 就 append 到当前消息。
2. **一个 HTTP server**，接收 query，调用 L8 的 `run_agent`（下一层），把它产出的事件一条条 flush 回去。

## 目录

```
01_app/
├── backend/
│   └── main.py          # FastAPI + SSE，~50 行
├── frontend/
│   └── index.html       # 单文件纯 HTML + JS，~80 行
└── README.md
```

- [`backend/main.py`](https://github.com/fxp/LLM-from-query-to-result/blob/main/01_app/backend/main.py)
- [`frontend/index.html`](https://github.com/fxp/LLM-from-query-to-result/blob/main/01_app/frontend/index.html)

## 怎么跑

需要两个终端——L8 依赖 L6，必须先起 L6：

```bash
# 终端 1：起 L6 推理服务（GPT-2 small）
cd 03_model && python server.py

# 终端 2：起 L7 web app
cd 01_app && uvicorn backend.main:app --reload --port 8000
# 浏览器打开 http://localhost:8000
```

输入 `What is the capital of France?`，你会看到 token 一个一个流式涌出。每个 token 都是本地 GPT-2 (L2) 的 forward pass 在 L1 上算出来的——零外部 API。

> 注：当前版本不做 agentic 任务（`write_file` / `run_shell`），因为 GPT-2 124M 不具备 tool-use 能力。详见 [L8 · Agent 循环](02-agent.md) "为什么不是 Agent"。

## 和其他层的接口

- **往上（用户）**：HTTP + SSE。
- **往下（L8 客户端）**：一个 Python 生成器 `run_agent(query) -> Iterator[Event]`，每个 Event 是 `{"type": "token"|"done"|"error", ...}`。

## 这一层的"最小"在哪里

- 没用前端框架（纯 HTML + fetch streaming），避免 npm 和 bundler 把故事搞乱。
- 没做鉴权、没做会话持久化、没做多用户。一个 query = 一个请求 = 一次推理。真实产品里这些都在 L7 这一层展开（JWT、Redis session、rate limit 等），但那是工程问题不是架构问题。

---

下一层 → [L8 · Agent 循环](02-agent.md)
