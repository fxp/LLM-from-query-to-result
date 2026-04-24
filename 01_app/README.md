# L1 · App 层

**一句话**：把用户在浏览器里敲的字，变成一条 HTTP 请求送到 Agent 手上；再把 Agent 一个 token 一个 token 吐出来的结果，流式画回浏览器。

## 为什么是这样

ChatGPT 给人"一个字一个字蹦出来"的感觉，不是界面动画，而是**真的**一个字一个字从服务端到客户端传过来的。实现这件事最简单的技术是 **SSE (Server-Sent Events)**：一条长连接的 HTTP 响应，服务端 flush 一行，浏览器就收到一行。

```
浏览器                              后端
  │                                  │
  │  POST /chat  {query: "..."}      │
  │ ───────────────────────────────▶ │
  │                                  │  agent.run(query) 开始
  │  data: {"type":"token","v":"好"} │
  │ ◀─────────────────────────────── │  yield 一个 token
  │  data: {"type":"token","v":"的"} │
  │ ◀─────────────────────────────── │  yield 一个 token
  │  data: {"type":"tool","name":"write_file","path":"index.html"}
  │ ◀─────────────────────────────── │  agent 触发 tool call
  │  data: {"type":"done"}           │
  │ ◀─────────────────────────────── │
```

所以 L1 其实是两件事：
1. **一个 HTML 页面**，显示消息气泡，用 `fetch` + `ReadableStream` 读 SSE，每收到一个 token 就 append 到当前消息。
2. **一个 HTTP server**，接收 query，调用 L2 的 agent（下一层），把 agent 产出的事件一条条 flush 回去。

## 目录

```
01_app/
├── backend/
│   └── main.py          # FastAPI + SSE，~80 行
├── frontend/
│   └── index.html       # 单文件纯 HTML + JS，~90 行
└── README.md
```

## 怎么跑

```bash
export ANTHROPIC_API_KEY=sk-...
cd 01_app
uvicorn backend.main:app --reload --port 8000
# 浏览器打开 http://localhost:8000
```

输入 `帮我做一个 Todo 网站`，你会看到：

- 先是几段文本 token 流式涌出（Agent 在解释/规划）
- 然后出现 `🔧 write_file: index.html` 这样的 tool 事件条
- 继续交替若干轮
- 最后生成的文件落在 `generated/` 目录里，浏览器可以直接打开

## 和其他层的接口

- **往上（用户）**：HTTP + SSE。
- **往下（L2 Agent）**：一个 Python 生成器 `agent.run(query) -> Iterator[Event]`，每个 Event 是 `{"type": "token"|"tool"|"done", ...}`。

## 这一层的"最小"在哪里

- 没用前端框架（纯 HTML + fetch streaming），避免 npm 和 bundler 把故事搞乱。
- 没做鉴权、没做会话持久化、没做多用户。一个 query = 一个请求 = 一个 agent 实例。真实产品里这些都在 L1 这一层展开（JWT、Redis session、rate limit 等），但那是工程问题不是架构问题。
