# 03 · L1：浏览器里那一个个蹦出来的字

> [← L0.5 SFT](02-L0.5-sft.md) ｜ 代码：[`01_app/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/01_app) ｜ [下一篇 →](04-L2-chat-client.md)

ChatGPT 给人"一个字一个字蹦出来"的感觉，**不是**界面动画。是真的——服务端真的一个字一个字地把 token 通过网络送到浏览器，浏览器收到一个就 append 一个。这一篇就讲这件事的实现。

整个 L1 = 一个 ~80 行的 HTML 页面 + 一个 ~50 行的 FastAPI backend。少到我可以把核心代码全贴进来。

## 真相：SSE (Server-Sent Events)

实现"流式输出"最简单的技术叫 SSE：

- 一条普通的 HTTP GET/POST 响应
- 服务端不一次性写完 body，而是 keep-alive 慢慢写
- 每一段写一个 `data: <json>\n\n` 的"事件帧"
- 浏览器用 `EventSource` 或 `fetch` + `ReadableStream` 一段一段读

不是 WebSocket，不是 long-polling，不是 SignalR。就是普通 HTTP，慢慢写。

为什么 ChatGPT 用 SSE 而不是 WebSocket？SSE 是**单向**（服务端推浏览器），刚好匹配 LLM 输出的特性——浏览器不需要往服务端推 token。WebSocket 双向，但带来更多复杂性（重连、心跳、协议帧），对于"流式输出"这一个用例没必要。

## Backend (FastAPI)

整个 `01_app/backend/main.py` 50 行：

```python
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from agent import run_agent  # the L2 chat client

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

@app.get("/")
def index():
    return FileResponse("frontend/index.html")

@app.post("/chat")
def chat(req: ChatRequest):
    def sse():
        try:
            for event in run_agent(req.query):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream")
```

关键三行：

1. `def sse(): ... yield ...` — 一个 Python generator。每 `yield` 一次，FastAPI 给浏览器写一段。
2. `for event in run_agent(req.query):` — `run_agent` 是 L2 那个 chat 客户端（下一篇讲）。它是另一个 generator，每流式收到一个 token 就 yield 一个 event dict。
3. `StreamingResponse(sse(), media_type="text/event-stream")` — FastAPI 知道这是 SSE，会设对的 HTTP header，并在 generator 每 yield 一次时立刻 flush 到 socket。

**没有特殊的"streaming framework"**。Python generator + FastAPI 内置支持，就够了。

## Frontend (vanilla JS)

`01_app/frontend/index.html` 80 行。核心：

```html
<form id="f">
  <input type="text" id="q" />
  <button>发送</button>
</form>
<div id="log"></div>

<script>
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = document.getElementById('q').value.trim();
  if (!query) return;

  const resp = await fetch('/chat', {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({query}),
  });

  // Stream SSE: split on "\n\n", each chunk starts with "data: "
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, i); buf = buf.slice(i + 2);
      if (!frame.startsWith('data: ')) continue;
      const ev = JSON.parse(frame.slice(6));
      render(ev);
    }
  }
});
</script>
```

四件事：

1. **`fetch('/chat', ...)`** — 不用任何前端框架。原生 fetch。
2. **`resp.body.getReader()`** — 拿一个流读取器。`fetch` 默认就是流式响应，body 是个 `ReadableStream`。
3. **逐 chunk 读 + 按 `\n\n` 分帧** — SSE 格式：每个 event 用 `data: <json>\n\n` 包，相邻 event 之间一个空行。所以我维护一个 buffer，每收到 chunk 拼接进去，找到 `\n\n` 就切一个 frame 出来 parse。
4. **`render(ev)`** — 拿到 event 后追加到页面：

```javascript
function render(ev) {
  if (ev.type === 'token') {
    log.append(ev.v);   // 直接追加到 #log，浏览器自动渲染
  } else if (ev.type === 'done') {
    log.append('✓ done');
  } else if (ev.type === 'error') {
    /* 红色显示 */
  }
}
```

**没有 React、没有 Vue、没有 EventSource API**。`fetch` + `ReadableStream` + `TextDecoder`——三个原生 API，搞定。

## 为什么不用 EventSource？

浏览器有个内置的 `EventSource` API，专门读 SSE。它更短：

```javascript
const es = new EventSource('/chat?query=...');
es.onmessage = (e) => render(JSON.parse(e.data));
```

但 `EventSource` 只能 GET，body 通过 query string 传——chat query 可能很长（多轮对话历史），URL 字符串有长度限制，并且 query 显示在 server log / 浏览器历史里不优雅。

所以用 `fetch + POST` 更合适。代价是要自己分帧。15 行手写 streaming parser 的事。

## 为什么 80 行就够

ChatGPT 的前端肯定不止 80 行。它有：
- 用户登录、会话持久化
- 多轮对话历史的渲染
- Markdown / 代码块语法高亮
- 文件上传、图片预览
- 移动端响应式
- 主题切换、可访问性
- 错误恢复、重试逻辑
- 用量统计、付费弹窗

这些**都不是 LLM 流式输出本身需要的**。它们是产品特性。一个 chat 服务的 streaming UI 核心，就是 `fetch + ReadableStream + append`，80 行讲清楚。

## 实测时序

跑：

```bash
# 终端 1：起 L3 推理服务（下一层讲）
cd 03_model && python server.py

# 终端 2：起 L1 web app
cd 01_app && uvicorn backend.main:app --reload
```

浏览器打开 `http://localhost:8000`，输入 "What is the capital of France?"，按发送：

```
t=0 ms      用户按 Enter
t=1 ms      fetch('/chat') 发出 POST
t=5 ms      L1 backend 收到，调 run_agent (L2)
t=6 ms      L2 build prompt → POST /generate (L3)
t=10 ms     L3 BPE tokenize prompt
t=12 ms     L3 prefill GPU forward → 第 1 个 token "Paris" 出来
t=12.5 ms   L3 yield SSE frame → L2 收到
t=12.6 ms   L2 yield event → L1 backend
t=12.7 ms   L1 backend yield SSE frame → 浏览器
t=13 ms     浏览器 render(' Paris') → 屏幕上看到了 "Paris"
t=15 ms     L3 decode 第 2 个 token "."
t=17 ms     ...
```

总共 ~17 ms 就看到了 "Paris."（5090 上）。**这种延迟已经感觉不出"等"，几乎就是"敲完就出来"**。

> 🤔 **思考题**：为什么 ChatGPT 的"打字感"通常更慢——你能看到单词一个个出来？因为它的 model 大（GPT-4 1.7T 参数级），单个 token 的 forward 在多 GPU 集群上要 ~30-50 ms，所以**用户看到的速度是被 token 生成速度卡住的**，不是网络。

## 这一层的"最小"在哪里

- **没有用前端框架**：避免 npm + bundler + 一堆配置。一个 .html 文件，浏览器直接打开。
- **没有鉴权**：一个 query = 一个请求。生产里这一层要 JWT、session、rate limit、CSRF 防护——但那是工程问题不是架构问题。
- **没有持久化**：刷新页面对话就没了。要做 history 就 `localStorage` 或 server-side session。
- **没有 markdown 渲染**：直接 `<div>{token}</div>`。要渲染就接 marked.js 或 markdown-it。
- **没有错误恢复**：网断了就挂。生产里要重试、断线重连。

## 接口（往下）

L1 跟 L2 通过一个 Python generator 调用：

```python
for event in run_agent(query):
    yield f"data: {json.dumps(event)}\n\n"
```

`run_agent` 是 L2 实现的接口：

```python
def run_agent(query: str) -> Iterator[dict]:
    """yields events: {type: 'token', v: str} | {type: 'done'} | {type: 'error', message: str}"""
```

L1 不关心 L2 内部怎么工作——它只关心拿到事件流，转成 SSE 推给浏览器。这是一个**进程内函数调用**（不是 HTTP）——L1 + L2 跑在同一个 uvicorn 进程里，因为 L2 也很轻（就是个 HTTP 客户端），没必要拆出来再加一层网络。

下一篇我们看 L2 内部：它怎么把 query 变成 L3 能理解的 prompt，怎么把 L3 的 SSE 翻译成 L1 要的 event 流。

## 下一篇

[L2 — Chat 客户端的最小本质 →](04-L2-chat-client.md)
