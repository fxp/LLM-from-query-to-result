# 04 · L8：Chat 客户端的最小本质

> [← L7 web app](03-L1-app.md) ｜ 代码：[`02_agent/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/02_agent) ｜ [下一篇 →](05-L3-inference-server.md)

L8 在我心里是这本书最有故事的一层。**它一开始有 200 行 + Anthropic SDK 依赖，最后变成 ~80 行 + 标准库**——不是因为我写得更好了，而是因为我意识到这一层不应该做 agent。

## 第一版：tool-use agent loop（已删）

repo 最早有过一个 `tools.py`，里面有 `write_file` / `run_shell` 两个 tool，加上一个 200 行的 `agent.py` 用 Anthropic 的 tool-use API 跑 Plan→Act→Observe 循环。当时的"贯穿例子"是用户输入 "帮我做一个 Todo 网站"，agent 通过几轮 tool call 写出一个 index.html + Flask 后端。

很酷。删了。

为什么？因为这个 repo 的整条招牌是 **"LLM from scratch"**——L2 自己实现 GPT-2 small (124M, 2019)，然后 L6 host 它给 L8 用。但 GPT-2 做不了 tool-use agent：

1. 它是 base LM，没有 instruction tuning，**不会**输出结构化 `tool_use` JSON
2. 即使硬解析，它**写不出**能跑的网站
3. 所以 agent loop 那 200 行是给 Claude / GPT-4 服务的——但那把"from scratch"招牌打成了"我们用别人家的 model"

我面对一个选择：
- **(a)** 把 L2 升级到能 tool-use 的 instruct 模型（Qwen-2.5-Coder 等）。意味着重写 BPE 适配新 tokenizer，重写 KV cache 适配新模型变体，改训练循环。一个月的活儿。
- **(b)** 删掉 agent loop，承认这个 repo 不演示 agent，而是演示**chat 客户端**。10 分钟。

选 (b)。

## 第二版：纯 chat 客户端

L8 现在做三件事，加起来 ~80 行：

1. **Build prompt**：把用户 query 包成 LM 能续写的形式
2. **POST → L6**：发 HTTP 请求触发推理
3. **Relay SSE**：把 L6 的 token 流翻译成 L7 要的 event 流

整段 [`02_agent/agent.py`](https://github.com/fxp/LLM-from-query-to-result/blob/main/02_agent/agent.py)：

```python
import json, os, urllib.request

L3_URL = os.environ.get("L3_URL", "http://localhost:9000/generate")

def build_prompt(query: str) -> str:
    """Frame the user's query for a base LM (no chat template, no tools)."""
    return f"Q: {query}\nA:"

def run_agent(query: str) -> Iterator[dict]:
    payload = json.dumps({
        "prompt": build_prompt(query),
        "max_tokens": 64,
        "temperature": 0.8,
    }).encode()
    req = urllib.request.Request(L3_URL, data=payload,
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                line = raw.decode().rstrip()
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("done"):
                    yield {"type": "done"}
                    return
                if "token" in event:
                    yield {"type": "token", "v": event["token"]}
    except urllib.error.URLError as exc:
        yield {"type": "error", "message": f"can't reach L6 at {L3_URL}: {exc.reason}"}
```

整个 module 一个外部依赖：`urllib`。Python 标准库，没装啥。

## 关键 1：Prompt template

```python
return f"Q: {query}\nA:"
```

这一行决定了模型怎么续写。GPT-2 是 **base model**，它不知道 "user" / "assistant" 这种概念——它只能续写文本。所以我们得伪装成"训练数据里的样子"让它续写出答案。

`Q: ...\nA:` 这种格式直接对齐 [L4](02-L0.5-sft.md) SFT 数据：

```json
{"q": "What is the capital of France?", "a": "Paris."}
```

被构造成训练样本：

```
Q: What is the capital of France?
A: Paris.<|endoftext|>
```

L8 build 的 prompt 就是 `Q: <query>\nA:`，模型续写 ` Paris.` + EOT，停止。完美对接。

> ⚠️ **chat template 的复杂世界**：现代 instruct 模型（Llama 3、Qwen2.5、ChatGPT）有更复杂的 chat template，比如：
> ```
> <|im_start|>system\nYou are a helpful assistant.<|im_end|>
> <|im_start|>user\nWhat is the capital of France?<|im_end|>
> <|im_start|>assistant\n
> ```
> 不同模型用不同 template，**用错 template 模型会傻掉**。这个 repo 故意保持最简（单 turn `Q:/A:`），因为我们 host 的就是没 chat template 概念的 base model。换 instruct 模型时改 `build_prompt` 一行。

## 关键 2：HTTP 而不是函数调用

L7 调 L8 是 Python 函数调用（同进程）。但 L8 调 L6 是 **HTTP**：

```python
urllib.request.urlopen("http://localhost:9000/generate", data=payload)
```

为什么？因为 **L6 是个独立服务**——它有自己的生命周期（model 加载几秒）、自己的资源（GPU memory），可能跟 L7 跑在不同机器上。

这个边界在生产里很重要：
- L7 可能 horizontal scale（多个 web server 实例）
- L6 是垂直 scale（一个有 GPU 的大机器）
- L8 + L7 同进程：开销小，简单
- L8 → L6：HTTP 跨进程/跨机器，付一次网络往返开销，换来横向解耦

我们 demo 里 L7 + L8 都在 `01_app/backend/main.py` 这个 uvicorn 进程，L6 是 `03_model/server.py` 另一个 uvicorn 进程。两个进程独立启动、独立崩溃。

## 关键 3：流式 relay

L6 SSE 的格式是 `data: {"token": " Paris"}\n\n`（每一个 token 一帧）。L7 要的格式是 `{"type": "token", "v": " Paris"}`（每个 event 一个 dict）。L8 干的就是这个翻译。

```python
for raw in resp:                      # urllib 给 line iterator，自动按 \n 切
    line = raw.decode().rstrip()
    if not line.startswith("data: "):
        continue                       # SSE keep-alive 帧或空行，跳过
    event = json.loads(line[6:])       # parse "data: {...}"
    if event.get("done"):
        yield {"type": "done"}
        return
    if "token" in event:
        yield {"type": "token", "v": event["token"]}
```

注意 `for raw in resp` 是 Python 标准库的隐藏宝藏：`http.client.HTTPResponse` 实现了行迭代器，按 `\n` 切。每收到一行就执行一次循环——**实时的，不缓存**。

这意味着 L6 yield 一个 token，L8 立刻 yield 一个 event，L7 立刻 yield 一个 SSE 帧，浏览器立刻 render。**端到端没有任何 buffer**。

## 故意没有的东西

`run_agent` 大概省略了 100 个东西。诚实清单：

- **没有对话历史**：每次 query 都是 one-shot。要做多轮就把历史拼到 prompt 里：`f"Q: {q1}\nA: {a1}\n\nQ: {q2}\nA:"`。
- **没有重试 / 限流 / circuit breaker**：production agent 必须有这些。
- **没有 streaming 错误恢复**：如果中间某帧 parse 错了，整个 stream 就挂了。
- **没有用量追踪**：tokens 数、延迟、错误率都不记。
- **没有 chat template**：base GPT-2 不需要。换 instruct 模型时要在 `build_prompt` 里加 `<|im_start|>` / `[INST]` 这类 token。

为什么省略？因为这一层的**本质**就是 "build prompt → POST → relay"。这三件事是 chat 客户端不可分的最小核（minimum viable kernel）。其他都是产品/可靠性层。

## 这种"删掉再删掉"的设计美学

我反思 L8 这一层的过程，写下了一条原则：

> **如果你能用一个外部 API 实现某个东西，那个东西就不是 from-scratch 的核心。把它删掉。**

第一版的 agent loop 用 Anthropic API + tool use 实现，它工作得很好——但那意味着这一层的核心智能在 Anthropic 那边，本 repo 只是个壳。删掉之后，留下的"build prompt → POST → relay" 才是 chat 客户端真正的本质——一个**协议适配器**。

这个原则也体现在其他层：
- L6 把 transformers runtime 删掉了 → 留下 KV cache 自己写
- L2 把 tiktoken 删掉了 → 留下 BPE 自己写
- L3 把 OpenAI 权重删掉了（默认）→ 留下训练循环本身

每删一次，"自己写"的部分就更明显。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `L3_URL` | `http://localhost:9000/generate` | L6 推理端点 |
| `L2_MAX_TOKENS` | `64` | 最多生成多少 token |
| `L2_TEMPERATURE` | `0.8` | 0 = greedy argmax；越大越随机 |

测 SFT 模型时把 temperature 设成 0：

```bash
L2_TEMPERATURE=0 python agent.py "What is the capital of France?"
# →  Paris.<|endoftext|>
```

Temperature=0 时 model 一定输出最高概率 token，对 well-trained Q/A 几乎肯定正确。Temperature=0.8 时偶尔会答 "world." 之类的——sampling 噪音。

## 实测：直接调 L8

需要 L6 在 background 跑：

```bash
# 终端 1
cd 03_model && python server.py

# 终端 2
cd 02_agent && python agent.py "What is the capital of France?"
```

输出（以 SFT'd 124M base 为例）：

```
[L8] prompt: 'Q: What is the capital of France?\nA:'
[L8] streaming from http://localhost:9000/generate ...

 Paris.<|endoftext|>

[done]
```

每一行：
- prompt 显示 build 出来的字符串
- streaming 显示连接状态
- 然后逐字输出 token（含 `<|endoftext|>` 这个 stop signal）
- `[done]` 是循环正常结束

每个 token 来自 [L6](05-L3-inference-server.md)，L6 调 [L2 GPT.step()](06-L4-transformer.md)，每个 token 都是我们自己训的 weights 算出来的。

## 下一篇

L8 把 prompt POST 给 L6。L6 是个 FastAPI 服务，加载 ckpt，跑 forward，流式吐 token——不依赖 transformers runtime。我们自己写了 KV cache，下一篇细讲。

[L6 — 自己写 KV cache 的推理服务 →](05-L3-inference-server.md)
