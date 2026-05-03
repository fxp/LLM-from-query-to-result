# L2 · Agent / Chat 客户端层

**一句话**：把用户的一句话变成 prompt，通过 HTTP 流式问 L3，把吐回来的 token 一个个抛给上层。

源码：[`02_agent/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/02_agent)

## 这一层为什么存在

L4 实现了 transformer，L3 把它包成一个能接受 prompt 的 HTTP 服务。但用户不会直接发 `POST /generate` JSON——他在浏览器里输入 "巴黎在哪"。中间需要一层做：

1. **prompt framing**：base LM（GPT-2）没有 `user` / `assistant` 概念，只能续写文本。要把"巴黎在哪"包装成 `Question: 巴黎在哪\nAnswer:` 这种形式，让模型从语料里学到的"问答"模式接续下去。
2. **streaming relay**：把 L3 的 SSE 帧解析成结构化事件，给 L1 用。
3. **失败兜底**：L3 挂了的话，给 L1 一个清晰的错误。

`agent.py` 就这点东西，~90 行。

## 为什么不是 "Agent"

最初的设计想做完整的 Plan → Act → Observe 循环（用 `write_file` / `run_shell` 真造一个 Todo 网站）。但这个 repo 的招牌是 "LLM from scratch"——L4 自己实现 GPT-2 small (124M, 2019)。GPT-2 做不了 agent：

- 它是 base LM，没有 instruction tuning，不会输出结构化的 `tool_use` JSON
- 即使硬解析，它也写不出能跑的网站

所以这里诚实地退回到 "chat completion"——只演示**chat 客户端怎么和推理服务对话**。完整的 agentic 循环需要换一个能 tool-use 的 instruct 模型（Qwen-2.5-Coder、Llama-3.1-Instruct 等），那是另一个练习。

## 目录

```
02_agent/
├── agent.py        # 流式 chat 客户端，~90 行
└── README.md
```

- [`agent.py`](https://github.com/fxp/LLM-from-query-to-result/blob/main/02_agent/agent.py)

## 怎么跑

先在另一个终端起 L3：

```bash
cd 03_model && python server.py
```

然后：

```bash
cd 02_agent && python agent.py "What is the capital of France?"
```

输出：

```
[L2] prompt: 'Question: What is the capital of France?\nAnswer:'
[L2] streaming from http://localhost:9000/generate ...

 The capital is the capital of France.
Zachary N. Egbert: ...

[done]
```

GPT-2 124M 答得很差是正常的——这台模型 2019 年发布，没经过 RLHF/SFT。重点不是答案多好，是**这条链路里没有任何外部 API**：每个 token 都是 L4 的 GPT-2 forward pass 在 L5 的 matmul 上算出来的。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `L3_URL` | `http://localhost:9000/generate` | L3 推理端点 |
| `L2_MAX_TOKENS` | `64` | 最多生成多少 token |
| `L2_TEMPERATURE` | `0.8` | 0 = greedy argmax；越大越随机 |

## 和其他层的接口

- **往上（L1 App）**：`run_agent(query) -> Iterator[Event]`，事件类型 `token / done / error`。
- **往下（L3 Model）**：一个 `POST /generate` SSE 请求。这个 HTTP 调用就是"让 L3 跑一次 transformer forward，把 token stream 回来"。

## 这一层的"最小"在哪里

- **没有对话历史**：每次 `run_agent` 都是一次 one-shot。要做多轮，把历史拼到 prompt 里就行（L3 会重新 prefill；要省钱就上 prefix cache，那是 L3 的活）。
- **没有 tool use**：见上面"为什么不是 Agent"。
- **没有 chat template**：base GPT-2 不需要。换 instruct 模型时要在 `build_prompt` 里加上 `<|im_start|>` / `[INST]` 这类 token。
- **没有重试 / 限流 / 指标**：production agent 需要这些；本 demo 故意省掉，让"streaming chat client 的本质"露出来。

---

[← L1 · App 层](01-app.md) | 下一层 → [L3 · Model 层](03-model.md)
