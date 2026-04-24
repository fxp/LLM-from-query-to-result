# LLM-from-query-to-result

**从一行 query，到 GPU 上的一次浮点乘法——一本可以跑起来的全栈教科书。**

当你在 ChatGPT 里输入 "帮我做一个 Todo 网站"，按下回车，到屏幕上出现一个能用的网站，这中间到底发生了什么？

这个站点把整条链路切成 5 层，**每一层都有独立的讲解和最小可运行代码**。你可以单独跑任意一层，也可以把它们串起来看完整 trace。

源码：[fxp/LLM-from-query-to-result](https://github.com/fxp/LLM-from-query-to-result)

## 贯穿全 repo 的例子

```
用户 query:  "帮我做一个 Todo 网站"
最终产物:    examples/todo-app/ 下一个能跑的前后端网站
```

这个 query 会穿过 5 层，我们会在每一层把它的"形态"打印出来：
在 L1 它是一串 HTTP bytes，在 L2 它变成了一个 plan + 若干 tool call，
在 L3 它是一个 batch 里的 prompt，在 L4 它是 tensor，在 L5 它是 GPU SM 上的指令。

## 五层架构

```
┌───────────────────────────────────────────────────────────────┐
│ L1  App 层        用户看到的聊天界面 + 后端 SSE 流式输出     │
│     01_app/       (HTML + FastAPI)                            │
├───────────────────────────────────────────────────────────────┤
│ L2  Agent 层      Plan → Act (tool use) → Observe 循环        │
│     02_agent/     (Claude API + write_file/run_shell 工具)   │
├───────────────────────────────────────────────────────────────┤
│ L3  Model 层      推理服务：tokenize / batch / KV cache /    │
│     03_model/     stream (HuggingFace transformers)          │
├───────────────────────────────────────────────────────────────┤
│ L4  Transformer   从零实现 GPT-2：embed / MHA / FFN / LN     │
│     04_transformer/  (PyTorch, ~300 行，可加载 HF 权重)       │
├───────────────────────────────────────────────────────────────┤
│ L5  GPU 层        矩阵乘和 attention 在 GPU 上怎么跑         │
│     05_gpu/       (CUDA matmul + Triton flash-attention)      │
└───────────────────────────────────────────────────────────────┘
```

**每一层独立且可跑**：进入任意层的讲解页，按里面的命令就能运行。

## 一个请求的生命周期（概览）

```
  浏览器                  后端                 Agent 循环
  ───────                 ─────                ──────────
    │                       │                      │
    │  POST /chat           │                      │
    │ ─────────────────────▶│                      │
    │                       │  agent.run(query)    │
    │                       │ ────────────────────▶│
    │                       │                      │
    │                       │                      │  ┌──────────────┐
    │                       │                      │  │ LLM API call │──▶ L3 推理服务
    │                       │                      │  └──────────────┘        │
    │                       │                      │        ▲                 │ forward()
    │                       │                      │        │                 ▼
    │                       │                      │   next token       L4 Transformer
    │                       │                      │                         │
    │                       │                      │                         │ matmul()
    │  SSE: "正在创建..."   │                      │                         ▼
    │ ◀─────────────────────│                      │                     L5 GPU kernel
    │                       │                      │
    │                       │                      │  tool: write_file("index.html")
    │                       │                      │  tool: write_file("server.py")
    │                       │                      │  tool: run_shell("pip install flask")
    │                       │                      │
    │  SSE: done            │                      │
    │ ◀─────────────────────│                      │
```

## 快速开始

### 环境
```bash
git clone https://github.com/fxp/LLM-from-query-to-result
cd LLM-from-query-to-result
pip install -r requirements.txt
# L5 的 Triton 部分需要 CUDA GPU；没 GPU 可跳过。
```

### 端到端跑一遍（L1 + L2）
```bash
export ANTHROPIC_API_KEY=sk-...
cd 01_app && uvicorn backend.main:app --reload
# 浏览器打开 http://localhost:8000，输入 "帮我做一个 Todo 网站"
```

### 独立跑每一层
```bash
cd 02_agent && python agent.py "帮我做一个 Todo 网站"
cd 03_model && python server.py        # 另开一个终端跑 client.py
cd 04_transformer && python inference.py "Hello, I am"
cd 05_gpu && python benchmark.py
```

## 怎么读

**产品/应用开发者**：从 [L1](01-app.md)、[L2](02-agent.md) 开始，看到 "Agent 是怎么把一句话变成一堆 tool call 的" 就够用了。

**做 infra / 推理优化**：重点看 [L3](03-model.md)（batching、KV cache 的实际实现）和 [L5](05-gpu.md)（kernel 层做优化的地方）。

**想理解模型本身**：[L4](04-transformer.md) 是核心——300 行看懂 transformer。

**全都想懂**：按顺序读下来，[端到端 Trace](trace.md) 里有一条从 query 一路到 GPU 指令的完整 trace，可以作为串线索的地图。

## 目录

| 层 | 讲解 | 源码 | 运行依赖 |
|---|---|---|---|
| L1 App | [01-app](01-app.md) | [`01_app/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/01_app) | FastAPI |
| L2 Agent | [02-agent](02-agent.md) | [`02_agent/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/02_agent) | anthropic SDK |
| L3 Model 服务 | [03-model](03-model.md) | [`03_model/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/03_model) | transformers, torch |
| L4 Transformer | [04-transformer](04-transformer.md) | [`04_transformer/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/04_transformer) | torch |
| L5 GPU kernel | [05-gpu](05-gpu.md) | [`05_gpu/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/05_gpu) | nvcc, triton, CUDA GPU |

## 设计原则

- **每层代码 < 300 行**：超过就说明讲多了，砍掉。
- **不引入陌生抽象**：能用标准库就用标准库，不造框架。
- **"看得见"优先于"快"**：L3 的 batch 是 print 出来的，L4 的每层激活 shape 是打印的，L5 有 roofline benchmark——看得见才算讲清楚了。
- **一个贯穿例子**：所有层都用 "帮我做一个 Todo 网站"，避免读者 context-switch。
