# LLM-from-query-to-result

**从一行 query，到 GPU 上的一次浮点乘法——一本可以跑起来的全栈教科书。**

源码：[fxp/LLM-from-query-to-result](https://github.com/fxp/LLM-from-query-to-result)

当你在 ChatGPT 里敲一句话，按下回车，屏幕上一个字一个字蹦出回答——这中间到底发生了什么？从浏览器里的字符，到 GPU 上的一次浮点乘法，要穿过多少层？

这个站点把整条链路切成 8 层，**每一层都有独立的讲解和最小可运行代码**。你可以单独跑任意一层，也可以把它们串起来看完整 trace。**整条链路零外部 LLM API、零外部 model 权重**——L3 在莎士比亚语料上从随机初始化训出 7M 参数的 GPT，L4 SFT 后让它能听问答，L6 加载它，L7/L8 把 token 流式回浏览器。

> **GPU 实测验证 (RTX 5090, 2026-05)**：L3 训练 12 秒、L4 SFT on 124M 33.9 秒、L1 Triton flash-attn 比 PyTorch 快 8.5×。详见 [L1 GPU 层](05-gpu.md)、[L3 训练实测](00-train.md)。

## 贯穿全 repo 的例子

```
SFT 后 query:  "What is the capital of France?"
最终产物:       浏览器里流式涌出 " Paris.<|endoftext|>"
              这 3 个 token 全部由本仓自己训出来的 7M GPT 产出
```

这个 query 会穿过 8 层，我们会在每一层把它的"形态"打印出来：
**在 L7 它是一串 HTTP bytes**，**在 L8 它被包成 prompt 字符串发往 L6**，
**在 L6 它是一个 batch 里的 prompt + KV cache**，**在 L2 它是 tensor**，**在 L1 它是 GPU SM 上的指令**。
（往回追：训练阶段它在 **L3** 是训练数据里的 token 流和 loss，**L4** 是 (Q, A) 二元组 + masked cross-entropy，**L5** 是 ReAct trace 中的工具调用 token。）

## 八层架构（模型结构 → 训练 → 推理 → 应用）

```
┌───────────────────────────────────────────────────────────────┐
│  模型结构                                                     │
│ L1   GPU 基础层   矩阵乘 / flash-attention 在 GPU 上怎么跑    │
│      05_gpu/      (CUDA matmul + Triton flash-attention)      │
├───────────────────────────────────────────────────────────────┤
│ L2   Transformer  从零实现 GPT-2：embed / MHA / FFN / LN      │
│      04_transformer/  (PyTorch, ~330 行 + 手写 BPE ~230 行)   │
├───────────────────────────────────────────────────────────────┤
│  训练                                                         │
│ L3   预训练       prepare data → train loop → checkpoint      │
│      00_train/    (PyTorch + AdamW, ~140 行 ≈ 6 min CPU)      │
├───────────────────────────────────────────────────────────────┤
│ L4   指令 SFT     base ckpt + Q/A → instruction-tuned ckpt    │
│      00b_sft/     (~140 行 + 242 条手写 Q/A ≈ 28 sec)         │
├───────────────────────────────────────────────────────────────┤
│ L5   Agent SFT    instr ckpt + ReAct traces → tool-using ckpt │
│      00c_agent_sft/  (~150 行 + 258 条程序合成 ≈ 33 sec)      │
├───────────────────────────────────────────────────────────────┤
│  推理与应用                                                   │
│ L6   推理服务     tokenize / KV cache / SSE 流式输出          │
│      03_model/    (FastAPI + 我们的 GPT.step,零 transformers) │
├───────────────────────────────────────────────────────────────┤
│ L7   App / Web UI 用户看到的聊天界面 + 后端 SSE 流式输出      │
│      01_app/      (HTML + FastAPI)                            │
├───────────────────────────────────────────────────────────────┤
│ L8   Agent 循环   build prompt → 工具调度 → 注入 OBSERVATION  │
│      02_agent/    (HTTP 客户端,纯 urllib + ReAct loop)        │
└───────────────────────────────────────────────────────────────┘
```

**每一层独立且可跑**：进入任意层的讲解页，按里面的命令就能运行。

## 一个请求的生命周期（概览）

```
  浏览器              L7 后端           L8 客户端         L6 推理服务
  ───────             ─────             ─────────         ──────────
    │                   │                  │                  │
    │  POST /chat       │                  │                  │
    │ ─────────────────▶│                  │                  │
    │                   │  run_agent(q)    │                  │
    │                   │ ────────────────▶│                  │
    │                   │                  │ POST /generate   │
    │                   │                  │ ────────────────▶│
    │                   │                  │                  │ tokenize + forward
    │                   │                  │                  │   ↓ L2 (GPT-2)
    │                   │                  │                  │   ↓ L1 (matmul)
    │                   │                  │ data: {token}    │
    │                   │   token event    │ ◀────────────────│
    │  SSE: token       │ ◀────────────────│   ...            │
    │ ◀─────────────────│                  │ ◀────────────────│
    │   ...             │                  │                  │
    │  SSE: done        │                  │                  │
    │ ◀─────────────────│                  │                  │
```

## 快速开始

### 环境
```bash
git clone https://github.com/fxp/LLM-from-query-to-result
cd LLM-from-query-to-result
pip install -r requirements.txt
# L1 的 Triton/CUDA 部分需要 NVIDIA GPU；没 GPU 可跳过。
```

!!! tip "网络受限地区（如中国大陆）"
    1. `git clone` 走 mirror：`git clone https://gh-proxy.com/https://github.com/...`
    2. HF Hub 直连不通时 `GPT.from_pretrained("gpt2")` **自动 probe + fallback** 到 `https://hf-mirror.com`，并设 `HF_HUB_DISABLE_XET=1` 绕开慢的 Xet CDN。控制台会打印 `HF Hub direct unreachable; using mirror: ...`
    3. Tiny Shakespeare 已 bundle 在 `00_train/data/input.txt`，无需下载。
    4. BPE vocab 来自 `openaipublic.blob.core.windows.net`，CN 区一般可直连。

### 完全 from-scratch（推荐先做这一遍）

```bash
# Step 1: 训自己的 base model（M1 CPU ~6 min）
cd 00_train && python prepare.py && python train.py

# Step 2: SFT 让它能听懂问答（M1 CPU ~28 sec）
cd ../00b_sft && python train.py

# Step 3: 用 SFT'd ckpt 起 L6 服务
MODEL_PATH=$(pwd)/out/sft.pt python ../03_model/server.py

# Step 4: 另开终端，起 L7 web app + 浏览器
cd ../01_app && uvicorn backend.main:app --reload
# 浏览器打开 http://localhost:8000，问 "What is the capital of France?"
# → " Paris."
```

### 跳过 L3/L4，用 OpenAI 预训权重

```bash
# 终端 1：L6（不设 MODEL_PATH 就走 GPT.from_pretrained("gpt2")，首次下载 ~500MB）
cd 03_model && python server.py

# 终端 2：L7 web app
cd 01_app && uvicorn backend.main:app --reload
```

### 独立跑每一层
```bash
cd 00_train && python prepare.py && python train.py    # 训 base
cd 00b_sft  && python train.py                         # SFT
cd 02_agent && python agent.py "What is the capital of France?"
cd 03_model && python server.py
cd 04_transformer && python inference.py "Hello, I am"
cd 04_transformer && python bpe.py                     # 验证手写 BPE
cd 05_gpu   && python benchmark.py                     # 需要 NVIDIA GPU
```

## 配套博客

11 篇为什么 + 怎么做 + 多少数字的系列文章——[blog/ 目录](https://github.com/fxp/LLM-from-query-to-result/tree/main/blog)：序章 + L1（GPU）+ L2a（Transformer）+ L2b（BPE）+ L3（预训练）+ L4（指令 SFT）+ L5（Agent SFT）+ L6（推理服务）+ L7（Web UI）+ L8（Agent 循环）+ 端到端 trace 各一篇。每篇 5-10 分钟。

## 怎么读

**想看 model 是怎么"诞生"的**：[L3](00-train.md) — 6 分钟在 CPU 上把 loss 从 10.8 (random) 降到 ~4.5，看到 forward → loss → backward → optimizer 闭环。

**产品/应用开发者**：从 [L7](01-app.md)、[L8](02-agent.md) 开始，看清楚一条 chat 请求从浏览器一路下到推理服务的每个环节。

**做 infra / 推理优化**：重点看 [L6](03-model.md)（batching、KV cache 的实际实现）和 [L1](05-gpu.md)（kernel 层做优化的地方）。

**想理解模型本身**：[L2](04-transformer.md) 是核心——300 行看懂 transformer。L3 用的就是这个类。

**全都想懂**：按顺序读下来，[端到端 Trace](trace.md) 里有一条从 query 一路到 GPU 指令的完整 trace，可以作为串线索的地图。

## 目录

| 层 | 讲解 | 源码 | 运行依赖 |
|---|---|---|---|
| L1 GPU 基础层 | [05-gpu](05-gpu.md) | [`05_gpu/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/05_gpu) | nvcc, triton, CUDA GPU |
| L2 Transformer 架构 | [04-transformer](04-transformer.md) | [`04_transformer/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/04_transformer) | torch |
| L3 预训练 | [00-train](00-train.md) | [`00_train/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/00_train) | torch, regex |
| L4 指令 SFT | [00b-sft](00b-sft.md) | [`00b_sft/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/00b_sft) | torch |
| L5 Agent SFT | [00c-agent-sft](00c-agent-sft.md) | [`00c_agent_sft/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/00c_agent_sft) | torch |
| L6 推理服务 | [03-model](03-model.md) | [`03_model/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/03_model) | transformers, torch |
| L7 App / Web UI | [01-app](01-app.md) | [`01_app/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/01_app) | FastAPI |
| L8 Agent 循环 | [02-agent](02-agent.md) | [`02_agent/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/02_agent) | 标准库 (urllib) |

## 设计原则

- **每层代码 < 300 行**：超过就说明讲多了，砍掉。
- **不引入陌生抽象**：能用标准库就用标准库，不造框架。
- **零外部 LLM API、零外部 model 权重**：L3 在本地数据上训出 model，L6 加载它服务，所有 token 都来自本仓自己的 forward。代价是 model 智力有限，收益是整条链路完全可见、可改、可调试。
- **"看得见"优先于"快"**：L3 print loss 下降，L6 print KV cache 长度，L2 print 每层激活 shape，L1 有 roofline benchmark——看得见才算讲清楚了。
- **一个贯穿例子**：所有层用同一个莎士比亚 prompt（`ROMEO:`），避免读者 context-switch。
