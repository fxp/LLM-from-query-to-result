# LLM-from-query-to-result

**从一行 query，到 GPU 上的一次浮点乘法——一本可以跑起来的全栈教科书。**

源码：[fxp/LLM-from-query-to-result](https://github.com/fxp/LLM-from-query-to-result)

当你在 ChatGPT 里敲一句话，按下回车，屏幕上一个字一个字蹦出回答——这中间到底发生了什么？从浏览器里的字符，到 GPU 上的一次浮点乘法，要穿过多少层？

这个站点把整条链路切成 7 层，**每一层都有独立的讲解和最小可运行代码**。你可以单独跑任意一层，也可以把它们串起来看完整 trace。**整条链路零外部 LLM API、零外部 model 权重**——L0 在莎士比亚语料上从随机初始化训出 7M 参数的 GPT，L0.5 SFT 后让它能听问答，L3 加载它，L1/L2 把 token 流式回浏览器。

> **GPU 实测验证 (RTX 5090, 2026-05)**：L0 训练 12 秒、L0.5 SFT on 124M 33.9 秒、L5 Triton flash-attn 比 PyTorch 快 8.5×。详见 [L5 实测样例](05-gpu.md#实测样例)、[L0.5 RTX 5090 实测](00-train.md)。

## 贯穿全 repo 的例子

```
SFT 后 query:  "What is the capital of France?"
最终产物:       浏览器里流式涌出 " Paris.<|endoftext|>"
              这 3 个 token 全部由本仓自己训出来的 7M GPT 产出
```

这个 query 会穿过 7 层，我们会在每一层把它的"形态"打印出来：
**在 L0 它是训练数据里的 token 流和 loss**，
**在 L0.5 它是 (Q, A) 二元组 + masked cross-entropy**，
在 L1 它是一串 HTTP bytes，在 L2 它被包成 prompt 字符串发往 L3，
在 L3 它是一个 batch 里的 prompt + KV cache，在 L4 它是 tensor，在 L5 它是 GPU SM 上的指令。

## 七层架构

```
┌───────────────────────────────────────────────────────────────┐
│ L0   训练层       prepare data → train loop → checkpoint      │
│      00_train/    (PyTorch + AdamW, ~140 行 ≈ 6 min CPU)      │
├───────────────────────────────────────────────────────────────┤
│ L0.5 SFT 层       base ckpt + Q/A → instruction-tuned ckpt    │
│      00b_sft/     (~140 行 + 60 条手写数据 ≈ 28 sec CPU)      │
├───────────────────────────────────────────────────────────────┤
│ L1   App 层       用户看到的聊天界面 + 后端 SSE 流式输出     │
│      01_app/      (HTML + FastAPI)                            │
├───────────────────────────────────────────────────────────────┤
│ L2   Chat 客户端  build prompt → POST /generate → relay 流    │
│      02_agent/    (HTTP 客户端,纯 urllib)                     │
├───────────────────────────────────────────────────────────────┤
│ L3   Model 层     推理服务：tokenize / KV cache / SSE         │
│      03_model/    (FastAPI + 我们的 GPT.step,零 transformers) │
├───────────────────────────────────────────────────────────────┤
│ L4   Transformer  从零实现 GPT-2：embed / MHA / FFN / LN     │
│      04_transformer/  (PyTorch, ~330 行 + 手写 BPE ~150 行)   │
├───────────────────────────────────────────────────────────────┤
│ L5   GPU 层       矩阵乘和 attention 在 GPU 上怎么跑         │
│      05_gpu/      (CUDA matmul + Triton flash-attention)      │
└───────────────────────────────────────────────────────────────┘
```

**每一层独立且可跑**：进入任意层的讲解页，按里面的命令就能运行。

## 一个请求的生命周期（概览）

```
  浏览器              L1 后端           L2 客户端         L3 推理服务
  ───────             ─────             ─────────         ──────────
    │                   │                  │                  │
    │  POST /chat       │                  │                  │
    │ ─────────────────▶│                  │                  │
    │                   │  run_agent(q)    │                  │
    │                   │ ────────────────▶│                  │
    │                   │                  │ POST /generate   │
    │                   │                  │ ────────────────▶│
    │                   │                  │                  │ tokenize + forward
    │                   │                  │                  │   ↓ L4 (GPT-2)
    │                   │                  │                  │   ↓ L5 (matmul)
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
# L5 的 Triton/CUDA 部分需要 NVIDIA GPU；没 GPU 可跳过。
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

# Step 3: 用 SFT'd ckpt 起 L3 服务
MODEL_PATH=$(pwd)/out/sft.pt python ../03_model/server.py

# Step 4: 另开终端，起 L1 web app + 浏览器
cd ../01_app && uvicorn backend.main:app --reload
# 浏览器打开 http://localhost:8000，问 "What is the capital of France?"
# → " Paris."
```

### 跳过 L0/L0.5，用 OpenAI 预训权重

```bash
# 终端 1：L3（不设 MODEL_PATH 就走 GPT.from_pretrained("gpt2")，首次下载 ~500MB）
cd 03_model && python server.py

# 终端 2：L1 web app
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

10 篇为什么 + 怎么做 + 多少数字的系列文章——[blog/ 目录](https://github.com/fxp/LLM-from-query-to-result/tree/main/blog)：序章 + L0/L0.5/L1/L2/L3/L4a/L4b/L5 各一篇 + 端到端 trace 一篇。每篇 5-10 分钟。

## 怎么读

**想看 model 是怎么"诞生"的**：[L0](00-train.md) — 6 分钟在 CPU 上把 loss 从 10.8 (random) 降到 ~4.5，看到 forward → loss → backward → optimizer 闭环。

**产品/应用开发者**：从 [L1](01-app.md)、[L2](02-agent.md) 开始，看清楚一条 chat 请求从浏览器一路下到推理服务的每个环节。

**做 infra / 推理优化**：重点看 [L3](03-model.md)（batching、KV cache 的实际实现）和 [L5](05-gpu.md)（kernel 层做优化的地方）。

**想理解模型本身**：[L4](04-transformer.md) 是核心——300 行看懂 transformer。L0 用的就是这个类。

**全都想懂**：按顺序读下来，[端到端 Trace](trace.md) 里有一条从 query 一路到 GPU 指令的完整 trace，可以作为串线索的地图。

## 目录

| 层 | 讲解 | 源码 | 运行依赖 |
|---|---|---|---|
| L0 训练 | [00-train](00-train.md) | [`00_train/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/00_train) | torch, regex |
| L1 App | [01-app](01-app.md) | [`01_app/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/01_app) | FastAPI |
| L2 Chat 客户端 | [02-agent](02-agent.md) | [`02_agent/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/02_agent) | 标准库 (urllib) |
| L3 Model 服务 | [03-model](03-model.md) | [`03_model/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/03_model) | transformers, torch |
| L4 Transformer | [04-transformer](04-transformer.md) | [`04_transformer/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/04_transformer) | torch |
| L5 GPU kernel | [05-gpu](05-gpu.md) | [`05_gpu/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/05_gpu) | nvcc, triton, CUDA GPU |

## 设计原则

- **每层代码 < 300 行**：超过就说明讲多了，砍掉。
- **不引入陌生抽象**：能用标准库就用标准库，不造框架。
- **零外部 LLM API、零外部 model 权重**：L0 在本地数据上训出 model，L3 加载它服务，所有 token 都来自本仓自己的 forward。代价是 model 智力有限，收益是整条链路完全可见、可改、可调试。
- **"看得见"优先于"快"**：L0 print loss 下降，L3 print KV cache 长度，L4 print 每层激活 shape，L5 有 roofline benchmark——看得见才算讲清楚了。
- **一个贯穿例子**：所有层用同一个莎士比亚 prompt（`ROMEO:`），避免读者 context-switch。
