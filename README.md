# LLM-from-query-to-result

**从一行 query，到 GPU 上的一次浮点乘法——一本可以跑起来的全栈教科书。**

📖 **在线阅读**：<https://fxp.github.io/LLM-from-query-to-result/>

当你在 ChatGPT 里敲一句话，按下回车，屏幕上一个字一个字蹦出回答——这中间到底发生了什么？从浏览器里的字符，到 GPU 上的一次浮点乘法，要穿过多少层？

这个 repo 把整条链路切成 7 层，**每一层都有独立的讲解和最小可运行代码**。你可以单独跑任意一层，也可以把它们串起来看完整 trace。

**这是真正的 from-scratch**：
- ✅ 模型架构（L4，~330 行手写 GPT-2）
- ✅ 训练循环（L0，~140 行 AdamW + cosine schedule）
- ✅ 训练数据（L0，1.1 MB 莎士比亚纯文本）
- ✅ Tokenizer（L4 `bpe.py`，手写的 BPE，与 tiktoken 在中/日/emoji/标点上 bit-for-bit 等价）
- ✅ KV cache（L4 `GPT.step`，自己实现）
- ✅ 推理服务（L3，~140 行 FastAPI + SSE，**零 transformers runtime 依赖**）
- ✅ Instruct 微调（L0.5，~140 行 SFT 在自己 base model 上跑 28 秒，把 base 的"接龙莎翁"变成"答 'Paris.'"）
- ✅ Chat 客户端（L2，纯 urllib HTTP）
- ✅ Web UI（L1，纯 HTML + fetch streaming）
- ✅ GPU kernels（L5，手写 CUDA matmul + Triton flash-attention，[实测见 05_gpu/README.md](./05_gpu/README.md#实测样例)）

**唯一的"借"**：PyTorch 的 tensor / autograd（这是底座，不重写），可选下载 OpenAI 公开的 GPT-2 124M 权重（仅 fallback，默认走我们自训的 model）。

> **GPU 实测验证 (RTX 5090, 2026-05)**：
> - L0 训练 1000 步：12.2 秒（CPU 6 分钟，60×加速）
> - L0.5 SFT on 124M base：30 epochs / 33.9 秒，loss 1.6 → 0.0
> - 推理：prefill 1.8 ms / decode 2.6 ms/token (124M, batch=1, fp32)
> - L5 matmul 2048³：tiled 9.2 / cuBLAS 68.9 TFLOPS（cuBLAS 用 Tensor Core）
> - L5 attention：Triton flash-attn 比 unfused PyTorch **快 8.5×**

## 贯穿全 repo 的例子

```
SFT 后用户 query:  "What is the capital of France?"
最终产物:           浏览器里流式涌出 " Paris.<|endoftext|>"  （greedy 模式）
                  这 3 个 token 全部由本仓自己训出来的 7M GPT 产出
```

完整流转：
- **L0**：莎士比亚 1.1MB → 我们的 BPE → 0.34M tokens → 我们的 GPT (n_layer=4, n_embd=128) → AdamW 1000 步 → `ckpt.pt`
- **L0.5**：base ckpt + 63 条手写 Q/A → SFT 50 epochs (28 秒) → `sft.pt`（学会 instruction 格式 + 个别事实）
- **L3**：load `sft.pt` 用我们的 GPT.step() 跑推理（KV cache 自己实现）
- **L2**：把 query 包成 `Q: ...\nA:` POST 给 L3
- **L1**：浏览器 → FastAPI → SSE 流式返回每个 token

**整条链路里，PyTorch 是唯一非自家的依赖**（tensor 库 + autograd 是底座，不重写）。每个 token 经过的代码都在本 repo 里：算法、权重、tokenizer、KV cache、SSE 路由、UI 全部都是。

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

**每一层独立且可跑**：进入任意子目录，`cat README.md` 看讲解，按里面的命令就能运行。

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
pip install -r requirements.txt
# L5 的 Triton/CUDA 部分需要 NVIDIA GPU；没 GPU 可跳过。
```

> **网络受限地区（如中国大陆）的运行清单**：
>
> 1. **`git clone`** github.com 直连超时——用 mirror 前缀：
>    ```bash
>    git clone https://gh-proxy.com/https://github.com/fxp/LLM-from-query-to-result.git
>    ```
> 2. **`pip install`** 默认走 PyPI，CN 区一般 ok（或自己配 aliyun/tsinghua mirror）。
> 3. **HF Hub** (`GPT.from_pretrained("gpt2")`) 直连不通时**自动 probe + fallback** 到 `https://hf-mirror.com`，并设 `HF_HUB_DISABLE_XET=1` 绕开慢的 Xet CDN。控制台会打印：
>    ```
>    HF Hub direct unreachable; using mirror: https://hf-mirror.com
>    (set HF_HUB_DISABLE_XET=1, HF_HUB_DOWNLOAD_TIMEOUT=60)
>    ```
>    手动指定 endpoint：`export HF_ENDPOINT=...`（已设的会跳过 probe）。
> 4. **BPE vocab** (`encoder.json` + `vocab.bpe`) 来自 `openaipublic.blob.core.windows.net`——CN 区可直连。
> 5. **Tiny Shakespeare** 已 bundle 在 `00_train/data/input.txt`，**完全无需下载**。

### 完全 from-scratch（推荐先做这一遍）

```bash
# Step 1: 训自己的 base model（M1 CPU ~6 min，1000 步，loss 10.8 → ~4.5）
cd 00_train && python prepare.py && python train.py

# Step 2: SFT 让它能听懂问答（M1 CPU ~28 sec，50 epochs，loss 9 → 1.7）
cd ../00b_sft && python train.py

# Step 3: 用 SFT'd ckpt 起 L3 服务
MODEL_PATH=$(pwd)/out/sft.pt python ../03_model/server.py

# Step 4: 另开终端，起 L1 web app + 浏览器
cd ../01_app && uvicorn backend.main:app --reload
# 浏览器打开 http://localhost:8000，问 "What is the capital of France?"
# → " Paris."  ← 这 2 个 token 你自己端到端造出来的
```

### 跳过 L0/L0.5，用 OpenAI 预训权重

如果不想等训练，可以加载 OpenAI 公开的 GPT-2 124M 权重（首次会下载 ~500MB）：

```bash
# 终端 1：L3（不设 MODEL_PATH 就走 GPT.from_pretrained("gpt2")）
cd 03_model && python server.py

# 终端 2：L1 web app
cd 01_app && uvicorn backend.main:app --reload
```

### 独立跑每一层
```bash
cd 00_train  && python prepare.py && python train.py    # 训 base
cd 00_train  && python sample.py "ROMEO:"               # 采样 base
cd 00b_sft   && python train.py                         # SFT
# L2 / client.py 依赖 L3 先起 03_model/server.py
cd 02_agent  && python agent.py "What is the capital of France?"
cd 03_model  && python server.py
cd 04_transformer && python inference.py "Hello, I am"
cd 04_transformer && python bpe.py                     # 验证手写 BPE
cd 05_gpu    && python benchmark.py                    # 需要 NVIDIA GPU
```

## 配套博客

如果你不想读源码，先想看为什么这么设计、每一段代码背后的"为什么"——有一个 10 篇的配套系列：

📖 **[blog/](./blog) · 从一行 query 到 GPU 上的一次浮点乘法**

| # | 文章 | 主题 |
|---|---|---|
| 00 | [序章](blog/00-overview.md) | 全栈视角 + 设计原则 |
| 01 | [L0：从莎士比亚训出一个 GPT](blog/01-L0-training.md) | 预训练循环、loss 为什么从 10.8 降到 4.5 |
| 02 | [L0.5：24 秒变 instruct](blog/02-L0.5-sft.md) | SFT、loss masking、知识 vs 格式 |
| 03 | [L1：浏览器里那一个个蹦出来的字](blog/03-L1-app.md) | SSE + ~80 行前端 |
| 04 | [L2：Chat 客户端的最小本质](blog/04-L2-chat-client.md) | 删掉所有不必要的抽象 |
| 05 | [L3：自己写 KV cache 的推理服务](blog/05-L3-inference-server.md) | prefill vs decode、HF mirror fallback |
| 06 | [L4a：300 行手写 GPT-2](blog/06-L4-transformer.md) | embed / MHA / FFN / LN / KV cache |
| 07 | [L4b：手写 BPE，bit-for-bit ≡ tiktoken](blog/07-L4-bpe.md) | byte 映射、regex 预切词、merge 规则 |
| 08 | [L5：一次矩阵乘在 GPU 上到底怎么跑](blog/08-L5-gpu.md) | naive vs tiled vs cuBLAS、Triton flash-attn |
| 09 | [端到端 trace：从一句 query 到一次浮点乘法](blog/09-end-to-end-trace.md) | 9 层串起来 |

每篇 1500-3000 字，5-10 分钟。源码都开放，跑一遍看实测数字。

## 怎么读这个 repo

**想看 model 是怎么"诞生"的**：[L0](./00_train) — 6 分钟在 CPU 上把 loss 从 10.8 (random) 降到 ~4.5，看到 forward → loss → backward → optimizer 闭环。

**想看 base model 怎么变成 instruct model**：[L0.5](./00b_sft) — 28 秒 SFT 把 "续写莎翁" 的 base 变成 "答 'Paris.'" 的 instruct 模型。

**产品/应用开发者**：从 [L1](./01_app)、[L2](./02_agent) 开始，看清楚一条 chat 请求从浏览器一路下到推理服务的每个环节。

**做 infra / 推理优化**：重点看 [L3](./03_model)（KV cache 的实际实现，~140 行无外部依赖）和 [L5](./05_gpu)（kernel 层做优化的地方）。

**想理解模型本身**：[L4](./04_transformer) 是核心——~330 行看懂 transformer，加 ~150 行手写 BPE。L0 / L0.5 / L3 都用同一个 GPT 类。

**全都想懂**：按 L0 → L0.5 → L1..L5 顺序读下来，`examples/trace.md` 里有一条从 query 一路到 GPU 指令的完整 trace。

## 目录

| 目录 | 层 | 语言 | 运行依赖 |
|---|---|---|---|
| [`00_train/`](./00_train) | 训练 base | Python | torch, regex |
| [`00b_sft/`](./00b_sft) | SFT 微调 | Python + JSON | torch |
| [`01_app/`](./01_app) | Web App | Python + HTML/JS | FastAPI |
| [`02_agent/`](./02_agent) | Chat 客户端 | Python | 标准库 (urllib) |
| [`03_model/`](./03_model) | Model 服务 | Python | FastAPI, torch |
| [`04_transformer/`](./04_transformer) | Transformer + BPE | Python | torch, regex |
| [`05_gpu/`](./05_gpu) | GPU kernel | CUDA / Triton | nvcc, triton, CUDA GPU |
| [`examples/`](./examples) | end-to-end trace | Markdown | — |

## 设计原则

- **每层核心代码 < 300 行**：超过就说明讲多了，砍掉。
- **不引入陌生抽象**：能用标准库就用标准库，不造框架。第三方依赖只剩 PyTorch（不重写）+ FastAPI（写 web server 没必要重新造）+ regex（BPE 需要 unicode pattern）。
- **零外部 LLM API、零外部 LLM 库 runtime**：L0 训 base、L0.5 SFT、L3 推理（含 KV cache）、L4 BPE 全部本地代码。L4 的 `from_pretrained("gpt2")` 是唯一可选用 transformers 的地方（仅下载 OpenAI 权重时），跳过它全栈零 transformers/tiktoken 依赖。
- **"看得见"优先于"快"**：L0 print loss 下降，L0.5 print SFT 前后对比，L3 print KV cache 长度，L4 print 每层激活 shape，L4 BPE 自测对齐 tiktoken，L5 有 roofline benchmark——看得见才算讲清楚了。
- **一个贯穿例子**：base 模型用 `ROMEO:` 续写莎翁，SFT 后用 `What is the capital of France?` 测 instruction following。

## License

MIT
