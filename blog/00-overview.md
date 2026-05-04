# 00 · 序章：为什么写这个项目

> 当你在 ChatGPT 里敲一句话，按下回车，屏幕上一个字一个字蹦出回答——这中间到底发生了什么？从浏览器里的一个字符，到 GPU 上的一次浮点乘法，要穿过多少层？

我对这个问题有过非常糟糕的回答能力。能说"嗯，HTTP 请求到后端，后端调一个推理服务，推理服务里跑 transformer，transformer 里有 attention"——但每一个名词背后到底是什么、写出来什么样、跑起来多少 ms，都模糊。

更糟的是：我读过 transformer 论文、用过 OpenAI API、看过 vllm 的 README——这些只是更多名词。**抽象太多，具体太少**。

所以我写了这个 repo：把整条链路切成 8 层，每一层 < 300 行代码、能独立跑、有实测数字，而且——**整条链路不调任何外部 LLM API、不用任何外部 model 权重**。从一个空白随机权重的网络开始，在莎士比亚剧本上训出一个 GPT，然后让它通过自己写的推理服务、自己写的 chat client、自己写的 web app，最终在浏览器里一个字一个字蹦出 "Paris."。

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
│      03_model/    (FastAPI + 我们的 GPT.step，零 transformers)│
├───────────────────────────────────────────────────────────────┤
│ L7   App / Web UI 用户看到的聊天界面 + 后端 SSE 流式输出      │
│      01_app/      (HTML + FastAPI)                            │
├───────────────────────────────────────────────────────────────┤
│ L8   Agent 循环   build prompt → 工具调度 → 注入 OBSERVATION  │
│      02_agent/    (HTTP 客户端，纯 urllib + ReAct loop)       │
└───────────────────────────────────────────────────────────────┘
```

每一层独立可跑：进任意子目录，按 README 的命令就能运行——独立验证你对那一层的理解。

## 一个 token 的完整旅程

贯穿全 repo 的例子是一句问答：

```
问：What is the capital of France?
答：Paris.
```

这两个 token（` Paris` + `.` + EOT，共 3 个）的产生过程，**每一步都在自家代码里**：

1. 浏览器里的字符 → POST `/chat` → L7 FastAPI 收到 (~5 ms)
2. L7 调 `run_agent(query)` → L8 把 query 包成 `Q: ...\nA:` → POST `/generate` 给 L6 (~1 ms)
3. L6 用我们手写的 BPE 把 prompt 切成 12 个 token (`[48, 25, 1867, ...]`) (~5 ms)
4. L6 调 `model.step(input_ids)` 做 prefill：12 层 × 6 个 matmul 算出每个位置的 KV，缓存住 (~2 ms on 5090)
5. 从最后位置的 logits 采样 → 得到 token `' Paris'` → SSE 推回 L8 → L8 转 L7 → L7 SSE 推到浏览器 (~1 ms)
6. L6 用上一轮缓存好的 KV，把这个新 token 喂进去做 1 步 decode：又一个 matmul → 得到 `'.'` (~2 ms)
7. 又一步 decode → 得到 `<|endoftext|>` → 流结束

所有 matmul 都跑在 PyTorch + CUDA。如果你嫌不够"自己"，L1 给了手写的 CUDA tiled matmul 和 Triton flash-attention——结果一致，只是慢 8-10× 因为没用 Tensor Core。

## 这个项目跟外面那些"从零写 GPT"教程的区别

业界已经有很多优秀的"从零写 GPT"教程：[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)、[karpathy/minbpe](https://github.com/karpathy/minbpe)、[lucidrains 的几十个 transformer 实现](https://github.com/lucidrains)。这些都很好。

但它们都聚焦在**模型本身**——预训练循环、attention 实现、tokenizer 算法。一旦模型训出来，怎么 serve、怎么让浏览器看到、KV cache 怎么写、为什么 OpenAI API 收 0.0005 美元一次调用——这些"应用侧"的东西通常是另一个完全不同的世界。

这个项目把**整条链路**串起来：

| 主题 | 大多数教程 | 本项目 |
|---|---|---|
| GPT 架构 | ✅ | ✅ |
| 预训练循环 | ✅ | ✅ |
| BPE tokenizer | 一些（karpathy/minbpe） | ✅（且与 tiktoken bit-for-bit 等价） |
| **SFT instruction tuning** | 很少 | ✅ |
| **推理服务（KV cache + 流式）** | 很少 | ✅ |
| **Chat 客户端 / web 前端** | 几乎没有 | ✅ |
| **GPU kernel 内幕** | 一些（CUDA mode） | ✅ |
| **从浏览器一直到 matmul 的 trace** | 几乎没有 | ✅ |

代价是**每一层都很小**——5090 上 12 秒训完一个 7M 参数的 model，它当然不会答你的问题。但教学完整链路本身的意义，比训出一个像样模型更大。

## "完全 from scratch" 到底有多 from scratch

诚实清单：

| 部件 | 自家代码？ |
|---|---|
| Web 前端 (HTML + fetch streaming) | ✅ |
| Web 后端 (FastAPI + SSE) | ✅ |
| Chat 客户端 (urllib) | ✅ |
| 推理服务 (FastAPI + 自家 GPT.step) | ✅ |
| Tokenizer (BPE) | ✅（230 行手写，验证 bit-for-bit ≡ tiktoken） |
| Model 架构 (GPT-2) | ✅（330 行手写） |
| Model 权重 | ✅（L3 训练得到） |
| Instruct tuning | ✅（L4 在我们 base 上 SFT） |
| 训练数据 | ✅（1.1 MB Tiny Shakespeare 公共领域，bundle 在 repo 里） |
| 训练循环 | ✅（AdamW + cosine + grad clip） |
| GPU kernel | ✅（CUDA matmul + Triton flash-attention，可选） |

**借的部分**：

- **PyTorch**（tensor 运算 + autograd）——这是底座，重写它不是这本书的目标。L1 给了一些 CUDA / Triton kernel 让你看到"如果不用 PyTorch，自己怎么写"。
- **FastAPI / Starlette**（web server）——同理。
- **regex** 库（我们的 BPE 用 `\p{L}` `\p{N}` unicode pattern；Python 标准库 `re` 不支持）。
- **transformers** 库（**仅** L2 `from_pretrained()` 下载 OpenAI gpt2-124M 权重时用，runtime 完全不用）。

跳过 OpenAI 权重的话（只跑 L3 训出来的 7M）整栈零 transformers/tiktoken 依赖。

## 这本书想让你学到什么

1. **抽象塌缩到具体**。读完 L2，你不会再说"transformer 里面有 attention"——你会说"`c_attn` 是个 [D, 3D] 的 Linear，把 x 投影成 q/k/v 三块，然后 reshape 成 `[B, n_head, T, head_dim]`，然后做 `scaled_dot_product_attention`，这个底层在 GPU 上是 cuBLAS 的两个 matmul 加一个 softmax kernel"。
2. **每个数字都看得见**。loss 从 10.815（= ln 50257，理论值）降到 4.5，前向 prefill 多少 ms、decode 多少 ms、`tiled / naive` 1.3×、`Triton fused / unfused` 10×——你不再凭感觉，凭数字。
3. **完整链路的形态感**。当下次有人说"我们的推理服务慢，要做 KV cache"——你能立刻在脑子里算：12 层 × 8K context × 768 dim × 2 (K and V) × fp16 = ~37.7 MB per sample；1 个用户 batch 1 占 37 MB GPU memory；100 个用户并发要 3.7 GB，48 GB 卡能扛 ~1300 用户。这是**形态感**——抽象坍缩出来的东西。

## 怎么读

按数字 00 → 09 顺序读。每篇 1500-3000 字，~5-10 分钟。

或者按需跳：

- 想看 model 怎么诞生：[01-L0-training](01-L0-training.md) → [06-L4-transformer](06-L4-transformer.md) → [02-L0.5-sft](02-L0.5-sft.md)
- 想做产品/全栈：[03-L1-app](03-L1-app.md) → [04-L2-chat-client](04-L2-chat-client.md) → [05-L3-inference-server](05-L3-inference-server.md)
- 想做 infra / 推理优化：[05-L3-inference-server](05-L3-inference-server.md) → [08-L5-gpu](08-L5-gpu.md)

每篇末尾都有"下一篇"指引和"完整代码在哪"的链接。

下一站：[L3 — 从莎士比亚训出一个 GPT →](01-L0-training.md)
