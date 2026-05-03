# Blog 系列：从一行 query 到 GPU 上的一次浮点乘法

这是 [LLM-from-query-to-result](https://github.com/fxp/LLM-from-query-to-result) 项目的配套博客系列。每一篇对应 repo 里的一层，把"为什么这么做、怎么做、实测多少数字"讲清楚。

读完整个系列，你会从浏览器里那一个个蹦出来的字开始，一路追到 GPU 里的一次矩阵乘——而中间每一行代码都看得见。

## 目录

| # | 文章 | 对应代码 | 主题 |
|---|---|---|---|
| 00 | [序章：为什么写这个项目](00-overview.md) | — | 全栈视角 + 设计原则 |
| 01 | [L0：从莎士比亚训出一个 GPT](01-L0-training.md) | `00_train/` | 预训练循环、loss 为什么从 10.8 降到 4.5 |
| 02 | [L0.5：24 秒把"接龙莎翁"变成"答 Paris"](02-L0.5-sft.md) | `00b_sft/` | SFT、loss masking、为什么需要 124M base 才能泛化 |
| 03 | [L1：浏览器里那一个个蹦出来的字](03-L1-app.md) | `01_app/` | SSE、async generator、~80 行前端 |
| 04 | [L2：Chat 客户端的最小本质](04-L2-chat-client.md) | `02_agent/` | build prompt → POST → relay，删掉所有不必要的抽象 |
| 05 | [L3：自己写 KV cache 的推理服务](05-L3-inference-server.md) | `03_model/` | 不靠 transformers runtime、prefill vs decode、HF mirror 自动 fallback |
| 06 | [L4a：300 行 GPT-2](06-L4-transformer.md) | `04_transformer/model.py` | embed / MHA / FFN / LN / weight init / KV cache |
| 07 | [L4b：手写 BPE，bit-for-bit 等价 tiktoken](07-L4-bpe.md) | `04_transformer/bpe.py` | byte 映射、regex 预切词、merge 规则 |
| 08 | [L5：一次矩阵乘在 GPU 上到底怎么跑](08-L5-gpu.md) | `05_gpu/` | naive vs tiled vs cuBLAS、Triton flash-attn |
| 09 | [端到端 trace：从一句 query 到一次浮点乘法](09-end-to-end-trace.md) | — | 把 9 层串起来，一个 token 的完整旅程 |

## 适合谁读

- **想搞清楚 ChatGPT 怎么"运转"的工程师**：这本书把抽象塌缩成具体——每一层 < 300 行代码，可改可调试。
- **想入门 LLM 训练的人**：L0 + L0.5 是完整的 pretrain + SFT 流程，CPU 几分钟、GPU 几秒能跑完。
- **想理解推理性能的人**：L3 + L5 把 KV cache、prefill/decode 拆成 prefill/decode、Tensor Core、flash-attention 都讲了。
- **想写自己的 LLM 服务的人**：L1/L2/L3 给了一个最小的、能跑的 chat 服务（前端 + SSE + 推理引擎），不到 300 行。

## 推荐阅读顺序

- **第一次**：从 [00-overview](00-overview.md) 开始，按数字读下来。
- **想看 model 怎么诞生**：跳到 [01-L0-training](01-L0-training.md) → [02-L0.5-sft](02-L0.5-sft.md) → [06-L4-transformer](06-L4-transformer.md)。
- **做产品/全栈**：[03-L1-app](03-L1-app.md) → [04-L2-chat-client](04-L2-chat-client.md) → [05-L3-inference-server](05-L3-inference-server.md)。
- **做 infra / 推理优化**：[05-L3-inference-server](05-L3-inference-server.md) + [08-L5-gpu](08-L5-gpu.md)。

每一篇都自包含——可以单独读。
