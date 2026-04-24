# L3 · Model 层（推理服务）

**一句话**：收到一个 HTTP 请求里的 prompt，把它转成 token，喂进 transformer，把输出 token 一个个 stream 回去。

源码：[`03_model/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/03_model)

## 为什么要有这一层

L2 的 Agent 调模型时写的是：
```python
client.messages.stream(model=..., messages=[...])
```
看起来像调个普通函数，其实背后是一次 HTTP 请求打到推理服务。**为什么需要一个"服务"，不是直接 import 模型？** 三个原因：

1. **模型权重几 GB 到几百 GB**，加载一次要几十秒，不能每个请求加载一次。
2. **GPU 是共享资源**，多个请求要**排队 + batching**才能喂满 GPU。单请求跑 GPT-2 大概只用 10% 的算力，batch 32 个请求能打满。
3. **Tokenization + sampling + streaming** 都是服务端做的，客户端只关心文字。

所以 L3 的核心是三件事：
```
  HTTP 请求 ─▶ 排队 ─▶ batch 若干请求 ─▶ forward 一步（用 KV cache）
                                              │
                                              ▼
                                        采样一个 token
                                              │
                                              ▼
                                        stream 回客户端 ─▶ 下一步循环
```

## 这个最小实现做了什么

真正的推理服务（vLLM、TGI）做 **continuous batching、paged attention、speculative decoding**，代码量上万行。本 demo 只抓最核心的三件事：

| 做了 | 没做 |
|---|---|
| HTTP + SSE 接入 | 多 GPU / 分布式 |
| tokenize / detokenize | continuous batching（只做了 static batching）|
| 用 `past_key_values` 做 KV cache | paged attention |
| 温度采样 | beam search / speculative decoding |

足够让你看清"一个 HTTP 请求是怎么变成 GPU 上的一次 forward 的"。

## 目录

```
03_model/
├── server.py      # FastAPI server：/generate 接收 prompt，SSE 吐 token
├── client.py      # 示例客户端，对比"无 cache"和"有 cache"的速度
└── README.md
```

- [`server.py`](https://github.com/fxp/LLM-from-query-to-result/blob/main/03_model/server.py)
- [`client.py`](https://github.com/fxp/LLM-from-query-to-result/blob/main/03_model/client.py)

## 怎么跑

```bash
# 终端 A：启动服务（会下载 gpt2 small，约 500MB）
cd 03_model
python server.py
# Loaded gpt2 on cuda, 124M params. Listening on :9000

# 终端 B：发个请求
python client.py "Once upon a time"
```

你会看到 server 的日志打印每一步的 batch 大小、KV cache 大小、每个 token 的耗时——这些数字是本层的主角。

## 和其他层的接口

- **往上（L2）**：HTTP `POST /generate {prompt, max_tokens, temperature}`，返回 SSE token 流。本 demo 的协议比 OpenAI/Anthropic 的简单，但结构相同。
- **往下（L4）**：每步 forward 都是 `model(input_ids, past_key_values=...)`。`model` 是 HuggingFace 的 `GPT2LMHeadModel`——但**它内部就是 L4 手写版的那种 Transformer**，只是更全。

## 看得见的东西

运行 server 时会打印：
```
[batch 1] tokens=[15496, 2267, 287, 257, 640] (5 tokens)
[step 0] forward took 23.1 ms · kv_cache: none -> 5×12×64×768
[step 1] forward took 4.2 ms  · kv_cache: 5 -> 6   (sampled " ,")
[step 2] forward took 4.1 ms  · kv_cache: 6 -> 7   (sampled " there")
...
```

注意 step 0 vs step 1：**第一步慢 5 倍**，因为要处理整个 prompt（"prefill"）；之后每步只处理一个新 token（"decode"），用 KV cache 复用前面的计算。这是 LLM 推理最核心的一个优化，也解释了为什么"首 token 延迟"和"后续每 token 延迟"是两个独立指标。

---

[← L2 · Agent 层](02-agent.md) | 下一层 → [L4 · Transformer 层](04-transformer.md)
