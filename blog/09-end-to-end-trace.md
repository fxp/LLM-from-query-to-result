# 09 · 端到端 trace：从一句 query 到一次浮点乘法

> [← L5 GPU](08-L5-gpu.md) ｜ 配套源码：[`examples/trace.md`](https://github.com/fxp/LLM-from-query-to-result/blob/main/examples/trace.md)

到这里我们走完了 9 篇。每一层独立讲过——L0 训出 model、L0.5 SFT 让它能答问、L1-L5 把它服务出来。这一篇把所有层串起来：跟着用户在浏览器输入的那一句 query "What is the capital of France?"，看它穿过 9 层、最终在屏幕上变成 "Paris." 的全过程。

每一段都标了大约的 wall-clock 时间，从 t=0 开始。所有数字来自 RTX 5090 实测。

---

## t=0 ms · L1：一串键盘输入

用户在浏览器里敲完回车。`index.html` 的 JS 把这段字符串 POST 到本地后端：

```http
POST /chat HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{"query": "What is the capital of France?"}
```

后端 FastAPI 收到，调 `run_agent(query)`，建一个 SSE 响应——准备开始流式往回推 token。

**这一层的关键数据结构**：一个字符串。仅此而已。

---

## t=1 ms · L2：字符串变成 prompt

`agent.py` 的 `run_agent` 把 query 包成 base LM 能续写的形式（GPT-2 没有 user/assistant 概念，只能续写文本）：

```python
prompt = f"Q: {query}\nA:"
# = "Q: What is the capital of France?\nA:"
```

打成 HTTP 请求发给本地 L3：

```http
POST http://localhost:9000/generate HTTP/1.1
Content-Type: application/json

{"prompt": "Q: What is the capital of France?\nA:",
 "max_tokens": 64, "temperature": 0.0}
```

**这一层的关键**：`run_agent` 全部代码 ~80 行，只做 "build prompt → POST → relay SSE"。没有 tool use、没有对话历史、没有重试——故意最小，让 chat 客户端的本质露出来。

---

## t=5 ms · L3：字符串变成 token

`server.py` 收到 prompt，先 BPE tokenize（用我们 [L4b](07-L4-bpe.md) 那个手写的 BPE）：

```
"Q: What is the capital of France?\nA:"
  → [48, 25, 1867, 318, 262, 3139, 286, 4881, 30, 198, 32, 25]
  (12 tokens)
```

每个 token id 对应 vocab 表里的一个 piece：
- `48` = 'Q'
- `25` = ':'
- `1867` = ' What'（注意前导空格归词）
- `318` = ' is'
- `262` = ' the'
- `3139` = ' capital'
- `286` = ' of'
- `4881` = ' France'
- `30` = '?'
- `198` = '\n'
- `32` = 'A'
- `25` = ':'

然后进入 **prefill**——一次性把这 12 个 token 喂进 model：

```python
logits, kv_caches = model.step(input_ids)   # input_ids shape [1, 12]
# logits [1, 12, 50257], kv_caches: list of 12 tuples (K[1, 12, 12, 64], V[...])
```

**这一层的关键**：BPE encode + prefill。从这一刻起，文本变成了**张量**——往后都是浮点数运算。

---

## t=12 ms · L4：tensor 流过 12 层 transformer

进入 model.step，一次 forward pass 内部：

```
input_ids [1, 12]
   │
   ▼ wte + wpe
   x [1, 12, 768]
   │
   ▼ Block 0: ln_1 → c_attn(x) [1,12,2304] → split q/k/v → reshape [1,12 head,12,64]
   │          → scaled_dot_product_attention (内部调 cuBLAS + softmax + cuBLAS, 在 L5 里)
   │          → c_proj [1,12,768] → +residual
   │          → ln_2 → c_fc [1,12,3072] → GELU → c_proj [1,12,768] → +residual
   │  x [1, 12, 768]
   │
   ▼ Block 1, 2, ..., 11   (11 个相同的 block)
   │
   ▼ ln_f
   ▼ logits = x @ wte.T   →  [1, 12, 50257]
```

这里发生了 **12 层 × 6 matmul = 72 个矩阵乘**：
- 每个 block 的 c_attn (D → 3D)
- attention 的 Q@K.T 和 (softmax @ V)
- c_proj (D → D)
- mlp.c_fc (D → 4D), mlp.c_proj (4D → D)

加上 final lm_head 的 1 个 matmul = **73 个 matmul**。

最大的两个：
- `mlp.c_fc`: `[12, 768] × [768, 3072]` ≈ 0.06 GFLOP
- `lm_head`: `[12, 768] × [768, 50257]` ≈ 0.93 GFLOP

整次 prefill 约 **5 GFLOP**。

**实测 5090 上 prefill 1.79 ms**——意味着这 73 个 matmul + 12 个 LN + 12 个 GELU + 12 个 attention softmax + 各种 reshape，全部在 1.79 ms 内完成。

> 详细时序见 [04_transformer/inference.py "Hello, I am" 实测](https://github.com/fxp/LLM-from-query-to-result/blob/main/04_transformer/inference.py)：第一个 block 100+ ms（CUDA kernel JIT 编译开销），后续每个 block ~0.2-0.7 ms。

---

## t=12.06 ms（一个 matmul 之内）· L5：tensor 变成机器指令

挑最大的那个 matmul：`lm_head` 的 `[12, 768] × [768, 50257]`。

cuBLAS 接到这个调用（PyTorch 内部转发），launch 一个 CUDA kernel：

- **Grid**：C 形状 [12, 50257]，按 TILE_M × TILE_N = 128 × 128 切 → ⌈12/128⌉ × ⌈50257/128⌉ = 1 × 393 = **393 个 block**
- **每 block**：256 个 thread，每 32 个一 warp，分发到 SM 上跑
- **每个 thread**：算输出 tile 里的几个元素
- **K 维**（768）按 TILE_K=32 分块，每步：
  1. 256 个 thread 协作把 A 的 [128, 32] tile 和 B 的 [32, 128] tile 从 HBM 搬到 shared memory
  2. 每个 thread 在 shared memory 里做 32 次乘加，累加到 register
  3. `__syncthreads()`，下一个 K tile
- **结束**：把 register 里的累加结果写回 HBM

cuBLAS 还有更深的优化（Tensor Core、async copy、software pipelining），实际跑在 Tensor Core 上以 TF32 精度执行——0.93 GFLOP / 0.013 ms ≈ 70 TFLOPS。**这是 5090 在小 matmul 下的效率**，比 fp32 朴素 80 TFLOPS 峰值还高（因为是 TF32）。

> 如果用我们手写的 `matmul_tiled.cu`：每个 C 元素 HBM 读 2K = 1536 个 float = 6 KB，加上数据复用大概 5× 减少。但没用 Tensor Core，跑 fp32，~9 TFLOPS。**比 cuBLAS 慢 7×**——这就是 [L5](08-L5-gpu.md) 那篇展示的差距。

**这一层的关键**：到 L5 我们已经忘了 "France" 这个字了。剩下的全是浮点数乘加。

---

## t=14 ms · 回到 L3：第一个 token 出来

prefill 完，从 logits 最后位置（位置 11，对应 prompt 末尾的 `:`）取 argmax（temperature=0 greedy）：

```python
last = logits[0, -1]   # [50257]
next_id = int(last.argmax().item())   # 7843 = ' Paris'
```

是 7843，对应 BPE 表里的 ` Paris`。

L3 yield 一个 SSE 帧：

```
data: {"token": " Paris"}
```

---

## t=14.5 ms · L2 → L1 → 浏览器：第一个 token 显示

```
L3 SSE: data: {"token": " Paris"}\n\n
   │
   ▼ L2 (urllib 行迭代器)
yield {"type": "token", "v": " Paris"}
   │
   ▼ L1 (FastAPI generator → StreamingResponse)
yield "data: {\"type\": \"token\", \"v\": \" Paris\"}\n\n"
   │
   ▼ 浏览器 ReadableStream reader
buf 解析帧 → ev = {"type": "token", "v": " Paris"}
   │
   ▼ render(ev)
log.append(" Paris")
```

**屏幕上看到 "Paris"**。整条链路 14 ms 不到——比眨眼快。

---

## t=17 ms · L3：第二个 token，用 KV cache

把刚生成的 ` Paris` (id 7843) 喂回 model.step，**带上之前 prefill 的 KV cache**：

```python
next_id = torch.tensor([[7843]], device='cuda')
logits, kv_caches = model.step(next_id, kv_caches)
# now kv_caches[0][0].size(2) == 13 (was 12)
```

每个 block 的 attention：
- q 是 [1, 12_head, 1, 64]（只有 1 个新 query）
- k 是 [1, 12_head, 13, 64]（12 个 prompt + 1 个新）
- v 同 k

scaled_dot_product_attention 算完，q 对所有 13 个 k 做注意力，得到 [1, 12_head, 1, 64] 的输出。

**实测 decode 一步 2.63 ms**（5090）。整个 forward 内部相比 prefill 差一个数量级（`q @ k.T` 是 [1, h, 1, 13] 而不是 [1, h, 12, 12]，几乎可以忽略）。

argmax 出来：id 13 = `'.'`。yield 给 L2。屏幕上看到 "Paris."。

---

## t=20 ms · L3：第三个 token，EOT

再喂 `'.'` 回去，model 输出最高概率 token id 50256 = `<|endoftext|>`。

```
data: {"token": "<|endoftext|>"}
data: {"done": true}
```

L3 stream 结束。L2 yield `{"type": "done"}`。L1 yield `data: {"done": true}\n\n`。浏览器收到 `done` event，render `✓ done`。

---

## 时间总账

实测 RTX 5090（cold-start 之后，HF cache 已就位）：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| 网络 / SSE / Python 调度 | ~5 ms | 35% |
| BPE encode (12 tokens) | ~0.5 ms | 3% |
| **L4 prefill (124M, T=12)** | **1.79 ms** | **13%** |
| 生成第 1 个 token (sample + KV cache 回传) | ~2 ms | 14% |
| **L4 decode 第 2 个 token** | **2.63 ms** | **19%** |
| **L4 decode 第 3 个 token (EOT)** | **2.63 ms** | **19%** |
| 总计端到端 | **~14 ms** | 100% |

**14 ms 看到完整答案 "Paris."**。比眨眼快。

主要消耗：网络/调度 ~5 ms，三次 model forward 共 ~7 ms，剩 ~2 ms 是 sampling、tokenize、SSE 帧序列化等。

如果改成更大的模型（GPT-4 那种 1.7T 规模），单次 forward ~50 ms，"打字感"会回来。这也是为什么 ChatGPT 用户能看到字一个一个蹦——不是网络慢，是模型 forward 慢。

---

## 数字对比表（用我们 SFT'd 124M）

| 层 | "形态" | 量级 |
|---|---|---|
| L1 | HTTP body | 80 字节 in，几 KB out (SSE 帧) |
| L2 | HTTP request | 1 个 (POST /generate) |
| L3 | tokens | 12 prompt + 3 generated |
| L3 | forward 次数 | 1 prefill + 2 decode = 3 |
| L4 | matmul 次数 | (12 layer × 6 + 1 lm_head) × 3 = 219 个 matmul |
| L5 | 浮点运算 | ~5 GFLOP prefill + 2 × 5 GFLOP decode ≈ 15 GFLOP |
| L5 | HBM 流量（权重） | 单次 forward ~500 MB，总计 ~1.5 GB |
| 时间 | end-to-end | 14 ms |

---

## 把这条链路记进脑子

如果让我用一个图概括整本书最重要的图，是这个：

```
浏览器                L1                 L2                 L3                  L4                L5
  │                    │                  │                  │                   │                 │
  │ POST /chat         │                  │                  │                   │                 │
  │ ──────────────────▶│ run_agent(q)     │                  │                   │                 │
  │                    │ ────────────────▶│ POST /generate   │                   │                 │
  │                    │                  │ ────────────────▶│ tokenizer.encode │                 │
  │                    │                  │                  │ → 12 tokens       │                 │
  │                    │                  │                  │ model.step ─────▶ │ wte+wpe+12block │
  │                    │                  │                  │                   │ → cuBLAS matmul │
  │                    │                  │                  │                   │ ───────────────▶│
  │                    │                  │                  │                   │                 │ Tensor Core
  │                    │                  │                  │                   │                 │ TF32 ~70 TFLOPS
  │                    │                  │                  │                   │ ◀───────────────│
  │                    │                  │                  │ ◀──────────────── │
  │                    │                  │ data: " Paris"   │ argmax → " Paris" │                 │
  │ SSE: token         │ event {token}    │ ◀────────────────│                   │                 │
  │ ◀──────────────────│ ◀────────────────│                  │                   │                 │
```

**每一道箭头都是一个明确的接口、一段代码、一个性能数字**。这就是这本书想交付的形态感。

---

## 想继续学

整本书是个起点，不是终点。如果你想再深入：

- **训一个真正能用的 model**：[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)，复现 GPT-2 small（124M × 10B WebText token，~$200 GPU 时）。
- **看 production 推理服务怎么写**：[vllm](https://github.com/vllm-project/vllm) (continuous batching + PagedAttention) 或 [sglang](https://github.com/sgl-project/sglang)。
- **GPU 编程深入**：[CUTLASS](https://github.com/NVIDIA/cutlass)（NVIDIA 高性能 matmul/conv 模板库）和 [Triton Puzzles](https://github.com/srush/Triton-Puzzles)。
- **flash-attention 论文**：[Dao et al. 2022](https://arxiv.org/abs/2205.14135) + [FA2 paper](https://arxiv.org/abs/2307.08691)。
- **真正的 chat agent**：[OpenAI tool use](https://platform.openai.com/docs/guides/function-calling) 文档 + [Anthropic computer use](https://www.anthropic.com/news/3-5-models-and-computer-use)。

每一个方向都是一本书。这本书只是地图——告诉你这些方向都在哪里、它们怎么连起来。

---

[← 回到序章](00-overview.md) ｜ [回到目录](README.md)
