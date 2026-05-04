# 一个 query 的完整旅程

目标：用户敲入 `"What is the capital of France?"`，浏览器里看到本地 GPT-2 流式涌出的回答。
下面按时间顺序，看这个 query 在五层里每一层长什么样。**整条链路本地、零外部 API**——所有数字都来自 `03_model/server.py` + `04_transformer/inference.py` 在一台 M1 Mac CPU 上的实测。

---

## t=0 ms · L7：一串键盘输入

用户在浏览器里敲完回车。`index.html` 里的 JS 把这段字符串 POST 到本地后端：

```http
POST /chat HTTP/1.1
Content-Type: application/json

{"query": "What is the capital of France?"}
```

后端 `main.py` 调 `run_agent(query)`，建一个 SSE 响应，准备流式往回推事件。

**这一层的关键数据结构**：一个字符串。仅此而已。

---

## t=1 ms · L8：字符串变成 prompt

`agent.py` 的 `run_agent` 把 query 包成 base LM 能续写的形式（GPT-2 没有 `user`/`assistant` 概念，只能续写文本）：

```python
prompt = f"Question: {query}\nAnswer:"
# = "Question: What is the capital of France?\nAnswer:"
```

打成一个 HTTP 请求发给本地 L6：

```http
POST http://localhost:9000/generate HTTP/1.1
Content-Type: application/json

{"prompt": "Question: What is the capital of France?\nAnswer:",
 "max_tokens": 64, "temperature": 0.8}
```

**这一层的关键**：`run_agent` 全部代码 ~90 行，只做"build prompt → POST → relay SSE"。没有 tool use、没有对话历史、没有重试——故意最小，让"streaming chat client 的本质"露出来。要换 Qwen-2.5-Coder 跑真正的 agent loop 是另一个练习。

---

## t=10 ms · L6：字符串变成 token

`server.py` 拿到 prompt，先 tokenize（GPT-2 用的是 BPE）：

```
"Question: What is the capital of France?\nAnswer:"
  →  [24361, 25, 1867, 318, 262, 3139, 286, 4881, 30, 198, 33706, 25]
  (12 tokens)
```

然后进入 **prefill**：一次性把这 12 个 token 喂进模型，算出所有位置的 KV，缓存起来。

```
[prefill] B=1, T=12 tokens
  forward 一次：shape=[1, 12, 768]
  耗时（Mac M1 CPU, GPT-2 124M）：~1300 ms
  产生 past_key_values：12 层 × 2 × [1, 12 heads, 12, 64]
```

接下来 **decode** 循环：每次把上一轮采样出的 1 个新 token 喂进模型，用 KV cache 算下一个。L6 server log 实测：

```
[prompt] 'Question: What is the capital of France?\nAnswer:' -> tokens [24361, 25, 1867, 318, 262, 3139, 286, 4881, 30, 198, 33706, 25]
[step  0] prefill   16.8 ms  kv_len=12   -> ' France'
[step  1] decode     9.4 ms  kv_len=13   -> ' is'
[step  2] decode    10.2 ms  kv_len=14   -> ' the'
[step  3] decode     9.3 ms  kv_len=15   -> ' capital'
...
```

每个 decode step ~10 ms（Mac M1 CPU，warm）。冷启动第一次 prefill 会慢 50 倍（~1.3 s），是 PyTorch 第一次 dispatch 的开销，往后就稳定下来了。**这些 token 通过 SSE 一个个流回 L8，L8 转发回 L7，L7 转发回浏览器**。

**这一层的关键**：prefill 一次贵（算全部 12 个位置），decode 每次便宜（只算 1 个位置），但要跑几十步。"首 token 延迟" ≈ prefill，"后续 token 速度" ≈ decode。

> 顺便观察一下 GPT-2 124M 的回答：`The capital is the capital of the French republic of prussia...` 它看不懂自己在说什么——这是 2019 年的 base model，没经过 RLHF，重复 + 跑题是它的常态。本 repo 的承诺不是"答得好"，是"每个 token 你都能追到 matmul"。

---

## t=11 ms（prefill 中）· L2：token 变成 tensor

**每一次 forward 内部**发生的事（GPT-2 small：12 层、12 头、d=768）：

```
输入：input_ids  [B=1, T=12]
  │
  ▼  wte + wpe
  x  [1, 12, 768]
  │
  ▼  Block 0
  │  ┌── LN ──▶ [1,12,768]
  │  │
  │  ├── attn.c_attn(x)  → qkv [1, 12, 2304]   ← matmul ([1,12,768] @ [768,2304])
  │  ├── split to q, k, v [1, 12 heads, 12, 64]
  │  ├── scaled_dot_product_attention(q, k, v, is_causal=True)
  │  │     │
  │  │     ▼  ──── attention 内部，最终交给 L1 ────
  │  │     Q @ K.T  → [1, 12, 12, 12]       ← matmul
  │  │     softmax  → [1, 12, 12, 12]
  │  │     @ V      → [1, 12, 12, 64]       ← matmul
  │  │     reshape  → [1, 12, 768]
  │  │
  │  ├── attn.c_proj  → [1, 12, 768]        ← matmul
  │  ├── + residual
  │  └── LN, MLP (c_fc 768→3072, c_proj 3072→768) + GELU + residual   ← 2 matmul
  │
  ▼  再过 11 个 block
  │
  ▼  ln_f
  │
  ▼  lm_head = x @ wte.T → logits [1, 12, 50257]   ← matmul
```

**一次 prefill forward 里：12 层 × 6 matmul/层 + 1 lm_head ≈ 73 个 matmul**。最大的两个：
- MLP `c_fc`：`[12, 768] @ [768, 3072]` ≈ 0.06 GFLOP
- lm_head：`[12, 768] @ [768, 50257]` ≈ 0.93 GFLOP

整次 prefill 大约 5 GFLOP——M1 CPU 单核 ~50 GFLOPS 浮点峰值，理论 0.1 秒，实测 1.3 秒（PyTorch 调度 + 没向量化好），符合预期。

**这一层的关键**：到这里我们已经忘了 "France" 这个字了。剩下的全是浮点数乘加。

跑一次 `cd 04_transformer && python inference.py "Hello, I am"` 就能看到逐层 shape：

```
loaded gpt2: 124.44M params  (device=cpu)
tokens: [15496, 11, 314, 716]  (4 tokens)

forward pass, one step:
  x = embed(ids) + pos(ids)          shape=(1, 4, 768)
  block  0 attn+ffn                   shape=(1, 4, 768)  32.3 ms
  block  1 attn+ffn                   shape=(1, 4, 768)  10.8 ms
  ...
  block 11 attn+ffn                   shape=(1, 4, 768)  21.5 ms
  ln_f                                shape=(1, 4, 768)
  logits = x @ wte.T                 shape=(1, 4, 50257)

argmax at last pos -> 257  (' a')
```

---

## t=12 到 12.06 ms（一个 matmul 之内）· L1：tensor 变成机器指令

挑最大的 matmul：lm_head `[12, 768] @ [768, 50257]`。

在 NVIDIA GPU 上 cuBLAS 会把它 launch 成一个 CUDA kernel，大致是这样：

- **Grid**：C 的形状是 `[12, 50257]`，按 `TILE_M × TILE_N = 128 × 128` 切，得到 `⌈12/128⌉ × ⌈50257/128⌉ = 1 × 393 = 393` 个 block。
- **Block 内**：256 个线程，每个线程负责算输出 tile 里的几个元素。
- **工作流**：K 维（=768）按 `TILE_K=32` 分块，每一步：
  1. 所有 256 线程协作，把 A 的一个 `[128, 32]` tile 和 B 的一个 `[32, 128]` tile 从 HBM 搬到 shared memory；
  2. 每个线程在 shared memory 里做 32 次乘加，累加到自己的寄存器里；
  3. `__syncthreads()`，下一个 K tile。
- **结束**：把累加结果写回 HBM。

整个 kernel 的 HBM 读入量 ≈ `|A| + |B| = 12·768·4 + 768·50257·4 ≈ 155 MB`；算力 ≈ `2 · 12 · 768 · 50257 ≈ 0.93 GFLOP`。在 A100 上 < 0.1 ms。

**如果用我们的 `matmul_naive.cu`**，每个 C 元素都要从 HBM 读 K=768 个 A 元素、768 个 B 元素，没有 tile 复用——HBM 流量翻倍以上。这就是为什么要分块。

**Flash-attention 的省更夸张**（`05_gpu/attention_triton.py`）：把 Q@K.T、softmax、@V 三个 kernel 融起来，中间那个 `[B, H, T, T]` 的 attention matrix 都不会出现在 HBM 里。当 T=12 时省不了多少，但 T=8192 时就是几 GB 的差别——这就是现代 LLM 服务的入门票。

> 在这台 Mac 上没有 NVIDIA GPU，`05_gpu/benchmark.py` 会优雅地 skip。`matmul_naive.cu` / `matmul_tiled.cu` 需要 `nvcc` 编译；`attention_triton.py` 需要 Triton + CUDA。整个 forward 实际跑在 PyTorch 的 CPU matmul 上（Accelerate 框架的 BLAS）。

**这一层的关键**：GPU 不快在"算得多"，快在"每一行数据都在最近的 memory 里"。

---

## t≈3 s · 完成

64 个 decode step 之后（每个 ~30 ms），L6 推一个 `{"done": true}` 帧，L8 转成 `{"type": "done"}` 事件，L7 的 SSE 流结束，浏览器渲染最后的 ✓ done。

---

## 数字对齐一下

单次 `"What is the capital of France?"`，64-token 回答：

| 层 | 量级 |
|---|---|
| L7 字节 | 请求 ~80 B + 64 个 SSE token 帧 ~3 KB |
| L8 HTTP 调用 | 1 次（往 L6） |
| L6 token 数 | 12 prompt + 64 generated = 76 |
| L6 forward 次数 | 1 次 prefill (T=12) + 64 次 decode (T=1) |
| L2 matmul 次数 | (12 layer × 6 + 1) × 65 ≈ 4700 次 matmul |
| L1 浮点运算 | ~330 GFLOP（prefill 5 + 64 × decode 5 GFLOP） |
| L1 HBM 流量（权重 + KV cache） | 单次 forward ~500 MB，总计 ~32 GB |
| 实测耗时 | warm: prefill 17 ms + 64 × 10 ms ≈ 0.7 s；cold: 多 ~1.3 s 启动开销 |

GPT-2 124M 是这条链路里的"老式发动机"——慢、答得差，但每一步都看得见、改得动。把它换成 GPT-2 XL (1.5B)、Qwen-2.5-7B，数字会变（更慢、更聪明），但**每个数字背后的 matmul 还是同一个数**。这是这本"教科书"的承诺。
