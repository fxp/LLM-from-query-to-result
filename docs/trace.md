# 一个 query 的完整旅程

目标：用户敲入 `"帮我做一个 Todo 网站"`，拿到一个能跑的前后端网站。
下面按时间顺序，看这个 query 在五层里每一层长什么样。

---

## t=0 ms · L1：一串键盘输入

用户在浏览器里敲完回车。`index.html` 里的 JS 把这段字符串 POST 到后端：

```http
POST /chat HTTP/1.1
Content-Type: application/json

{"query": "帮我做一个 Todo 网站"}
```

后端 `main.py` 调 `run_agent(query, work_dir="generated/")`，建一个 SSE 响应，准备流式往回推事件。

**这一层的关键数据结构**：一个字符串。仅此而已。

---

## t=5 ms · L2：字符串变成"对话"

`run_agent` 把 query 包成 OpenAI/Anthropic 风格的消息：

```python
messages = [{"role": "user", "content": "帮我做一个 Todo 网站"}]
```

加上 `system_prompt`（定义 Agent 行为）和 `tools`（`write_file` / `run_shell` 的 JSON schema），通过 HTTPS 发给 Claude API。这个请求打到 Anthropic 的推理服务——也就是 L3。

**这一层的关键**：到这里，"自然语言需求"已经和"结构化工具集"绑在一起了。模型不是在空想，它拿到的 context 里明确列出了它能调的工具。

---

## t=50 ms · L3：字符串变成 token

Anthropic 的推理服务（概念上就是 `03_model/server.py` 的放大版）先把 prompt tokenize：

```
"帮我做一个 Todo 网站"  →  [56600, 20375, 35287, 48824, ..., 27670]
```

这一步是 BPE，用的是查表 + 合并规则，没算力消耗。但接下来这些 token 要跑 **prefill**：一次性把它们喂进模型，算出所有位置的 KV，缓存起来。

```
[prefill] B=1, T=~120 tokens
  forward 一次：shape=[1, 120, 4096 (or whatever Claude's D is)]
  耗时（A100 假设）：~50 ms
  产生 past_key_values：形状 [n_layer, 2, n_head, 120, head_dim]
```

然后进入 **decode** 循环：每次把上一轮采样出的 1 个新 token 喂进模型，用 KV cache 算下一个。

```
[decode step 0] 1 个 token 进去，采样出 "好"
[decode step 1] 1 个 token 进去，采样出 "的"
[decode step 2] 1 个 token 进去，采样出 "，"
...
```

每个 decode step 大约 5-20 ms。**这些 token 通过 SSE 一个个流回 L2**。

**这一层的关键**：prefill 一次贵（算全部 120 个位置），decode 每次便宜（只算 1 个位置），但要跑几百步。"首 token 延迟" ≈ prefill，"后续 token 速度" ≈ decode。

---

## t=55 ms（prefill 中）· L4：token 变成 tensor

**每一次 forward 内部**发生的事：

```
输入：input_ids  [B=1, T=120]
  │
  ▼  wte + wpe
  x  [1, 120, 4096]
  │
  ▼  Block 0
  │  ┌── LN ──▶ [1,120,4096]
  │  │
  │  ├── attn.c_attn(x)  → qkv [1, 120, 12288]  ← 这是个 matmul ([1,120,4096] @ [4096,12288])
  │  ├── split to q, k, v [1, 32 heads, 120, 128]
  │  ├── scaled_dot_product_attention(q, k, v, is_causal=True)
  │  │     │
  │  │     ▼  ──── 这是 attention kernel，交给 L5 ────
  │  │     Q @ K.T  → [1, 32, 120, 120]     ← matmul
  │  │     softmax  → [1, 32, 120, 120]
  │  │     @ V      → [1, 32, 120, 128]     ← matmul
  │  │     reshape  → [1, 120, 4096]
  │  │
  │  ├── attn.c_proj  → [1, 120, 4096]      ← matmul
  │  ├── + residual
  │  └── LN, MLP (两个 matmul + GELU), + residual
  │
  ▼  再过 N-1 个 block（对 Claude 假设 N≈80）
  │
  ▼  ln_f
  │
  ▼  lm_head = x @ wte.T → logits [1, 120, 100000ish]   ← matmul
```

**一次 forward 里：80 层 × 每层 ~6 个 matmul ≈ 480 个 matmul**。每个 matmul 的维度取决于 `[B, T, D]`：最大的那个（FFN 的 D→4D）在 prefill 时是 `[120, 4096] @ [4096, 16384]`，大约 16 GFLOPS。整次 prefill 在 10 TFLOPS 量级。

**这一层的关键**：到这里我们已经忘了"Todo 网站"了。剩下的全是浮点数乘加。

---

## t=55.0 到 55.3 ms（一个 matmul 之内）· L5：tensor 变成 GPU 指令

挑一个典型 matmul：`[120, 4096] @ [4096, 16384]`。

cuBLAS 会把它 launch 成一个 CUDA kernel，大致是这样：

- **Grid**：C 的形状是 `[120, 16384]`，按 `TILE_M × TILE_N = 128 × 128` 切，得到 `⌈120/128⌉ × ⌈16384/128⌉ = 1 × 128 = 128` 个 block。
- **Block 内**：256 个线程，每个线程负责算输出 tile 里的几个元素。
- **工作流**：K 维（=4096）按 `TILE_K=32` 分块，每一步
    1. 所有 256 线程协作，把 A 的一个 `[128, 32]` tile 和 B 的一个 `[32, 128]` tile 从 HBM 搬到 shared memory；
    2. 每个线程在 shared memory 里做 32 次乘加，累加到自己的寄存器里；
    3. `__syncthreads()`，下一个 K tile。
- **结束**：把累加结果写回 HBM。

整个 kernel 的 HBM 读入量 = `|A| + |B| = 120·4096·4 + 4096·16384·4 ≈ 270 MB`；
算力 = `2 · 120 · 16384 · 4096 ≈ 16 GFLOP`。
在 A100 上 ~0.05 ms 完成。

**如果用我们的 `matmul_naive.cu`**，每个 C 元素读 2×K = 8192 个 float，总读 `120 · 16384 · 8192 · 4B ≈ 65 GB`——HBM 带宽打满也要 ~45 ms（慢 1000 倍）。这就是为什么要分块。

**如果把整条 attention 用 flash-attention 融起来**（`05_gpu/attention_triton.py`），中间那个 `[1, 32, 120, 120]` 的 attention matrix 都不会出现在 HBM 里，省掉几 MB 的读写——当 T=120 时省不了多少，但 T=32768 时就是几 GB 的差别。

**这一层的关键**：GPU 不快在"算得多"，快在"每一行数据都在最近的 memory 里"。

---

## t≈2 s · 回到 L2：model 说话了

几百个 decode step 之后，Claude 吐出来的 token 流，解码回文字大概是：

```
好的，我帮你做一个 Todo 网站。我会写一个 index.html 加一个 Flask
后端。先写前端…
```

然后模型决定**调用工具**——它的输出里包含了一个结构化的 `tool_use` block：

```json
{"type": "tool_use",
 "name": "write_file",
 "input": {"path": "index.html", "content": "<!doctype html>..."}}
```

L2 的 `run_agent` 循环识别到这个 block，执行 `write_file`，把结果：

```json
{"type": "tool_result",
 "tool_use_id": "toolu_...",
 "content": "OK. 2134 bytes written to index.html."}
```

append 回 messages，再调一次 model。**这就是 Agent 循环**。

---

## t≈8 s · 结束

经过 5-8 轮循环，Agent 调了 `write_file(index.html)`、`write_file(server.py)`、`run_shell(pip install flask)` 等若干工具。最后一轮模型只返回文本（没有 tool_use），循环退出：

```
完成。运行 python generated/server.py 即可启动。
```

L2 yield 一个 `{"type": "done"}` 事件，L1 的 SSE 流结束，浏览器渲染最后一条气泡。`generated/` 里是一个能跑的网站。

---

## 数字对齐一下

单次 `"帮我做一个 Todo 网站"`，如果能看到所有层的细节：

| 层 | 量级 |
|---|---|
| L1 字节 | 请求/响应共 ~50 KB |
| L2 tool loop 轮数 | 5-8 轮 |
| L3 token 数 | ≈ 2000 输入 + 4000 输出 |
| L4 forward 次数 | ≈ 4000 次（每个输出 token 一次 decode） |
| L4 matmul 次数 | 4000 × 80 layer × 6 = ~1.9M 次 matmul |
| L5 浮点运算 | ~60 TFLOP |
| L5 HBM 读写（权重 + KV cache） | 单次 forward ~150 GB，总计 ~600 TB |

这是"一句话变一个网站"背后的账。每一层各自做好一件事，叠起来就成了 ChatGPT。

---

[← L5 · GPU 层](05-gpu.md) | [回到首页](index.md)
