# 05 · L3：自己写 KV cache 的推理服务

> [← L2 chat client](04-L2-chat-client.md) ｜ 代码：[`03_model/server.py`](https://github.com/fxp/LLM-from-query-to-result/blob/main/03_model/server.py) ｜ [下一篇 →](06-L4-transformer.md)

L3 是把 model 变成"服务"的那一层。它接收 HTTP 请求，跑 forward pass，流式返回 token。这听起来简单，但里面藏着 LLM 推理性能的整个故事——**KV cache、prefill vs decode、为什么 batched serving 是工业级 LLM 服务的命**。

这一层 ~140 行 + 0 个 transformers runtime 依赖。换成纯 PyTorch + 我们自己写的 GPT.step()。

## 这一层做什么

API：`POST /generate { prompt, max_tokens, temperature } → SSE token stream`

逻辑：
1. **Tokenize** prompt（用我们的 BPE）
2. **Prefill** 整个 prompt：一次 forward 算所有位置的 KV，缓存住
3. **Decode** 循环：从最后位置 sample 出下一个 token，喂回去（带 KV cache），算下一个 token
4. SSE yield 每个 token；遇到 EOT 或 max_tokens 停

## 之前的版本依赖 transformers，删了

最初 L3 直接用 `transformers.GPT2LMHeadModel`：

```python
out = model(input_ids=cur_ids, past_key_values=past, use_cache=True)
```

`use_cache=True` 让 transformers 内部管 KV cache。简单、能跑。

但这意味着：本 repo 真正"自己实现的"只到 L4 的架构定义为止，**KV cache 这层重要的推理优化是 transformers 帮我们做的**。

后来我把 L4 加了 `step(input_ids, kv_caches)` 方法，自己实现 KV cache 逻辑（详见 [06-L4-transformer](06-L4-transformer.md)）。然后 L3 就可以这么写：

```python
logits, kv_caches = model.step(input_ids)              # prefill
for _ in range(max_tokens):
    next_id = sample(logits[0, -1])
    logits, kv_caches = model.step(next_id, kv_caches)  # decode with cache
```

**0 行 transformers runtime**。L3 的 model 就是 L4 的 `GPT` 类。

## 双模式启动

L3 可以加载两种 ckpt：

```python
MODEL_PATH = os.environ.get("MODEL_PATH")

def load() -> Engine:
    if MODEL_PATH:
        # Load OUR trained checkpoint (L0 or L0.5 path A)
        blob = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        cfg = GPTConfig(**blob["config"])
        model = GPT(cfg).to(DEVICE).eval()
        model.load_state_dict(blob["model"])
        return Engine(model=model)

    # Default: load OpenAI's pretrained gpt2-124M via L4 from_pretrained
    model = GPT.from_pretrained("gpt2").to(DEVICE).eval()
    return Engine(model=model)
```

设了 `MODEL_PATH` 就用我们训的（7M 莎翁 model 或 124M SFT'd），不设就走 OpenAI gpt2。两条路径都是同一个 `GPT` 类——只是不同的权重。

测 SFT'd 124M：

```bash
MODEL_PATH=$(pwd)/00b_sft/out/sft_from_gpt2.pt python ../03_model/server.py
```

启动 log：

```
Loading local checkpoint -> .../sft_from_gpt2.pt on cuda...
Loaded: 124.44M params (local: n_layer=12 n_head=12 n_embd=768 block_size=1024)
  trained 930 steps, final loss: {'sft_epochs': 30, 'lr': 5e-05}
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9000
```

## /generate 实现

```python
@app.post("/generate")
async def generate(req: GenRequest) -> StreamingResponse:
    model = ENGINE.model
    eos_id = ENGINE.eos_id  # GPT-2 <|endoftext|> = 50256

    # === Tokenize ===
    ids = tokenizer.encode(req.prompt)
    input_ids = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    print(f"[prompt] {req.prompt!r} -> tokens {ids}")

    async def stream():
        # === Prefill ===
        t0 = time.perf_counter()
        logits, kvs = model.step(input_ids)
        prefill_ms = (time.perf_counter() - t0) * 1000

        for step_i in range(req.max_tokens):
            # Sample
            last = logits[0, -1]
            if req.temperature <= 0:
                next_id_int = int(last.argmax().item())
            else:
                probs = torch.softmax(last / req.temperature, dim=-1)
                next_id_int = int(torch.multinomial(probs, num_samples=1).item())

            piece = tokenizer.decode([next_id_int])
            print(f"[step {step_i}] kv_len={kvs[0][0].size(2)} -> {piece!r}")
            yield f"data: {json.dumps({'token': piece})}\n\n"

            if next_id_int == eos_id:
                break

            # === Decode (1 new token, attended against cached K/V) ===
            next_id = torch.tensor([[next_id_int]], device=DEVICE, dtype=torch.long)
            logits, kvs = model.step(next_id, kvs)
            await asyncio.sleep(0)  # let SSE flush

        yield 'data: {"done": true}\n\n'

    return StreamingResponse(stream(), media_type="text/event-stream")
```

100 行不到。两个核心调用：
- `model.step(input_ids)` — prefill（不带 cache，传进 prompt 全部 token）
- `model.step(next_id, kvs)` — decode（带上一步的 cache，只传 1 个新 token）

## Prefill vs Decode：性能的两个世界

跑一个 query，看 server log：

```
[prompt] 'Q: What is the capital of France?\nA:' -> tokens [48, 25, 1867, 318, 262, 3139, 286, 4881, 30, 198, 32, 25]
[step  0] prefill   1.79 ms  kv_len=12   -> ' Paris'
[step  1] decode    2.63 ms  kv_len=13   -> '.'
[step  2] decode    2.63 ms  kv_len=14   -> '<|endoftext|>'
```

观察：

**Prefill（step 0）= 1.79 ms**。一次 forward pass，处理整个 12 token 的 prompt。所有 12 个 layer × 12 个 head 的 K/V 都在这一步算出来。

**Decode（step 1+）= 2.63 ms 每步**。每次 forward 只输入 **1 个新 token**，但要算它跟所有过去 token 的 attention（这就是 KV cache 的 K/V 长度）。

为什么 decode 比 prefill 慢一点点（2.6 vs 1.8 ms）？理论上 decode 算的少：T_q = 1 vs T_q = 12。但实际上：

1. **kernel launch overhead**：每个 GPU 操作有 ~10 μs 的固定 launch 开销。Prefill 的 12 token 是同一个 batch 一次算完，decode 的每 1 token 都是单独 launch。所以 decode 摊到 per-token 的 overhead 占比高。
2. **cache 越长，attention 越重**：decode 时 Q 是 [1, hd]，K/V 是 [kv_len, hd]。kv_len 越长，attention matmul 量越大。

GPU 上这些都是几 ms 的事。CPU 上同样的工作 ~10 ms / token。

## 为什么 KV cache 是命

如果不缓存 K/V，每次生成新 token 都要重新算所有过去 token 的 K/V——前 N 个 token 重新过 12 层 attention。这是 O(N²) 的，T=2000 时简直没法用。

KV cache 让生成第 n 个 token 的成本变成 **O(N)**（只算新 token 的 K/V，复用所有过去的）。

但 cache 占 GPU memory：

$$
\text{cache size} = 2 \times n\_\text{layer} \times n\_\text{head} \times T \times \text{head\_dim} \times \text{dtype\_bytes}
$$

GPT-2 small（12 层 × 12 头 × 64 dim × fp16 × 2 for K and V）每 token = 36 KB。一个 8K context = 288 MB。一个用户 batch=1 占 288 MB。100 个并发用户要 28.8 GB——一张 H100 才扛得住。

这就是为什么真正的推理服务（vllm、TGI）有 **PagedAttention** / **continuous batching** 这些技术：把 cache 像虚拟内存一样分页管理，最大化 GPU memory 利用率。我们这里只支持 batch=1 单用户，演示原理就够了。

## HF mirror 自动 fallback

L4 的 `GPT.from_pretrained("gpt2")` 默认从 huggingface.co 下载权重。在中国大陆这个域名直连不通。

L4 内部加了 probe：开始下载前先 try huggingface.co，3 秒不通就自动 set `HF_ENDPOINT=https://hf-mirror.com`，再额外设 `HF_HUB_DISABLE_XET=1` 绕开慢的 Xet CDN。控制台会打印：

```
HF Hub direct unreachable; using mirror: https://hf-mirror.com
(set HF_HUB_DISABLE_XET=1, HF_HUB_DOWNLOAD_TIMEOUT=60)
```

这是 L3 端到端"完全自动"的关键。用户不用管 env var。详见 [06-L4-transformer](06-L4-transformer.md) 的 `_probe_and_set_hf_endpoint`。

## 实测端到端时序

完整一次 chat 请求：

```
0.0 ms    浏览器 POST /chat
3.5 ms    L1 backend 收到，调 run_agent
4.0 ms    L2 build prompt = "Q: ...\nA:"
4.5 ms    L2 POST /generate to L3
5.0 ms    L3 收到
5.2 ms    BPE encode (handwritten, ~250 行 Python) = 12 tokens
6.0 ms    model.step(input_ids) - prefill
7.79 ms   prefill done (1.79 ms GPU forward), logits[0,-1] -> argmax = ' Paris'
7.9 ms    L3 yield SSE "data: {token: ' Paris'}"
8.0 ms    L2 收到 SSE 帧, yield event
8.1 ms    L1 收到 event, yield SSE 帧
8.2 ms    浏览器收到, render(' Paris') -- 屏幕上看到 "Paris"
10.8 ms   L3 model.step(next_id, kvs) for '.' = 2.63 ms
11.0 ms   yield '.'
13.6 ms   L3 model.step(next_id, kvs) for EOT = 2.63 ms
13.8 ms   yield {done: true}
14.0 ms   浏览器看到 "✓ done"
```

**14 ms 端到端**——按下 Enter 到看到完整答案 `Paris.` 的总耗时。比眨眼快。

注意这里的最大开销不是 forward（1.8 ms + 2 × 2.6 = 7 ms = GPU work），而是网络/调度（~7 ms）。如果 model 是 GPT-4 那种规模（每个 token forward ~50 ms），那 forward 就主导了，网络变成噪音。

## 这一层的"最小"在哪里

- **Batch size = 1**：每次 /generate 处理一个用户。生产推理服务（vllm、SGLang）用 continuous batching 把多个用户的请求 fold 成一个 GPU forward——5-10× 吞吐量提升。
- **没有 PagedAttention**：cache 是连续 tensor。生产里要分页管理。
- **没有 quantization**：fp32/fp16/bf16。生产里 INT4/INT8/FP8 quantization 减一半显存、加速 2×。
- **没有 multi-GPU**：单卡。多卡需要 tensor parallel 或 pipeline parallel。
- **没有 speculative decoding**：让小 model 先 draft、大 model verify。
- **没有 prefix caching**：相同 prompt 前缀重用 KV cache。

每个删掉的特性都是工业级 LLM serving 的关键，但**对教学来说**，省略后留下的"prefill + decode + KV cache"才是核心。

## 接口（往下）

L3 跟 L4 通过两个方法调用：

```python
# Inference (KV cache 友好)
logits, kv_caches = model.step(input_ids)              # prefill
logits, kv_caches = model.step(next_id, kv_caches)     # decode

# 一次性 forward（训练或简单推理用）
logits = model(input_ids)
logits, loss = model(input_ids, targets=y)             # for training
```

`GPT` 类是 L4 实现的，下一篇（其实是两篇——transformer 架构 + BPE）讲它内部到底什么样。

## 下一篇

L3 调 `model.step` —— 这个 model 是个手写 GPT-2，330 行实现 embed / MHA / FFN / LN / KV cache。下一篇拆开看。

[L4a — 300 行手写 GPT-2 →](06-L4-transformer.md)
