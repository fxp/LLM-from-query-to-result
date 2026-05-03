# 我用 70 秒训出了一个能调工具的小 Agent

> 这是 [LLM-from-query-to-result](https://github.com/fxp/LLM-from-query-to-result) 项目的浓缩版叙述——把这个 11 篇分章博客 + 8 层代码的项目讲成一个故事。如果你想深入某一层，每段后面都有指向详细文档的链接。

---

## 一个让我难堪的问题

去年某次面试，我在白板上写 transformer attention，画 Q@K.T → softmax → @V 三步。面试官问我：**"那 KV cache 是什么？为什么它让 generate 从 O(N²) 变成 O(N)？"**

我答得磕磕巴巴。论文里那句"cache key and value tensors from previous timesteps"我背得出来，但具体到代码——某个 step 里 K 是 [B, h, T, d]，cache 完之后下一 step 的 K 怎么拼接、为什么不需要重算前面的 K——我画不出来。

那一刻我意识到：我读过 transformer 论文，用过 OpenAI API，看过 vllm 的 README，但**整个 LLM 工程栈对我来说还是个黑盒**。每次有同事问"我们推理服务为什么慢"，我只会建议"换更好的 GPU"或"加 batch"——没有结构化的诊断框架。

为了治这个病，我把整条链路从浏览器一直撕到 GPU 浮点乘法，自己用 ~10K 行 Python + 一点 CUDA 全部重写了一遍。**没有外部 LLM API。没有外部模型权重（除了可选 fallback）。没有 transformers / tiktoken / vllm 在推理路径。**

成果：一个 8 层栈，在 RTX 5090 上 70 秒训出能调工具的小 agent。

```
👤 What is 1234 plus 5678?
💭 I need to compute 1234 + 5678.       ← 我们训的 124M model 思考
🔧 calc(1234 + 5678)                    ← model 决定调 calc 工具
   ↳ 6912                                ← 我们写的 7 行 Python eval 真算的
🤖 6912.                                  ← model 把工具结果包成答案
```

注意 **1234+5678 不在训练数据里**。SFT 只给过模型 1-99 范围的加法。但 model 在推理时正确生成 `calc(1234 + 5678)`，由 Python 真算出 6912。

这就是 agent 的本质——**model 能力 = 知识 + 工具扩展，不靠脑补**。同样的机制，ChatGPT 接 web_search，Cursor 接 grep + edit，Claude 接 tool_use API。规模不同，本质一样。

下面我把这个项目的故事讲给你。

---

## 八层架构

```
┌──────────────────────────────────────────────────────────────┐
│ L0   00_train/        预训练循环（莎士比亚 → 7M GPT, 12 秒）│
│ L0.5 00b_sft/         SFT (instruction tuning, 28 秒)        │
│ L0.6 00c_agent_sft/   Agent SFT (ReAct + tools, 33 秒)      │
│ L1   01_app/          Web UI + SSE 流式输出                  │
│ L2   02_agent/        Chat 客户端 + ReAct agent loop         │
│ L3   03_model/        推理服务（手写 KV cache）              │
│ L4   04_transformer/  GPT-2 架构（330 行）+ BPE（230 行）    │
│ L5   05_gpu/          CUDA matmul + Triton flash-attention   │
└──────────────────────────────────────────────────────────────┘
```

每层独立可跑，独立 README，独立 blog 文章。但它们一起讲了一个故事——一个 query 怎么变成一个回答。

---

## L0 → L0.5 → L0.6：从随机权重到 instruct agent，70 秒

### L0：从零训出一个 GPT (12 秒)

数据是 1.1 MB 的 [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)——莎翁 33 个剧本拼起来的纯文本。模型是 GPT-2 架构的小版本：4 层、4 头、128 维，**7.24M 参数**。

```bash
cd 00_train && python prepare.py && python train.py
```

数据 prepare 1 秒，BPE tokenize 后得到 304K 训练 token + 33K 验证 token。训练 1000 步，**RTX 5090 上 12.2 秒**。

最关键的数字：**`step 0 loss = 10.815`**。

为什么？随机权重时模型对 50,257 个 token 的概率分布是均匀的，每个位置预测对的概率是 1/50,257。Cross-entropy loss = -log(1/50,257) = ln(50,257) ≈ 10.825。我们看到 10.815，吻合到小数点后两位——验证我们的权重初始化做对了。

> 早期版本我没做正确的 GPT-2 初始化，初始 loss 是 80。原因是 PyTorch 默认 Kaiming uniform 让 transformer 初始 logits 方差太大，softmax 严重偏向某些 token，cross-entropy 巨大。修复是 N(0, 0.02) 初始化 + 残差出口缩 1/√(2N)——nanoGPT 标准做法。详见 [blog/01-L0-training](01-L0-training.md)。

12 秒后 loss 降到 4.55。Sample 一下：

```
$ python sample.py "ROMEO:"
KING RICHARD II:
DUKE VINCENTIO:
O, you so.
ISABETH:
BRUTIO:
Come, good go.
ROMEO:
My wife, the name as thou, as no law,hip, go
```

学到了莎翁的格式（角色名 + 冒号 + 换行 + 台词）、真实角色名（KING RICHARD II、DUKE VINCENTIO 都在剧本里）、`'tis`、`thou` 这种风格。**没学到语义**——这是 7M 参数 + 0.34M token 训练 12 秒的诚实样子。学到结构，没学到含义。

### L0.5：把 base 变成 instruct (28 秒)

这个 7M base 现在被问 "What is the capital of France?" 会答什么？

```
A: KING RICHARD III: I'll be a man, I have not, I have not, ...
```

它不知道"问"和"答"是两件事。它学到的是**续写**——给一段文本，预测最可能的下一段文本。它见过的 token 全来自莎翁剧本，所以续写就是莎翁台词。

让它学问答，需要**SFT (Supervised Fine-Tuning)**。原理简单：

```
对每条 (question, answer):
   prompt = "Q: <question>\nA:"        ← 不计 loss（mask）
   target = " <answer><|endoftext|>"   ← 计 loss
```

把 prompt + target 拼起来送进模型做 next-token prediction，loss 只算 target 部分。继续训几个 epoch，模型学到三件事：
- 看到 `Q: ... A:` 就开始答
- 答完 emit `<|endoftext|>` 停下
- 对训练里见过的 fact 直接 parrot

我手写了 242 条 Q/A（地理、算术、化学、文学、动物、货币、几何），用 100 epochs × 16 batch_size SFT。**5090 上 26 秒**：

```
epoch  0/100 | loss 9.65
epoch 50/100 | loss 0.20
epoch 99/100 | loss 0.02
```

之后再问：

```
Q: What is the capital of France?
A: Paris.
```

完美。**24 秒把"接龙莎翁"变成"答 Paris"**。

但这个 model 仍然只对训练数据里的事实有效。问 "100 + 100" 它答 "13"——胡编。问 "capital of Mars" 它答 "Moscow"——纯乱说。**SFT 教格式，不教知识**。要让 SFT 之后真有知识泛化，base 模型需要在大语料上预训过——比如 OpenAI gpt2 在 8 GB WebText 上训过。

所以 L0.5 还有 path B：用 OpenAI 公开的 gpt2-124M 当 base，同样 SFT 我的 242 条 Q/A。**5090 上 35 秒**。然后：

```
Q: What is the capital of Iceland?       A: Reykjavík.    ← 不在 SFT data!
Q: What is the largest river in Africa?  A: The Nile.     ← 不在 SFT data!
Q: Who wrote Pride and Prejudice?        A: William Shakespeare.   ← 124M 不知 Austen
```

**Iceland → Reykjavík** 不在 SFT 训练里。124M base 在 WebText 上预训过，知识在权重里。SFT 只是把它"接"出来到指令格式。这是大 base + 小 SFT 数据的力量。

详见 [blog/02-L0.5-sft](02-L0.5-sft.md)。

### L0.6：教 124M 调工具 (33 秒)

L0.5 让 model 能答问，但仍是"凭脑补答"。问 "1234 + 5678" 它答错。我们要让它**调用工具**。

格式选 ReAct（[Yao et al. 2022](https://arxiv.org/abs/2210.03629)）：

```
Q: What is 23 plus 47?
THOUGHT: I need to compute 23 + 47.
ACTION: calc(23 + 47)
OBSERVATION: 70                    ← 由真工具产生，不是 model 写的
ANSWER: 70.<|endoftext|>
```

**关键 invariant**：OBSERVATION 内容必须由代码控制，不能让 model 自己写——否则就是幻觉工具结果。

工具有两个：
- `calc(expr)` — Python eval 加 sandbox（regex 限定 `[\d+\-*/().\s]+`，无 builtins）
- `lookup(key)` — 查 105 条 KB 事实

数据 258 条 traces，`build_data.py` **程序合成**——每条 trace 的 `observation` 字段在生成时调用真工具产生，保证训练数据 ≡ 运行时输出。

**Loss masking 是这一层最 tricky 的细节**：

```python
parts.append((f"Q: {q}\n",                  False))  # 用户输入
parts.append((f"THOUGHT: {thought}\n",      True))   # model emit
parts.append((f"ACTION: {action}\n",        True))   # model emit
parts.append(("OBSERVATION:",               True))   # ← 这五个字符必须 LEARN
parts.append((f" {observation}\n",          False))  # ← 但内容必须 MASK
parts.append((f"ANSWER: {answer}",          True))   # model emit
```

第一版我把整段 OBSERVATION 都 mask 了。结果：

```
THOUGHT: I should look up capital of France.
ACTION: lookup(capital of France)
ACTION: lookup(capital of France)         ← 第二个 ACTION！
ACTION: lookup(...)
... (循环)
```

Model 不会 emit `OBSERVATION:` 这个 stop signal——因为那段被 mask 了，backprop 没经过。改对（prefix LEARN，content MASK），model 立刻学会"emit ACTION 后接 emit OBSERVATION:"。

这是 agent SFT 数据 masking 最容易踩的坑，也是教学价值最大的细节。详见 [blog/10-L0.6-agent](10-L0.6-agent.md)。

20 epochs 训完，**5090 上 26 秒**。然后真正的 agent loop：

```
👤 Q: What is 1234 plus 5678?

L2 (agent_loop.py):
  prompt = "Q: What is 1234 plus 5678?\n"
  
  # generate, stop at "OBSERVATION:" or EOT
  chunk = generate_until(prompt, stop=["OBSERVATION:"])
  # chunk = "THOUGHT: I need to compute 1234 + 5678.\nACTION: calc(1234 + 5678)\n"
  
  # parse last ACTION
  action = "calc(1234 + 5678)"
  
  # call REAL Python tool
  obs = tools.call(action)  # → "6912"
  
  # inject into prompt and continue
  prompt = prompt + chunk + f"OBSERVATION: 6912\n"
  
  # generate again, stops at EOT
  chunk2 = generate_until(prompt, stop=["<|endoftext|>"])
  # chunk2 = "ANSWER: 6912."
  
🤖 6912.
```

10 个测试 query，**8/10 完全正确**：

| Query | 输出 | |
|---|---|---|
| capital of France? | Paris. | ✓ |
| **1234 plus 5678?** | **6912.** | **✓ 关键泛化** |
| Who wrote Hamlet? | Shakespeare. | ✓ |
| 8 times 7? | 56. | ✓ |
| chemical symbol of gold? | Au. | ✓ |
| capital of Mongolia? | not found. | ✓ 诚实（KB 没有） |
| How are you? | hallucinate `lookup(author)` | ✗ OOD |

`1234+5678 → 6912` 是这个项目最重要的 demo——**model 学到的不是数字，是格式契约**。它不会算大数，但它知道 calc 工具能算任意数字。

---

## L1 + L2 + L3：从浏览器到 KV cache

### 浏览器里那一个个蹦出来的字

ChatGPT 给人"一个字一个字蹦出来"的感觉，**不是动画**。是真的——服务端真的一个字一个字通过网络送给浏览器。

这件事的实现技术叫 **SSE (Server-Sent Events)**：一条普通的 HTTP 响应，body 不一次性写完，每次 flush 一段 `data: <json>\n\n` 帧。浏览器用 `fetch + ReadableStream` 一段一段读、一段一段渲染。

```javascript
// 整个前端 SSE 读取逻辑，~20 行
const resp = await fetch('/chat', { method: 'POST', body: JSON.stringify({query}) });
const reader = resp.body.getReader();
const dec = new TextDecoder();
let buf = '';
while (true) {
  const {value, done} = await reader.read();
  if (done) break;
  buf += dec.decode(value, {stream: true});
  let i;
  while ((i = buf.indexOf('\n\n')) >= 0) {
    const frame = buf.slice(0, i); buf = buf.slice(i + 2);
    if (!frame.startsWith('data: ')) continue;
    const ev = JSON.parse(frame.slice(6));
    render(ev);  // append to log div
  }
}
```

后端用 FastAPI `StreamingResponse + Python generator` 实现：

```python
@app.post("/chat")
def chat(req: ChatRequest):
    def sse():
        for event in run_agent(req.query):  # this is a generator from L2
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream")
```

整个 L1 = 80 行 HTML/JS + 50 行 Python。详见 [blog/03-L1-app](03-L1-app.md)。

### 推理服务自己写 KV cache

L3 接收 prompt，跑 forward，流式吐 token。最有意思的是 KV cache。

朴素 generate：每次产生新 token 都要重新 forward 整个序列。第 n 步需要算前 n 个 token 的所有 attention——O(n²)。Sequence 长 8K 时，attn matrix 一次 forward 就是 GB 级，根本没法用。

KV cache：缓存每层的 K/V，每个新 step 只算新 token 的 q + 和缓存的 K/V 做 attention。**O(N) 摊销，每个 token forward 时间常数**。

我手写的 KV cache 实现：

```python
@torch.no_grad()
def step(self, input_ids, kv_caches=None):
    B, T = input_ids.shape
    if kv_caches is None:
        kv_caches = [None] * len(self.h)
        pos_offset = 0
    else:
        pos_offset = kv_caches[0][0].size(2)  # past sequence length
    pos = torch.arange(pos_offset, pos_offset + T, device=input_ids.device)
    x = self.wte(input_ids) + self.wpe(pos)
    new_caches = []
    for block, kv in zip(self.h, kv_caches):
        x, new_kv = block(x, kv)  # block 内部 cat past K/V with new K/V
        new_caches.append(new_kv)
    x = self.ln_f(x)
    logits = x @ self.wte.weight.T
    return logits, new_caches
```

数值正确性验证：

```python
# Test: full forward (T tokens at once) ≡ step() with prefill + 1 decode
logits_full = m(full_input)
logits_pre, kvs = m.step(full_input[:, :T-1])
logits_dec, _ = m.step(full_input[:, T-1:T], kvs)

print((logits_full[0,-1] - logits_dec[0,-1]).abs().max())
# 2.4e-7  ← 浮点累加误差，等价
```

5090 上实测：

| 操作 | 用时 |
|---|---|
| Prefill (12 token prompt) | 1.79 ms |
| Decode (per token) | 2.63 ms |
| 单 batch 吞吐 | 380 tok/s |

Prefill 比 decode 还快——原因是 prefill 12 个 token 是一次 GPU launch，decode 每 1 token 单独 launch，launch overhead 占主导。这是 small-batch 推理的真实样子。详见 [blog/05-L3-inference-server](05-L3-inference-server.md)。

---

## L4：300 行手写 GPT-2 + 230 行手写 BPE

### Transformer 内部

整个 GPT-2 架构 ~330 行 Python：

```python
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        # lm_head 共享 wte 权重，省 ~40M 参数

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.wte(input_ids) + self.wpe(pos)
        for block in self.h:
            x, _ = block(x)
        x = self.ln_f(x)
        logits = x @ self.wte.weight.T  # tied
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1), ignore_index=-1)
        return logits, loss
```

每个 Block：

```python
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, kv_cache=None):
        attn_out, new_kv = self.attn(self.ln_1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv
```

整个 attention：

```python
def forward(self, x, kv_cache=None):
    B, T, D = x.shape
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    head_dim = D // self.n_head
    q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
    k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
    v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
    if kv_cache is not None:
        past_k, past_v = kv_cache
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
    is_causal = kv_cache is None
    y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    y = y.transpose(1, 2).contiguous().view(B, T, D)
    return self.c_proj(y), (k, v)
```

`F.scaled_dot_product_attention` 在 GPU 上自动调 [Flash Attention 2 kernel](https://arxiv.org/abs/2307.08691)——把 Q@K.T、softmax、@V 三个 op 融成 1 个 kernel，省掉中间 [B, h, T, T] attention matrix 的 HBM 读写。当 T=8192 时这个 matrix 有 GB 级大，省它就是命。

详见 [blog/06-L4-transformer](06-L4-transformer.md)。

### BPE bit-for-bit 等价 tiktoken

GPT-2 BPE 算法 4 个组件：

1. **bytes_to_unicode 映射**：每个 byte (0..255) 映射到一个可打印 unicode 字符。这样 BPE 操作的是 str，不是 bytes，但保持 byte-level 兼容性。
2. **Regex 预切词**：`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`——把字符串切成"词"，BPE 只在词内合并。
3. **BPE merge**：每个词初始拆成字符元组，反复找优先级最高的相邻 pair 合并，直到没有可合并的。Merge 优先级在 `vocab.bpe` 里。
4. **encode/decode**：text → 预切词 → byte→unicode 映射 → BPE merge → 查表 → token id 序列。Decode 反过来。

230 行 Python。验证：

```
$ python bpe.py
✓ 'Hello, world!'                            ours=4   ref=4   match=True
✓ 'The quick brown fox jumps...'             ours=10  ref=10  match=True
✓ 'ROMEO:\nO Juliet, wherefore art thou?'    ours=12  ref=12  match=True
✓ 'Question: What is the capital of France?' ours=12  ref=12  match=True
✓ '  multiple   spaces   and\ttabs\n\nnewlines' ours=15 ref=15 match=True
✓ '中文 日本語 🚀 emoji'                       ours=14  ref=14  match=True
✓ '1234567890 + - * / = (test)'              ours=12  ref=12  match=True
ALL MATCH
```

7/7 完全一致——中文、日文、emoji、特殊字符无一差错。详见 [blog/07-L4-bpe](07-L4-bpe.md)。

---

## L5：一次矩阵乘在 GPU 上到底怎么跑

跑一次 GPT forward 的 92% 时间花在 matmul 上。所以 LLM 推理性能 ≈ matmul 性能。

GPU 内存层级：

```
HBM (40-80 GB, ~1.5 TB/s)  ← 模型权重
  ↓ load (慢)
L2 cache (~40 MB, ~4 TB/s)
  ↓
Shared memory (per SM, ~100 KB, ~20 TB/s)  ← 分块时数据搬这里
  ↓
Registers (per thread, ~256 × 32-bit)  ← 计算在这里
```

GPU 算得比读得快太多。**朴素 matmul 慢不是因为算不过来，是因为一直在读 HBM**。

`matmul_naive.cu`：每个 thread 算输出一个元素。row=0 的 1024 个 thread 都读 A 第 0 行——HBM 上同一段数据被读 1024 次。

`matmul_tiled.cu`：一个 block 的 256 个 thread 协作，把 A 的 [128, 32] 块和 B 的 [32, 128] 块**一次性**搬到 shared memory，然后每 thread 在 shared 里做 32 次乘加。HBM 流量 32× 减少。

实测 RTX 5090 (fp32, 2048×2048×2048)：

| Kernel | 用时 | TFLOPS |
|---|---|---|
| naive | 2.39 ms | 7.21 |
| tiled | 1.86 ms | 9.24 |
| **cuBLAS (TF32 + Tensor Core)** | **0.25 ms** | **68.94** |

cuBLAS 比 tiled 快 7.5×。不是因为算法更优——是因为它用 Tensor Core 跑 TF32（10-bit mantissa），算力比 fp32 高 2-4 倍，加上 register tiling、async copy、software pipelining 这些深层优化。

**Triton flash-attention** 是另一个明星：

| 实现 | 用时 |
|---|---|
| PyTorch unfused (3 kernels) | 1.01 ms |
| Triton fused | **0.12 ms (8.4× faster)** |

trick：把 Q@K.T、softmax、@V 三步**融**进一个 kernel——用 online softmax 数学技巧，让 attention 一行可以增量计算，**永远不用把 [B, h, T, T] 矩阵存到 HBM**。当 T=8192 时省掉的 round-trip 是几 GB。

详见 [blog/08-L5-gpu](08-L5-gpu.md)。

---

## 端到端时序：14 ms

把所有层串起来。用户在浏览器输入 "What is 1234 plus 5678?"，按回车：

```
t=0     用户按 Enter
t=1ms   L1 FastAPI 收到 POST /chat
t=2ms   L1 调 run_agent(query) → L2 build prompt → POST /generate to L3
t=5ms   L3 BPE encode prompt → 12 tokens
t=6ms   model.step(input_ids) prefill (1.79 ms GPU)
t=8ms   logits[0,-1] argmax → "THOUGHT:" token → L3 yield → L1 SSE → 浏览器看到 💭
t=...   model 流式生成 THOUGHT、ACTION 行（每 decode step 2.63ms）
t=22ms  agent loop 检测到 "OBSERVATION:" stop string
t=22ms  parse last ACTION = "calc(1234 + 5678)" → call tools.calc → "6912"
t=23ms  L1 看到 ↳ 6912
t=...   prompt += chunk + "OBSERVATION: 6912\n", L3 继续 generate
t=27ms  ANSWER: 6912.<|endoftext|> 完成
t=28ms  浏览器看到 [done]
```

总共 ~28 毫秒。比眨眼快。

更详细的时序见 [blog/09-end-to-end-trace](09-end-to-end-trace.md)。

---

## 几个实战 takeaway

### 1. Loss masking 是 SFT 最容易踩的坑

每条训练样本的每个 token 都要明确标"学"或"mask"：
- 用户输入：mask（model 接收）
- Model 自己 emit：learn
- 工具输出（agent 场景）：mask
- 但 OBSERVATION: 这种 stop signal prefix：**learn**（让 model 学会发停止信号）

错一个 token，行为就坏。我在 L0.6 的第一版整段 OBSERVATION mask，model 训完循环 emit ACTION——花了 30 分钟才定位。修复后 1 行代码改动，立刻 work。

### 2. 现代 GPU 太快反而暴露架构 bottleneck

5090 上手写 tiled matmul 比 naive 只快 1.3×，A100 上是 6×。原因：5090 的 HBM3e 带宽 3 TB/s，naive 实现已经不再 memory-bound——L2 cache 吃住了大部分复读。

教学含义：**memory hierarchy 优化的相对重要性，跟硬件代际是反相关的**。十年前的 CPU/GPU 上 cache locality 就是命；今天的高端 GPU 上 cache 已经够用，瓶颈往往是其他地方（kernel launch overhead、数据精度、Tensor Core 利用率）。

### 3. agent 的本质是协议 + 工具，不是更聪明的 model

124M model 自己绝对算不对 1234+5678（chat-SFT 版本我们验证过会答错）。带 calc 就行。这跟 ChatGPT 接 web_search、Cursor 接 grep+edit、Claude 接 computer use 是同一个本质——**model 学的是工具调用的格式契约 + 何时调用，不是工具本身的能力**。

### 4. 整套 from-scratch 训练 < 一次 OpenAI gpt2 weights 下载

```
完整 cold-start (5090, CN 区):
  git clone               9 秒
  pip install            15 秒
  L0 train               12 秒
  L0.5 path A SFT        26 秒
  L0.5 path B SFT (含下载) ≈ 6 min 30 秒  ← 一次性 124M 权重下载
  L0.6 agent SFT         26 秒
  L5 build + benchmark   10 秒
  E2E web                 5 秒
─────────────────────────────────
总计 (含下载)              ≈ 17 分钟
总计 (无下载)              ≈ 100 秒
```

讽刺的是：**整个 from-scratch 训练耗时 < 一次模型权重下载耗时**。这说明现代 LLM 工程的本质：**算法没那么贵，数据/权重才贵**。

跳过 path B（不需要 OpenAI 权重），整个 from-scratch 路径——clone + install + L0 + L0.5A + L3 启动 + 浏览器答 "Paris."——**60 秒以内**。

---

## 收尾：这个项目想给你什么

我希望读完之后你能带走的东西：

1. **抽象塌缩到具体**。下次有人说"transformer 里有 attention"，你脑子里浮现的不是论文图，是 `c_attn = nn.Linear(D, 3*D); split → reshape → scaled_dot_product_attention(is_causal=True)` 的具体代码，是 `[B, h, T, head_dim]` 的具体张量形状，是 cuBLAS launch 的具体 grid 配置。

2. **数字感**。loss = 10.815 = ln(50,257)、prefill 1.8 ms、Triton 8.4× 加速、KV cache 数值精度 < 1e-6——这些不是教程上的概念，是你自己机器上跑出来的具体值，背后都连接到一行行代码。

3. **形态感**。当下次有人说"我们推理服务慢"，你能在脑子里算：12 层 × 8K context × 768 dim × 2 (K, V) × fp16 = 37 MB per sample, batch 1 占 37 MB GPU memory, 100 个并发用户 3.7 GB, 48 GB 卡能扛 1300 用户。这是**形态感**——抽象塌缩出来的工程直觉。

4. **一种学习方法**。从抽象塌缩到具体，再从具体爬回抽象——但这次抽象有具体的 token 流量、有具体的 ms 数字、有具体的 forward 路径。这种"具体 ↔ 抽象"的 loop 不止适用于 LLM，对任何复杂系统都管用。

完整代码、blog、实验报告都在 [github.com/fxp/LLM-from-query-to-result](https://github.com/fxp/LLM-from-query-to-result)。读了有想法欢迎交流。

---

*本文是分章 11 篇 [blog/](README.md) 系列的浓缩版。如果你想深入某一层，每个 ↑ 链接都指向更详细的写法。*
