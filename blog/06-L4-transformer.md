# 06 · L4a：300 行手写 GPT-2

> [← L6 inference server](05-L3-inference-server.md) ｜ 代码：[`04_transformer/model.py`](https://github.com/fxp/LLM-from-query-to-result/blob/main/04_transformer/model.py) ｜ [下一篇 →](07-L4-bpe.md)

GPT-2 的论文 13 页，重点全在前 4 页。模型本身少得令人惊讶：一个 transformer block 就 5 个操作（LayerNorm × 2，Attention，MLP，两个 residual），整个 GPT-2 small 就 12 个相同的 block 串起来。

这一篇把 330 行的 PyTorch 实现拆开看。每一段代码我都给一个"为什么"。

## 整体形状

```
input_ids [B, T]
   │
   ▼ wte + wpe          token + position embedding
   x [B, T, D]
   │
   ▼ Block × N          (LayerNorm, MHA, residual, LayerNorm, MLP, residual)
   x [B, T, D]
   │
   ▼ ln_f               final LayerNorm
   x [B, T, D]
   │
   ▼ x @ wte.T          tied lm_head
   logits [B, T, V]
```

GPT-2 small：N=12 layers, D=768, V=50257. **124 M 参数**。
我们 L3 用的小版本：N=4, D=128, V=50257. **7 M 参数**（其中 6.4M 是 embedding，所以非 embedding 只有 0.79M）。

## Config

```python
@dataclass
class GPTConfig:
    vocab_size: int = 50257       # GPT-2 BPE
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024        # max sequence length (positional embedding 大小)
    dropout: float = 0.0
```

`block_size` 是这个 model 能处理的最大上下文。GPT-2 是 1024，GPT-4 大概是 128K——后者用了 [RoPE](https://arxiv.org/abs/2104.09864) / sliding window 等技巧把 max context 撑大。这里学习目的，1024 够。

## CausalSelfAttention（多头自注意力）

整段代码：

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # combined Q, K, V projection
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)      # output projection

    def forward(self, x, kv_cache=None):
        B, T, D = x.shape
        qkv = self.c_attn(x)                              # [B, T, 3D]
        q, k, v = qkv.split(self.n_embd, dim=2)           # 3 × [B, T, D]
        head_dim = D // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # [B, h, T, hd]
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)             # 拼上过去的 K
            v = torch.cat([past_v, v], dim=2)
        is_causal = kv_cache is None  # decode (T_q=1) 不需要 causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.c_proj(y), (k, v)
```

**几个值得注意的设计**：

### 1. combined Q/K/V projection

`c_attn = nn.Linear(D, 3*D)`：一个矩阵乘把 x 投影出 q/k/v 三块，然后 `split` 切开。等价于三个独立的 `nn.Linear(D, D)`，但**少一次 matmul launch**（3 个 small matmul → 1 个 medium matmul，GPU 上更快）。GPT-2 论文这么做，我们照搬。

### 2. multi-head reshape

```python
q.view(B, T, n_head, head_dim).transpose(1, 2)  # [B, h, T, hd]
```

每个 head 看 **D / n_head = 64** 维的子空间。`transpose` 把 head 维度放前面，方便 PyTorch 把每个 head 当独立 batch 来算 attention。

### 3. KV cache

```python
if kv_cache is not None:
    past_k, past_v = kv_cache
    k = torch.cat([past_k, k], dim=2)
    v = torch.cat([past_v, v], dim=2)
```

如果调用方传了过去的 K/V（来自前面 step），我们就把当前的 K/V 拼上去。`q` 不拼——只用最新的 query。

这就是 **autoregressive decode 的核心**：每一步 q 是 [B, h, 1, hd]（1 个新 query），k/v 是 [B, h, T_total, hd]（所有过去 + 这步的）。

### 4. is_causal 是个细节坑

```python
is_causal = kv_cache is None
y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
```

`scaled_dot_product_attention` 内置 causal mask，但**只支持 T_q == T_k 的方阵**。

- Prefill (T_q = T_k = 12)：causal mask 有意义，position i 只能看 ≤ i 的 K。✓ `is_causal=True`
- Decode (T_q = 1, T_k = N)：不是方阵，causal mask 不适用。但 q 是新 token，k 包含所有过去 + 新——**单个 q 对所有 k 算 attention 就是对的**。✓ `is_causal=False`

这一行 `is_causal = kv_cache is None` 解决两种情况的差异。

> 💡 **`scaled_dot_product_attention` 是 PyTorch 2.0+ 的新 API**。它在 GPU 上自动调用 [Flash Attention 2](https://arxiv.org/abs/2307.08691) 的 fused kernel——把 Q@K.T、softmax、@V 三步融成一个 kernel，省掉中间 [B, h, T, T] attention matrix 的 HBM 读写。当 T=8192 时这个矩阵是 GB 级，省它就是命。详见 [L1 那一篇](08-L5-gpu.md)。

## MLP

```python
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))
```

D → 4D → D 的"feed-forward"，中间用 GELU 激活。**4× 这个倍数**是 transformer 论文的默认值，没人改过——它给模型最多的"思考空间"放每层的非线性变换。

`approximate="tanh"` 是 GPT-2 的精确实现细节（用 tanh 近似的 GELU）。如果要严格对齐 OpenAI 权重就得这么写，差一点权重 load 进来 forward 不一致。

## Block

```python
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp  = MLP(cfg)

    def forward(self, x, kv_cache=None):
        attn_out, new_kv = self.attn(self.ln_1(x), kv_cache)
        x = x + attn_out                # residual
        x = x + self.mlp(self.ln_2(x))  # residual
        return x, new_kv
```

GPT-2 用 **pre-LN**（LN 在 attn / mlp 之前）。GPT-1 / 原始 transformer 用 post-LN（LN 在 residual 之后）。Pre-LN 训练更稳定——原 transformer 论文里 post-LN 经常梯度爆炸，要用 warmup + 仔细 init 才能训。

两个 residual：`x + attn(LN(x))` 和 `x + mlp(LN(x))`。Residual 让梯度在深网里能直接 backprop 到底层，是深度学习最重要的发明之一（[ResNet](https://arxiv.org/abs/1512.03385)）。

## GPT 主类

```python
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte  = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token embedding
        self.wpe  = nn.Embedding(cfg.block_size, cfg.n_embd)  # learned position
        self.h    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        # lm_head shares weights with self.wte (tied) — saves ~40M params
        self.apply(self._init_weights)
        std_resid = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=std_resid)
```

**几件事讲清楚**：

### Position embedding

`wpe` 是 **learned** position embedding——不是 transformer 论文里那个 sine/cosine 的固定 embedding，也不是现代用的 [RoPE](https://arxiv.org/abs/2104.09864)。GPT-2 选了最简单的：每个位置有个可学 vector。代价是 `block_size` 写死了——超过 1024 token 不能用。

### Weight tying

`lm_head` 共享 `wte.weight`。具体在 forward 里：

```python
logits = x @ self.wte.weight.T  # [B, T, V]
```

而不是再定义一个 `nn.Linear(D, V)`。两个好处：
1. **省 ~40M 参数**：V × D = 50257 × 768 = 38.5M
2. **共享语义**：embedding 矩阵的行向量就是各 token 的"输入意义"，转置后用作 logits 的"输出预测"——同一个 token 的输入/输出表示是同一个。

### 权重初始化

```python
self.apply(self._init_weights)         # all Linear/Embedding 用 N(0, 0.02)
for name, p in self.named_parameters():
    if name.endswith("c_proj.weight"):
        nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
```

两条规则：

1. **N(0, 0.02)** for 所有 Linear weights 和 embedding。比 PyTorch 默认 Kaiming 小得多——transformer 配 Kaiming 初始 logits 方差大，softmax 偏向某个 token，cross-entropy 巨大（~80），训练前几百步全在挣扎。
2. **Residual c_proj 的 weights 缩 1/sqrt(2N)**。每个 block 有 2 个 residual 出口（attn 和 mlp 的 c_proj），总共 2N 个累加。如果不缩，残差流的方差会随深度爆炸。这是 [GPT-2 论文](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 第 2.2 节的细节。

我之前没做这个初始化，初始 loss = 80（vs 理论值 10.82）。修复后第一行就是 `step 0 | loss 10.815`——和 ln(50257) 严格对齐。

## forward (训练 + 推理)

```python
def forward(self, input_ids, targets=None, *, verbose=False):
    B, T = input_ids.shape
    assert T <= self.cfg.block_size
    pos = torch.arange(T, device=input_ids.device)
    x = self.wte(input_ids) + self.wpe(pos)              # [B, T, D]
    for block in self.h:
        x, _ = block(x)                                   # 训练时不用 KV cache
    x = self.ln_f(x)
    logits = x @ self.wte.weight.T                       # tied [B, T, V]
    if targets is None:
        return logits
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                           targets.view(-1),
                           ignore_index=-1)
    return logits, loss
```

两种模式靠 `targets` 切换：
- `targets=None` → 推理，返回 logits
- `targets=y` → 训练（next-token prediction），返回 (logits, loss)

L3/L4 训练用第二种。L2 inference.py 用第一种。L6 server 用 `step()`（KV cache 版本，下面讲）。

## step (KV cache inference)

```python
@torch.no_grad()
def step(self, input_ids, kv_caches=None):
    B, T = input_ids.shape
    if kv_caches is None:
        kv_caches = [None] * len(self.h)
        pos_offset = 0
    else:
        pos_offset = kv_caches[0][0].size(2)  # past sequence length
    assert pos_offset + T <= self.cfg.block_size
    pos = torch.arange(pos_offset, pos_offset + T, device=input_ids.device)
    x = self.wte(input_ids) + self.wpe(pos)
    new_caches = []
    for block, kv in zip(self.h, kv_caches):
        x, new_kv = block(x, kv)
        new_caches.append(new_kv)
    x = self.ln_f(x)
    logits = x @ self.wte.weight.T
    return logits, new_caches
```

跟 `forward` 几乎一样，加了两件事：

1. **`pos_offset`** 从 cache 长度推断。第 0 步 cache 空，pos = arange(T)；第 1 步 cache 长度=T，pos = arange(T, T+1)；以此类推。
2. **每个 block 接收 kv_cache 并返回 new_kv**。

我写完这个方法后做了一个数值 sanity check：

```python
m = GPT(cfg).eval()
# 完整 forward
logits_full = m(x)
# 用 step 分两段（prefill + decode）
logits_pre, kvs = m.step(x[:, :T-1])
logits_dec, _ = m.step(x[:, T-1:T], kvs)

# 最后位置应该一致
diff = (logits_full[0, -1] - logits_dec[0, -1]).abs().max()
print(diff)  # < 1e-6 (浮点误差)
```

**严格等价 full forward**。这给我信心 KV cache 没写错。

## from_pretrained：加载 OpenAI gpt2 权重

```python
@classmethod
def from_pretrained(cls, name="gpt2"):
    _probe_and_set_hf_endpoint()  # CN 区自动 fallback
    from transformers import GPT2LMHeadModel
    cfg = GPTConfig()
    model = cls(cfg)
    hf = GPT2LMHeadModel.from_pretrained(name)
    sd_hf = hf.state_dict()
    sd = model.state_dict()
    conv1d_keys = {"attn.c_attn.weight", "attn.c_proj.weight",
                   "mlp.c_fc.weight",   "mlp.c_proj.weight"}
    for k_hf, v in sd_hf.items():
        k = k_hf.replace("transformer.", "")
        if k == "lm_head.weight":
            continue  # weight-tied to wte
        if any(k.endswith(s) for s in conv1d_keys):
            v = v.t()  # HF Conv1D → our Linear: transpose
        sd[k].copy_(v)
    return model
```

关键细节：**HF GPT-2 用 Conv1D，我们用 nn.Linear**。Conv1D 的 weight shape 是 `[in, out]`，Linear 是 `[out, in]`——四个 attn/mlp 矩阵需要 transpose。其他参数（embedding、LayerNorm）的 layout 一致，直接 copy。

`_probe_and_set_hf_endpoint()` 是网络受限地区的自动 mirror fallback——我在 [05-L6](05-L3-inference-server.md) 那篇讲了。

## 模型多大？

L3 训的小 model：

```
n_layer=4 n_head=4 n_embd=128 block_size=128
wte:     50,257 × 128 = 6,432,896
wpe:        128 × 128 =    16,384
4 blocks × (LN×2 + attn + MLP) ≈ 800,000
ln_f:                          256
total: 7,250,000 ≈ 7.24M params
```

OpenAI gpt2-124M（forward 一次的 FLOP 大概）：

```
n_layer=12 n_head=12 n_embd=768 block_size=1024
embedding: 50257 × 768 = 38.6M
12 blocks × (12 × 768² × 4 ≈ 2.8M per block FFN + 0.6M attn) ≈ 85M
ln_f, lm_head (tied): 0
total: ~124M params

forward FLOP for T=12 prompt:
  ~2 × 12 × 768² × 4 (FFN per layer) × 12 (layers) ≈ 0.7 GFLOP
  + attention ~0.05 GFLOP
  ≈ 0.75 GFLOP
```

跟 [03_model 实测](05-L3-inference-server.md)：5090 上 prefill 1.79 ms。理论 0.75 GFLOP / 1.79 ms = 419 GFLOPS。5090 fp32 峰值大概 100 TFLOPS——但带 Tensor Core 跑 TF32 大概有 200 TFLOPS。我们看到 0.4 TFLOPS 实际效率，说明 prefill 这种 small batch + small T 的场景**严重 launch-overhead bound**，没充分利用 GPU。

## 这一层的"最小"在哪里

- **没有 dropout**：训练用 dropout=0。简单 model + 小数据，不过拟合得太快没必要。
- **没有 GQA / MQA**：每个 head 有独立的 K/V。Llama2 / Llama3 用 Grouped Query Attention 多个 query heads 共享一组 K/V，省 KV cache memory。GPT-2 没有，我们也没有。
- **没有 RoPE**：用 learned position embedding。block_size 写死。换 RoPE 是 ~50 行的事，但会破坏 OpenAI gpt2 兼容。
- **没有 SwiGLU / GeGLU**：MLP 是经典 GELU。Llama 用 SwiGLU，效果好一点。
- **没有 LayerNorm 替代**：用经典 LN。RMSNorm（Llama）省 1 个 mean / 1 个 std 计算，5-10% 提速。

## 下一篇

L2 还有一半：**手写 BPE**。tokenizer 比 transformer 容易理解但更难写对——50257 个 vocab、unicode 边界、regex 预切词、merge 优先级。我们的 BPE 跟 tiktoken bit-for-bit 等价。

[L4b — 手写 BPE，bit-for-bit 等价 tiktoken →](07-L4-bpe.md)
