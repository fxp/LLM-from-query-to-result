# L4 · Transformer 层

**一句话**：把 L3 送来的一串 token id，经过 embed → N 个 transformer block → 输出每个位置上"下一个 token 是谁"的概率分布。

## 为什么是这样

L3 把"prompt 文本"变成了 `[15496, 2267, 287, 257, 640]` 这样的 token id 序列。
L4 干的事只有一件：**给这串 id，算出下一个 id 的概率**。

这件事靠一个叫 Transformer 的神经网络做。结构是：

```
  token ids  (shape: [B, T])
       │
       ▼  token embedding + position embedding
  hidden states  [B, T, D]
       │
       ▼  ── Block 1 ──┐
       │               │
       │    ┌──────────▼────────────┐
       │    │  LayerNorm            │
       │    │  Multi-Head Attention │──▶ (每个位置看前面所有位置)
       │    │  + residual           │
       │    │  LayerNorm            │
       │    │  Feed-Forward (MLP)   │
       │    │  + residual           │
       │    └──────────┬────────────┘
       │               │
       ▼               ▼
  hidden states  [B, T, D]    ← 这就是 Block 的输出，shape 没变
       │
       ▼  ...再过 N-1 个 Block...
       │
       ▼  LayerNorm
  hidden states  [B, T, D]
       │
       ▼  乘以 embedding 矩阵的转置（"tie weights"）
  logits  [B, T, V]   ← V = 词表大小（GPT-2 是 50257）
```

**Attention** 做的事：每个位置的 token 生成一个 query / key / value 向量，然后这个位置的新状态 = 所有位置的 value 加权求和，权重 = softmax(query · 别人的 key)。简单说就是"每个 token 决定自己要从别人那吸收多少信息"。

**FFN** 做的事：每个位置独立地过一个两层 MLP（先升维到 4D 再降回 D）。它是模型"记住事实"的地方——近几年的研究表明知识主要存在 FFN 权重里。

**LayerNorm + residual**：让梯度能传到深层，让训练稳定。

## 目录

```
04_transformer/
├── model.py        # GPT 结构，~200 行，可加载 HF 权重
├── tokenizer.py    # 包装 tiktoken/HF 的 GPT-2 BPE
├── inference.py    # 加载权重，单次 forward，打印每层 shape 和耗时
└── README.md
```

## 怎么跑

```bash
cd 04_transformer
python inference.py "Hello, I am"
```

输出类似：
```
loaded gpt2: 124.44M params
tokens: [15496, 11, 314, 716]  (4 tokens)

forward pass, one step:
  x = embed(ids) + pos(ids)          shape=(1, 4, 768)
  block 0  attn out=(1, 4, 768)  ffn out=(1, 4, 768)  0.8 ms
  block 1  attn out=(1, 4, 768)  ffn out=(1, 4, 768)  0.7 ms
  ...
  block 11 attn out=(1, 4, 768)  ffn out=(1, 4, 768)  0.7 ms
  ln_f                                shape=(1, 4, 768)
  logits = x @ wte.T                 shape=(1, 4, 50257)
  argmax at last pos -> 257  (" a")

predicted next token: " a"
```

`" a"` 是 GPT-2 small 给 "Hello, I am" 接的下一个词——你跑一次试试。

## 和其他层的接口

- **往上（L3）**：`model(input_ids, past_key_values=...) -> logits, new_kv`。和 HuggingFace 的 API 形状一致，所以这份手写版可以无缝替换掉 L3 里的 `GPT2LMHeadModel`（但会更慢，因为没做任何内核优化——这是 L5 的活）。
- **往下（L5）**：每个 `LayerNorm` 后面有一个 `nn.Linear`，那**就是一个 matmul**。`attention` 里的 `Q @ K.T`、`attn @ V` 也是 matmul。一次 forward 里 95%+ 的时间花在这些 matmul 上，因此 L5 GPU 优化的目标就是它们。

## 这一层的"最小"在哪里

- **300 行以内**：nanoGPT 已经证明了这件事能做到，我们甚至再少一点（不含训练 loop）。
- **能加载 GPT-2 权重**：这样你改一行代码就能看到是否影响了输出——比自己训一个小模型直观得多。
- **没做量化、没做 LoRA、没做 MoE**：这些是正交的扩展，不影响"transformer 本身是什么"的答案。

## 推荐的读法

1. 先读 `model.py` 的 `forward()` 主干。
2. 再读 `CausalSelfAttention`——注意 causal mask 是怎么做出来的（一个三角矩阵）。
3. 最后读 `_load_gpt2_weights`，这里会看到 HuggingFace 的权重名是怎么映射到我们这个结构里的。如果名字能对上，你的实现大概率就是对的。
