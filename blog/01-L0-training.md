# 01 · L3：从莎士比亚训出一个 GPT

> [← 序章](00-overview.md) ｜ 代码：[`00_train/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/00_train)

每一个 ChatGPT 都从这里开始：一个完全随机权重的网络，一堆文本，一个训练循环。我从这一层开始讲——因为大多数"从零写 GPT"教程在这里就开始有意思了，但很多人没真正跑过一遍训练，就跳到了模型结构。

我想做到的是：从你按下 `python train.py` 开始，到屏幕上 loss 一行行往下掉，再到 6 分钟（CPU）/12 秒（5090）后保存 ckpt，**每个数字都解释清楚**。

## 数据

[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)：莎士比亚 33 个剧本拼起来的纯文本文件，**1.1 MB**。

我把它直接 commit 进 repo 了（`00_train/data/input.txt`）——这是公共领域、文件很小、避免任何网络依赖。

为什么用莎士比亚？两个原因：
1. **小**：CPU 几分钟训完。教学项目要让人在咖啡时间内看到 loss 下降。
2. **风格鲜明**：`ROMEO:`、`thee`、`thou`、`'tis` 这些词高频出现，模型学到东西后续写出来一眼能看出"它学了点什么"。

跑 `python prepare.py`：

```
using bundled: 00_train/data/input.txt
corpus length: 1,115,394 chars
tokenized:     338,025 BPE tokens  (vocab=50257)
unique tokens used: 11,706

wrote train.bin:  304,222 tokens  (0.61 MB)
wrote val.bin:     33,803 tokens  (0.07 MB)
```

几个观察：
- **GPT-2 BPE 把 1.1 MB 文本切成 33.8 万 token**，平均一个 token 对应 3.3 个字符。
- **整个语料只用了 11,706 个 unique token**——50,257 个词表里的 23%。剩下的 token（中文字符、罕见英文词等）权重完全练不到，只占空间不长本事。
- 90/10 分 train/val，存为 `uint16` 二进制（每个 token 2 字节）。这是 [nanoGPT](https://github.com/karpathy/nanoGPT) 的标准格式——训练时直接 `np.memmap` 切片采样，零 Python 循环。

## 模型

不复用别的，**直接拿 L2 那个 330 行手写的 GPT 类**（`04_transformer/model.py`），只是 config 缩小：

```python
CFG = GPTConfig(
    vocab_size=50257,   # GPT-2 BPE
    n_layer=4,          # 默认 12
    n_head=4,           # 默认 12
    n_embd=128,         # 默认 768
    block_size=128,     # 默认 1024
    dropout=0.0,
)
```

**总参数 7.24M**，其中 6.4M 是 token embedding（`50257 × 128`），剩下 0.79M 是 transformer block。这意味着：embedding 占了 88% 的参数——当语料只用 11,706 个 unique token 时，绝大多数 embedding 行根本没被反向传播过，是死参数。

这也意味着：**真正"学习"发生在那 0.79M 的非 embedding 参数里**。即便如此，loss 从 10.815 降到 4.5，是这 80 万参数的功劳。

## 训练循环

整段 `train.py` 不到 180 行。核心循环：

```python
for step in range(MAX_STEPS):
    lr = lr_at(step)                       # warmup + cosine decay
    for g in optim.param_groups:
        g["lr"] = lr

    x, y = get_batch("train")              # [B=32, T=128] from train.bin
    _, loss = model(x, targets=y)          # forward + cross-entropy
    optim.zero_grad(set_to_none=True)
    loss.backward()                        # backward
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()                           # AdamW
```

四件事：

1. **`get_batch`**：从 `train.bin` mmap 切随机窗口，组成 batch。每条样本是连续 128 个 token，target 是同样 128 个 token 但偏移 1（next-token prediction）。
2. **forward + loss**：`model(x, targets=y)` 调我们的 GPT 类。当 `targets` 给了，它返回 `(logits, loss)`，loss 是 cross-entropy（按 vocab 维度展开）。
3. **backward**：PyTorch 自动算所有梯度。
4. **gradient clip + AdamW step**：grad clip 防止偶尔的大梯度爆炸；AdamW 更新参数。

LR schedule：

```python
def lr_at(step):
    if step < WARMUP:                        # 100 步线性 warmup
        return LR_MAX * (step + 1) / WARMUP
    progress = (step - WARMUP) / (MAX_STEPS - WARMUP)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))  # cosine
```

3e-4 是 transformer 训练的"魔法值"（[Karpathy's tweet](https://twitter.com/karpathy/status/801621764144144384)）。warmup + cosine 是 GPT-2/3 论文里都用的 schedule。

## Loss 从哪里来，到哪里去

跑 `python train.py`，第一行：

```
step    0 | loss 10.815 | lr 3.00e-06 | ... |   0.5s
```

**为什么是 10.815？**

随机初始化时，模型对 50257 个词表项没任何 prior，每个 token 的概率均匀分布，softmax 输出每个位置都是 1/50257。Cross-entropy loss 就是：

$$
\text{loss} = -\log(1/V) = \log(V) = \log(50257) \approx 10.825
$$

我们看到的 10.815 跟这个理论值差 0.01，说明 init 做对了——权重均匀分布，没有偏向任何 token。

> ⚠️ **早期版本我没做 weight init，初值是 80**。原因是 PyTorch 默认 `nn.Linear` 用 Kaiming uniform，对 transformer 来说初始 logits 方差太大，softmax 严重偏向某些 token，cross-entropy 巨大。修复是加上 GPT-2 标准的 `N(0, 0.02)` 初始化，并把 residual 出口的 `c_proj` 缩 `1/sqrt(2N)`（防止 residual stream 方差随深度爆炸）。这是 [nanoGPT 也是 GPT-2 论文的标准做法](https://github.com/karpathy/nanoGPT/blob/master/model.py)。

下降轨迹（5090, 1000 步，12 秒）：

```
step    0 | loss 10.815  ← random init = ln(50257)
step  100 | loss  7.84   ← model 发现 "the/and/of" 比生僻词多
step  500 | loss  5.00   ← 学到 char-level 大小写、标点规律
step 1000 | loss  4.55   ← train, val 5.05
```

train/val 1500 处差距 0.5——轻微过拟合，正常的小数据训练表现。

## 训完的模型在说什么

`python sample.py 'ROMEO:'`：

```
ROMEO:
But touch the pleasure good looks else to the people, be much with grave!

BUCKINGHAM:
Why, I know my love aupon I must here.

She Lord:
KING RICHARD deliver,
JULIUS:
My lord, for I be such a heat to meet theORIZong in thy dark:
I did not to the day.
```

**学到了什么**：
- 莎士比亚剧本的格式（`CHARACTER:` + 换行 + 台词）✓
- 角色名（ROMEO、BUCKINGHAM、KING RICHARD、JULIUS——后两个是真实剧本里的）✓
- 莎翁词汇（thee/thou 风格，`'tis`，`be much with grave!`）✓
- 偶尔的押韵节奏

**没学到**：
- 语义连贯性
- 实际的莎翁台词（"But touch the pleasure good looks" 不是任何剧本里的）
- 任何"知识"

这是 **7M 参数 + 0.34M tokens 训练 12 秒** 的诚实样子：学到结构，没学到语义。

要让它学到更多？按 ROI 排序：
1. **加 step**：1000 → 10000 步，loss 还能再降 1-2 nat。CPU 5 分钟变 50 分钟。
2. **放大模型**：6 层、384 维、block_size=256 ≈ GPT-2 small 的 1/8。CPU 训不动了，需要 GPU。
3. **换数据**：莎翁太单一。OpenWebText 的 1% 子集（~400 MB token），同样的 model 学到的英文会通用得多。
4. **再大一档**：要做 GPT-2 small 规模（124M × 10B token），1×A100 上 ~4 天。这是 [nanoGPT](https://github.com/karpathy/nanoGPT) 的复现路线。

## 这一层的"最小"在哪里

- **没有 distributed training**：单机单 process。多 GPU / DDP 是工程问题，不影响"训练循环本身"的样子。
- **没有混合精度**：CPU 上没意义。GPU 上加 `torch.amp.autocast` 即可。
- **没有 dataloader 管线**：直接 mmap `.bin` 做 random sampling。生产里 dataloader 要 prefetch、shuffle、resharding，但当数据是 60 MB 时这些都不重要。
- **没有断点续训**：跑完就结束。要恢复就 `torch.load(ckpt) → load_state_dict → 重置 step 计数`，~10 行的事。
- **没有 wandb/tensorboard**：用 print。少一个依赖少一份心智负担。

## 实测数字

| 硬件 | 1000 步训练耗时 | 吞吐量 |
|---|---|---|
| RTX 5090 (Blackwell sm_120) | **12.2 秒** | ~400K tok/s |
| RTX 4080 SUPER (Ada sm_89) | 27.6 秒 | ~145K tok/s |
| Apple M1 (CPU) | ~6 分钟 | ~10K tok/s |

5090 上 30× CPU 加速。注意瓶颈：这个 model 太小（7M 参数），实际训练时 GPU 大半时间在等内核 launch / Python 调度，而不是真在算——所以更大的 model 5090 vs 4080S 差距会更显著。

## 下一篇

`Q: capital of France?` 输入这个 model，它会给你一段莎翁台词。**怎么把它从"接龙莎翁"变成"答 'Paris.'"？** 答案是 SFT，下一篇讲。

[L4 — 24 秒把"接龙莎翁"变成"答 Paris" →](02-L0.5-sft.md)
