# L0 · 训练层

**一句话**：从 0.61 MB 的莎士比亚语料出发，在 CPU 上跑 6 分钟，得到一个 7.24M 参数的 GPT，能续写出 vaguely Shakespeare-y 的英文，并能被 L3 直接服务。

## 这一层为什么存在

之前 L4 实现了 GPT-2 的**架构**，但权重是 `from_pretrained("gpt2")` 从 HuggingFace 下载的——也就是说：**架构是我们自己的，但 model 的"知识"是 OpenAI 2019 年训出来的**。这和招牌"LLM from scratch"差一截。

L0 把缺失的那一段补完整：
- **数据**（`prepare.py`）：下载 Tiny Shakespeare（1.1 MB 纯文本），用 GPT-2 BPE 切成 token，存成 `train.bin / val.bin`。和 L3 用同一个 tokenizer，所以训完直接能塞回去服务。
- **训练**（`train.py`）：用**完全相同**的 L4 `GPT` 类（`04_transformer/model.py`），只是缩小到 4 层 4 头 128 维。AdamW + cosine LR + grad clip，~150 行。loss 从 10.82 降到 ~3.5。
- **采样**（`sample.py`）：加载 checkpoint，本地生成文本验证。

整条链路 + L0 闭环：**数据 → tokenize → 自己的架构 forward → loss → backward → optimizer → checkpoint → L3 加载 → L1/L2 流式调用**。

## 目录

```
00_train/
├── prepare.py        # 下数据 + tokenize，~50 行
├── train.py          # 训练循环，~140 行（用 L4 的 GPT 类）
├── sample.py         # 加载 ckpt 生成文本，~50 行
├── data/
│   ├── input.txt     # 莎士比亚原文 1.1 MB（download cache）
│   ├── train.bin     # 304K tokens uint16
│   └── val.bin       # 33K  tokens uint16
├── out/
│   └── ckpt.pt       # 训练产物 ~30 MB
└── README.md
```

## 怎么跑

```bash
cd 00_train

# 1. 下数据（~5 秒，~1 MB）
python prepare.py

# 2. 训练（M1 CPU 约 6 分钟，1000 步）
python train.py

# 3. 本地采样验证
python sample.py "ROMEO:"

# 4. 用 L3 服务训出来的模型
MODEL_PATH=$(pwd)/out/ckpt.pt python ../03_model/server.py
# 然后另开终端：
# cd ../02_agent && python agent.py "ROMEO: O Juliet"
```

## 训练时你会看到

```
model: 7.24M params total  (0.79M non-embedding)  device=cpu
config: n_layer=4 n_head=4 n_embd=128 block_size=128

training for 1000 steps, batch_size=32, block_size=128

step    0 | loss 10.815 | lr 3.00e-06 | 208,731 tok/s |   0.5s
step   25 | loss 10.343 | lr 7.80e-05 |  11,971 tok/s |   9.0s
step  100 | loss  7.843 | lr 3.00e-04 |  10,612 tok/s |  36.8s
step  200 | loss  ...   | lr ...      |  ...
...
```

第一行的 `loss 10.815` 就是 `ln(50257) ≈ 10.82`——**完全随机的 50257-way classifier 的理论 loss**。如果你看到的初值显著高于 11，说明权重初始化有问题（早期 L4 没做 init，初值是 81）。

最后 loss 应该降到 **3.0 - 3.8** 左右。注意 train loss 和 val loss 可能很接近（数据集小，欠拟合占主导），也可能 train ≪ val（轻微过拟合）。两种情况都正常。

## 训完的模型有多"聪明"

**不聪明**。诚实数字：

| | 训练 token | 参数量 | 知识来源 |
|---|---|---|---|
| 我们的 L0 model | 0.3M | 7M | 莎士比亚 33 个剧本 |
| OpenAI GPT-2 small | ~10B | 124M | 8M 网页（WebText） |
| Llama-3.1-8B | ~15T | 8B | 全互联网 + 代码 + 书籍 |

差着 4-7 个数量级。我们的 model 学会了**英文的统计结构**——大小写、标点、莎士比亚特有的"O"、"thou"、"ROMEO:"格式，能续写 vaguely 押韵的台词，但不会回答问题、不懂语法、更不可能"知道法国首都"。

这是**故意**的：L0 演示的是**循环本身**——data → forward → loss → backward → step。把循环跑通了，剩下的全是 scale：更多数据、更大模型、更多 GPU 时。

## 怎么训得更好

按 ROI 排序：

1. **加 step**：把 `MAX_STEPS` 从 1000 加到 5000-10000，loss 还能再降 1.5-2 个 nat。CPU 上 ~30 min。
2. **放大模型**：`n_layer=6, n_head=6, n_embd=384, block_size=256`（约 GPT-2 small 1/8 大小）。CPU 训不动了，需要 1×GPU。
3. **换数据**：莎士比亚太单一。换 OpenWebText 1% 子集（~400 MB token），同样的 model 学到的英文会"通用"得多。
4. **再大一档**：要做出像样的 base model，至少 GPT-2 small 规模（124M 参数 × 10B token），1×A100 上 ~4 天。这是 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 的复现路线。
5. **指令微调（SFT）**：base model 学了"续写"，但用户想要的是"问答"。在 Alpaca / OpenAssistant 数据上 SFT 几小时，base model 就会"听话"。这是 L0.5 的活儿，本 repo 暂未提供。

## 和其他层的接口

- **往下没有了**：L0 是源头。
- **往上（L4）**：完全复用 `04_transformer/model.py` 的 `GPT` 类。我们改的只有：把 `forward` 加了 `targets=None` 参数让它返回 loss，加了权重初始化（GPT-2 / nanoGPT 那一套 N(0, 0.02)）。**一行 model 类的代码都没复制**。
- **往上（L3）**：训完的 `ckpt.pt` 通过 `MODEL_PATH=...` env var 让 L3 加载。L3 内部把我们的小架构搬进 HF 的 `GPT2LMHeadModel` 壳子（保留 KV cache 等推理优化）。同一份权重，两个 engine。

## 这一层的"最小"在哪里

- **没有 distributed training**：单机单 process。多 GPU / DDP 是工程问题，不影响训练循环本身的样子。
- **没有混合精度**：CPU 上没意义。GPU 上加 `torch.amp.autocast` 即可，是几行的改动。
- **没有数据流水线**：直接 mmap `.bin` 文件做 random sampling。真实训练里 dataloader 要做 prefetch、shuffle、resharding，但当数据小到 60 MB 时这些都不重要。
- **没有断点续训**：跑完一次就结束。要恢复就是 `torch.load(ckpt) → load_state_dict → 重置 step 计数`，~10 行的事，省略让代码主线突出。
- **没有 wandb/tensorboard**：用 print。少一个依赖少一份心智负担。

## 为什么用莎士比亚（而不是更大的数据）

唯一原因：**5 分钟 CPU 训完**。这是教学项目，得让人在咖啡时间内看到 loss 下降。

如果你想看真正像样的 base model 是怎么训出来的（同样的代码、同样的循环、只是数据 100 倍大、GPU 8 张），强烈推荐 [nanoGPT](https://github.com/karpathy/nanoGPT)——本 repo 的训练循环本质上是它的简化版。
