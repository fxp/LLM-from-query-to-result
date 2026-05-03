# L0.5 · SFT (Supervised Fine-Tuning)

**一句话**：拿 L0 训出的 base model（学会"续写英文"），在 ~60 条手写 Q/A 上微调 28 秒，让它学会"看到问题就回答"的 instruction 格式。

## 这一层为什么存在

L0 的 base model 学会的是**续写 next token**。给它 "What is the capital of France?" 它会接着写莎士比亚式的台词——因为它见过的所有训练 token 都来自莎翁的剧本。它不知道"问"和"答"是两件事。

把 base 变 instruct 的标准套路就是 **SFT**：

```
对每条 (question, answer):
   prompt = "Q: <question>\nA:"
   target = " <answer><|endoftext|>"
   loss 只在 target 部分计算（prompt 部分用 ignore_index=-1 mask 掉）
```

继续训几个 epoch，模型就学到三件事：
1. 看到 `Q: ... A:` 就开始回答
2. 答完 emit `<|endoftext|>` 停止
3. 对见过的 fact 直接 parrot 答案

ChatGPT、Claude 也都是这个套路开始的（再加 RLHF / DPO）。

## 目录

```
00b_sft/
├── data.json     # 63 条手写 Q/A（地理、算术、化学、文学）
├── train.py      # SFT 训练脚本，~140 行
├── out/sft.pt    # 训练产物 ~30 MB
└── README.md
```

## 怎么跑

需要 L0 base ckpt 先跑出来：

```bash
# 一次性
cd 00_train && python prepare.py && python train.py
cd ../00b_sft && python train.py
```

输出长这样（实测 28 秒，M1 CPU，50 epochs × 63 examples）：

```
loaded base: 7.24M params  pre-SFT loss={'train': 4.55, 'val': 5.05}
SFT data: 63 Q/A pairs

--- BEFORE SFT (base model) ---
  Q: What is the capital of France?
  A: KING RICHARD III: I'll be a man, I have not, I have not...
  Q: What is 2 plus 2?
  A: KING RICHARD III: I'll be a man...

SFT for 50 epochs, batch_size=8, lr=1e-4
epoch   0 | avg loss  9.06
epoch  10 | avg loss  5.67
epoch  25 | avg loss  3.79
epoch  49 | avg loss  1.69

--- AFTER SFT ---
  Q: What is the capital of France?
  A: Paris.
  Q: What is 2 plus 2?
  A:
  Q: How many continents are there?
  A:
```

模型从"接龙莎翁"变成"短答 + EOT"。`Paris.` 是它自己写出来的——之前根本不知道法国首都是哪。

## 跑通整条链路

把 SFT 后的 ckpt 喂给 L3：

```bash
MODEL_PATH=$(pwd)/out/sft.pt python ../03_model/server.py
# 另一个终端：
cd ../02_agent && L2_TEMPERATURE=0 python agent.py "What is the capital of France?"
# → " Paris.<|endoftext|>"
```

L2 的 prompt template (`Q: <query>\nA:`) 与本目录的 SFT 格式对齐——这样 SFT 后的 model 在 L2/L1 里能直接听懂用户问句。

## 实测：哪些 SFT 后能答、哪些不能

| Query | SFT 前 (base) | SFT 后 |
|---|---|---|
| `What is the capital of France?` | 莎翁乱说 | **`Paris.`** ✅ |
| `Who wrote Hamlet?` | 莎翁乱说 | `William` (截断的 William Shakespeare) |
| `What is the capital of Japan?` | 莎翁乱说 | (空，emit EOT) |
| `What is 2 plus 2?` | 莎翁乱说 | (空) |
| `How many continents are there?` | 莎翁乱说 | (空) |

为什么有的能答有的不能：
- **France→Paris**：训练数据里有，且 "France" 在莎士比亚剧本里也常见，base model 对这个词有 prior
- **Hamlet→William (Shakespeare)**：在数据里。"William" 也在莎翁剧本里，但 "Shakespeare" 不在 base 训练语料里，所以 SFT 后 model 倾向于 emit EOT 而不是补完
- **Japan/2+2/continents**：在训练数据里，但 base model 对这些词几乎没见过——SFT 不能凭空创造表征

**结论**：SFT 转换的是行为模式（接龙→问答），不是知识。要让 SFT 真有"知识泛化"，需要 base model 已经在大语料上预训过（GPT-2 small 124M 在 WebText 8M 网页上）。

## 这一层的"最小"在哪里

- **数据手写**：63 条 `data.json`，没用任何 dataset 库（datasets, alpaca-eval 等）。要扩展就加条目。
- **没有 RLHF / DPO**：只 SFT 一步。Anthropic / OpenAI 的真正 instruct 模型还会用偏好数据再训一轮，但那是 L0.6 的活儿。
- **Loss masking 只盖 prompt**：标准做法。如果不 mask，模型也会被惩罚去预测 question token，行为容易跑偏。
- **没有 LoRA**：全参数 SFT。7M model 全微调比 LoRA 还便宜。生产里 7B+ 才需要 LoRA 省显存。
- **没有 packing**：每个 batch 一条独立 example，padding 到最长。真训练里会把多条 examples pack 进同一序列省 padding，但这里数据小到无所谓。

## 想做得更像 ChatGPT 一点

按 ROI：

1. **换更大 base**：用 OpenAI 的 gpt2-small (124M, `from_pretrained("gpt2")`) 当起点。它在 8 GB WebText 上预训过，SFT 之后对世界常识有真的泛化。
   ```python
   # 改 train.py 的 BASE_CKPT 加载逻辑：
   model = GPT.from_pretrained("gpt2")  # 不从 ckpt，直接加载 OpenAI 权重
   ```
2. **加更多 SFT 数据**：Alpaca (52K) / OpenOrca / ShareGPT。手写不现实了，下数据集。
3. **Instruction tuning + chat template**：用 `<|im_start|>user\n...\n<|im_end|>` 这类格式，多轮对话。
4. **DPO/RLHF**：拿一组 preference pairs，让模型学"哪种答案更受欢迎"。

但这些都和"循环本身"无关——加上去只是 scale 和 data eng，不改训练代码的本质。SFT 这一步的核心算法在 `train.py` 已经齐了：mask + cross-entropy + AdamW。
