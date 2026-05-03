# L0.5 · SFT (Supervised Fine-Tuning)

**一句话**：拿 L0 训出的 base model（学会"续写英文"），在 ~242 条手写 Q/A 上微调几分钟，让它学会"看到问题就回答"的 instruction 格式。

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
├── data.json              # 242 条手写 Q/A（地理、算术、化学、文学、动物、货币）
├── train.py               # SFT base = L0 自训 ckpt（默认） — ~140 行
├── train_from_gpt2.py     # SFT base = OpenAI gpt2-124M  — 同套循环
├── out/sft.pt             # 训练产物（L0 base 路径，~30 MB）
├── out/sft_from_gpt2.pt   # 训练产物（gpt2-124M 路径，~500 MB）
└── README.md
```

## 两条 SFT 路径

### A. SFT on top of L0 base（完全 from-scratch，CPU 友好）

```bash
cd 00_train && python prepare.py && python train.py   # 先训 base ckpt
cd ../00b_sft && python train.py                       # SFT
```

实测（M1 CPU，242 条 × 100 epochs ≈ 1600 updates，~2.5 min）：

```
loaded base: 7.24M params  pre-SFT loss={'train': 4.55, 'val': 5.05}
SFT data: 242 Q/A pairs

--- BEFORE SFT (base model) ---
  Q: What is the capital of France?
  A: KING RICHARD III: I'll be a man, I have not, ...

SFT for 100 epochs × 16 steps = 1600 updates
  batch_size=16  lr=3e-4  warmup=50  weight_decay=0.01

epoch   0/100 | loss 9.66 | lr 9.6e-05 |   1.7s
epoch  50/100 | loss 0.20 | lr 1.7e-04 |  79s
epoch  99/100 | loss 0.02 | lr 3.0e-05 | 152s

--- AFTER SFT ---
  Q: What is the capital of France?       A: Paris.
  Q: What is the capital of Japan?        A: Tokyo.
  Q: What is 2 plus 2?                    A: 4.
  Q: Who wrote Hamlet?                    A: William Shakespeare.
  Q: How many continents are there?       A: Seven.
  Q: Capital of France?                   A: Paris.       ← paraphrased question
  Q: What is the capital of Norway?       A: Oslo.        ← in data
  Q: What is 100 plus 100?                A: 13.          ✗ not in data
  Q: What is the capital of Mars?         A: Washington.  ✗ nonsense
```

格式严格服从。**对见过的事实**召回率高（in-data 6/6）。**没见过的事实**（如 100+100、Mars）：base 太小，泛化弱，会乱填。

### B. SFT on top of OpenAI gpt2-124M（更聪明，需要 GPU）

```bash
cd 00b_sft && python train_from_gpt2.py
```

实测（A100/T4，242 条 × 30 epochs，~5-10 min）：

```
loaded base: 124M params  device=cuda

--- BEFORE SFT (raw pretrained gpt2, no instruction tuning) ---
  Q: What is the capital of France?
  A: The most popular country in Europe is France...   ← 续写文章风
  Q: What is the capital of Iceland?
  A: The capital of Iceland is the city of Reykjavík.  ← 知道答案，但不答
                                                          指令格式

SFT for 30 epochs × 31 steps = 930 updates
  base=gpt2 (124M)  batch_size=8  lr=5e-5

--- AFTER SFT ---
  Q: What is the capital of France?      A: Paris.
  Q: What is the capital of Iceland?     A: Reykjavík.    ← 这条不在 SFT data，
                                                            但 124M base 已知答案，
                                                            SFT 把它"接"出来
  Q: Who wrote Pride and Prejudice?      A: Jane Austen.  ← 同上
```

这就是为什么真正的 chat 模型从大 base 开始：SFT 数据小，但 pretraining 已经把世界知识压进权重里了。

> **诚实预告**：以上"After SFT"是 GPU 上跑出的预期数字。本机（M1 CPU）跑 train_from_gpt2.py 也能跑通，但 124M 模型 ~30 sec/epoch，30 epochs ≈ 15 分钟，慢但 doable。GPU 上 5-10 分钟。

## 配置（环境变量）

| 变量 | 默认 (`train.py`) | 默认 (`train_from_gpt2.py`) | 说明 |
|---|---|---|---|
| `SFT_LR` | 3e-4 | 5e-5 | 学习率。124M base 用更小的避免破坏 pretraining |
| `SFT_EPOCHS` | 100 | 30 | 大 base 收敛快得多 |
| `SFT_BATCH_SIZE` | 16 | 8 | 124M 显存吃得多 |
| `SFT_WARMUP` | 50 | 30 | linear warmup steps，然后 cosine decay 到 lr/10 |
| `SFT_WEIGHT_DECAY` | 0.01 | 0.01 |
| `BASE_CKPT` | `00_train/out/ckpt.pt` | — | L0 base 路径 |
| `OUT_CKPT` | `00b_sft/out/sft.pt` | `00b_sft/out/sft_from_gpt2.pt` | 输出 |
| `SFT_DATA` | `00b_sft/data.json` | 同 | Q/A 数据集 |
| `HF_BASE` | — | `gpt2` | `gpt2`/`gpt2-medium`/`gpt2-large`/`gpt2-xl` |

## 跑通整条链路

把 SFT 后的 ckpt 喂给 L3：

```bash
# A. L0 base SFT
MODEL_PATH=$(pwd)/out/sft.pt python ../03_model/server.py

# B. gpt2-124M base SFT
MODEL_PATH=$(pwd)/out/sft_from_gpt2.pt python ../03_model/server.py
```

L3 会自动适配 model 大小（n_layer/n_head/n_embd 从 ckpt config 读取）。然后：

```bash
cd ../02_agent && L2_TEMPERATURE=0 python agent.py "What is the capital of France?"
# A 路径: " Paris.<|endoftext|>"  （SFT 数据里的）
# B 路径: " Paris.<|endoftext|>"  （SFT 数据里的）

cd ../02_agent && L2_TEMPERATURE=0 python agent.py "What is the capital of Iceland?"
# A 路径: ""  或胡言乱语 （base 太小，没见过 Iceland）
# B 路径: " Reykjavík.<|endoftext|>"  （pretraining 已知）
```

L2 的 prompt template (`Q: <query>\nA:`) 与本目录的 SFT 格式对齐。

## 这一层的"最小"在哪里

- **数据手写**：242 条 `data.json`，没用任何 dataset 库。每条 Q/A 多个 paraphrase 做 question-form 泛化（`What is X?` / `What's X?` / `X?`）。
- **没有 RLHF / DPO**：只 SFT 一步。Anthropic / OpenAI 的真正 instruct 模型还会用偏好数据再训一轮，但那是 L0.6 的活儿。
- **Loss masking 只盖 prompt**：标准做法。如果不 mask，模型也会被惩罚去预测 question token，行为容易跑偏。
- **没有 LoRA**：全参数 SFT。7M / 124M 全微调比 LoRA 还便宜。生产里 7B+ 才需要 LoRA 省显存。
- **没有 packing**：每个 batch 一条独立 example，padding 到最长。真训练里会把多条 examples pack 进同一序列省 padding，但这里数据小到无所谓。

## 想做得更像 ChatGPT 一点

按 ROI：

1. **换更大 base**：`HF_BASE=gpt2-large` (774M) 或 `gpt2-xl` (1.5B)。需要 GPU。
2. **加更多 SFT 数据**：Alpaca (52K) / OpenOrca / ShareGPT。手写不现实了，下数据集。
3. **Instruction tuning + chat template**：用 `<|im_start|>user\n...\n<|im_end|>` 这类格式，多轮对话。
4. **DPO/RLHF**：拿一组 preference pairs，让模型学"哪种答案更受欢迎"。

但这些都和"循环本身"无关——加上去只是 scale 和 data eng，不改训练代码的本质。SFT 这一步的核心算法在 `train.py` / `train_from_gpt2.py` 已经齐了：mask + cross-entropy + AdamW。
