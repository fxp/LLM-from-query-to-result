# L0.6 · Agent SFT

**一句话**：在 L0.5 SFT'd 124M 上继续 SFT，教它 ReAct 风格的"Plan → Act → Observe"循环。**33 秒**训完，L2 接 agent loop 后能真的调 `calc` 和 `lookup` 工具答问题。

## 这一层为什么存在

[L0.5](../00b_sft) 教的 model 知道"看到 Q: 就 emit A:"，但仅此而已。它不知道：
- 该用工具去算 `1234 + 5678`，而不是凭脑补
- 该用工具去查 `chemical symbol of gold`，而不是依赖 124M 那点 prior

L0.6 教的就是这件事：**model 不再尝试"自己知道"，而是 emit 一个 ACTION 让外部工具来给出 OBSERVATION**。这是从"chat completion"到"agent"的本质区别。

## 格式（ReAct）

```
Q: <question>
THOUGHT: <reasoning>
ACTION: tool_name(args)
OBSERVATION: <real tool output goes here>
THOUGHT: ...                ← 可多步
ACTION: ...
OBSERVATION: ...
ANSWER: <final><|endoftext|>
```

## Loss masking — 关键细节

每条 training example 把 token 标成 "学" 或 "mask"：

```
Q: ... \n                               MASK   (用户输入，model 接收)
THOUGHT: ... \n                         LEARN  (model 自己生成)
ACTION: ... \n                          LEARN  (model 自己生成)
OBSERVATION:                            LEARN  ← 关键！这是 model 的"我要工具结果"信号
 <observation> \n                       MASK   (tool 给的内容)
ANSWER: <final> <EOT>                   LEARN
```

**为什么 `OBSERVATION:` 这 prefix 必须 LEARN**：因为这是 model 主动告诉 agent loop "我等工具结果" 的 stop 信号。如果 mask 了，模型不会 emit 它，会一直循环 emit 更多 ACTION。我第一版就这么错了——见 [`blog/10-L0.6-agent.md`](../blog/10-L0.6-agent.md)。

## 工具

`tools.py` 给两个：

```python
calc("23 + 47")         # → "70"
calc("3 * (4 + 5)")     # → "27"
lookup("capital of France")  # → "Paris"
lookup("author of Hamlet")   # → "Shakespeare"
```

`calc` 是 sandboxed `eval`（regex 限定为 `[\d+\-*/().\s]+`，没 builtins）。`lookup` 查 `kb.json`（105 条事实，从 [00b_sft/data.json](../00b_sft/data.json) 整理而来）。

## 数据生成（程序合成）

`build_data.py` 程序生成 258 条 traces：

- **lookup × 173 条**：每条 KB 事实生成 1-2 个问句变体
- **calc × 85 条**：随机 30 加法 + 15 减法 + 20 乘法 + 10 除法 + 10 混合表达

```bash
cd 00c_agent_sft
python build_data.py    # 生成 kb.json + data.json
python tools.py         # self-test：calc/lookup 10/10 ✓
```

> 这是真正的 SFT 数据生成方式——production 的 instruct dataset（Alpaca、OpenOrca）也都是合成或半合成的。手写数据的工业实践早就过时了。

## 训练

需要 L0.5 path-B ckpt（`../00b_sft/out/sft_from_gpt2.pt`）作为起点。

```bash
cd 00c_agent_sft && python train.py
```

实测 RTX 4080 SUPER：

```
loaded base: 124.44M params
agent traces: 258  avg len=48.2 tokens

Agent SFT for 20 epochs × 33 steps = 660 updates  batch_size=8 lr=5e-5
epoch  0/20 | avg loss 2.293 |  2.1s
epoch  4/20 | avg loss 0.000 |  8.8s   ← 几乎瞬间收敛
epoch 19/20 | avg loss 0.000 | 33.5s

saved -> out/agent.pt (498 MB)
```

为什么收敛快：258 个 traces × avg 48 tokens × 学 ~50% = ~6K target tokens，对 124M 参数来说太小，瞬间记住。这有点像 L0.5 path B：base 已经强，SFT 教格式不教知识。

## 跑 agent loop

L3 加载 agent.pt，L2 设 `AGENT_MODE=1` 启用 ReAct loop：

```bash
# 终端 1
MODEL_PATH=$(pwd)/out/agent.pt python ../03_model/server.py

# 终端 2
cd ../02_agent
AGENT_MODE=1 python agent.py "What is the capital of France?"
```

输出：

```
[L2] AGENT_MODE — running ReAct loop on http://localhost:9000/generate

💭 I should look up capital of France.
🔧 lookup(capital of France)
   ↳ Paris
Paris.

[done]
```

`💭` THOUGHT、`🔧` ACTION、`↳` OBSERVATION（来自 tool）、最后 ANSWER。

## 实测：能 / 不能 答什么

| Query | 输出 | 评 |
|---|---|---|
| `capital of France?` | `Paris.` ✓ | KB 命中 |
| `Who wrote Hamlet?` | `Shakespeare.` ✓ | KB 命中 |
| `chemical symbol of gold?` | `Au.` ✓ | KB 命中 |
| `What is 8 times 7?` | `56.` ✓ | calc 真算 |
| **`What is 1234 plus 5678?`** | **`6912.`** ✓ | **calc 算超训练范围的数** |
| `What is 10 divided by 3?` | `3.3333...` ✓ | float division |
| `capital of Mongolia?` | `not found.` | KB 没有 → 老实复述 not found |
| `Mona Lisa?` | "painter of **the** Mona Lisa" → `not found` | model 学到模式但 key normalize 不对 |
| `How are you today?` | hallucinate `lookup(author of this paper)` | 完全 out-of-distribution |

**核心成果**：`1234 + 5678 → 6912` 是 agent 的胜利——这个具体数字**绝不在 SFT 训练数据里**，model 把 "23 + 47" 这类模式泛化到了任意数字，靠的是 calc 工具真算出来。这就是 agent 比 chat 强的本质——**外部工具扩展了模型能力的边界**。

## 这一层的"最小"在哪里

- **2 个工具**：真实 agent（Claude Code、Cursor）有十几个工具（read_file、write_file、grep、shell、web_fetch...）。这里 2 个够演示循环结构。
- **Single-step**：258 traces 都是 1 步 ACTION。多步 agent 需要更复杂的 SFT 数据（"先 lookup A，再 calc 用结果"），是另一个练习。
- **没有错误恢复**：如果 model emit 了 malformed action（很少见但会），agent 会报错退出。
- **没有思维链 reasoning**：THOUGHT 字段是浅的"我应该 lookup X"，不是真正的多步 reasoning。124M 这个 size 也展不开。
- **没有工具调用 reflection**：tool 报错了 model 不会自我纠正。
- **没有终止决策**：model 总是 emit ANSWER，不会说"我不知道"。

## 关键 takeaway

1. **Agent 不靠 model "更聪明"，靠工具扩展**。124M 不会算 1234+5678，但有了 calc 工具就能算。
2. **格式契约 + loss masking 是关键**。SFT 让 model 学 emit `THOUGHT/ACTION/OBSERVATION:`（结构契约），mask `<observation_content>`（不让 model 自己造工具结果），mask `Q:`（用户给的）。
3. **30 秒 SFT 就能教格式**。124M base 见过 ReAct 风格的英文（Alpaca、ShareGPT 等数据集存在于 WebText 后期）——SFT 只是 "调出来" 这个行为，不是从头教。
