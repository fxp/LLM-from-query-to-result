# 07 · L4b：手写 BPE，bit-for-bit 等价 tiktoken

> [← L4a transformer](06-L4-transformer.md) ｜ 代码：[`04_transformer/bpe.py`](https://github.com/fxp/LLM-from-query-to-result/blob/main/04_transformer/bpe.py) ｜ [下一篇 →](08-L5-gpu.md)

Tokenizer 在 LLM 教学里通常被一笔带过——"用 tiktoken 就行"。但 tokenizer 是 model 跟世界的边界，**它做错了所有 model 都白做**。这一篇讲我怎么手写 BPE，并验证它跟 OpenAI 官方实现 bit-for-bit 等价。

230 行 Python，covers:
- 字节 → unicode 映射
- GPT-2 的 regex 预切词
- BPE merge 算法
- encode / decode / roundtrip

## Tokenizer 是什么

LLM 的视角里，文本不是字符串，是**整数序列**。`"Hello, world!"` 经过 GPT-2 BPE 变成 `[15496, 11, 995, 0]`——4 个整数，每个是 vocab 表里的 token id。

这个映射要满足：
1. **可逆**：`decode(encode(s)) == s`
2. **字节级 safe**：能 encode 任意 UTF-8 字符串（中文、日文、emoji）
3. **高效**：常见词组合成一个 token，罕见词拆成多个

GPT-2 的 vocab 是 **50,257**：
- 50,000 BPE merge 合成的 token
- 256 字节 fallback（保证任意 byte 串都能 encode）
- 1 个 `<|endoftext|>` (id 50256)

## 算法概览

GPT-2 BPE 的 encode 分 4 步：

```
"Hello, world!"
   │
   ▼ regex 预切词 (pre-tokenize)
['Hello', ',', ' world', '!']
   │
   ▼ 每个 chunk 独立处理：
   ▼ chunk 的 UTF-8 字节 → 用 byte→unicode 表映射
'Hello' → 'Hello'           (ASCII 直接对应)
' world' → 'Ġworld'         (空格 0x20 → 'Ġ' = U+0120)
   │
   ▼ BPE merge: 反复找最高优先级的相邻 token 对，合并
'H','e','l','l','o' → 'Hel','l','o' → 'Hell','o' → 'Hello'
   │
   ▼ 查表 → token id
'Hello' → 15496
',' → 11
'Ġworld' → 995
'!' → 0
```

四个组件每一个都有微妙处。我一个个讲。

## 1. byte → unicode 映射

GPT-2 的 BPE 在 **字节** 上做（不是 unicode 字符），但代码处理用 **字符串**。所以需要一个可逆映射：每个 byte (0..255) 映射到一个可显示的 unicode 字符。

为什么？两个原因：

1. **覆盖所有 byte**。Python 的 `str` 不能直接放控制字符（`\x00`-`\x1f`）和很多非 ASCII 字节。映射到打印字符让 `' '`, `'\t'`, `'\xe4'` 这些都能进字符串。
2. **可显示**。tokenization 之后的中间表示可以打印 / 拷贝粘贴 / 看 vocab.bpe 文件。

OpenAI 的实现：

```python
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1))    # 33..126 ASCII printable
    bs += list(range(ord("¡"), ord("¬") + 1))    # 161..172
    bs += list(range(ord("®"), ord("ÿ") + 1))    # 174..255 (sans 173 SOFT HYPHEN)
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)  # remaining bytes → U+0100, U+0101, ...
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
```

得到一个 256 项的字典：

| byte | unicode char |
|---|---|
| 0x20 (space) | `'Ġ'` (U+0120) |
| 0x21 (`!`) | `'!'` |
| 0x65 (`e`) | `'e'` |
| 0xC3 (UTF-8 lead byte) | `'Ã'` |
| ... | ... |

特点：
- 普通 ASCII（`!`-`~`）映射到自己
- 控制字符 + 高位字节映射到 U+0100 起的 PUA（Private Use Area）

decode 是反查表，把 unicode 字符串变回 byte 串再 UTF-8 decode。

## 2. Regex 预切词

GPT-2 不直接对整个字符串跑 BPE——先用 regex 把字符串切成"词"，每个词独立做 BPE。这样 `"hello world"` 不会让 `' world'` 跟前面的 `'o'` 合并。

GPT-2 的 regex（来自 [openai/gpt-2](https://github.com/openai/gpt-2/blob/master/src/encoder.py)）：

```python
PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
```

拆解：
- `'s|'t|'re|'ve|'m|'ll|'d` — 英文缩写后缀（"don't" → "don", "'t"）
- `' ?\p{L}+'` — 可选前导空格 + 一串字母
- `' ?\p{N}+'` — 可选前导空格 + 一串数字
- `' ?[^\s\p{L}\p{N}]+'` — 可选前导空格 + 一串"非字母非数字非空白"
- `'\s+(?!\S)'` — 末尾的连续空白（不跟非空白）
- `'\s+'` — 其他连续空白

关键设计："**前导空格归词**"——`' world'` 被切成一个 chunk，而不是 `' '` + `'world'`。这就是为什么 GPT-2 token 表里 `' world'` 是一个独立 token（id 995），不是 `' ' + 'world'`。

> ⚠️ Python 标准库 `re` 不支持 `\p{L}` `\p{N}`（unicode 类别）。需要用 `regex` 库。这是我 BPE 的唯一外部依赖。

## 3. BPE merge

每个 chunk（regex 切出的 piece）经过 byte → unicode 映射变成一个字符串。然后用 BPE 把这个字符串合成 token：

```python
def _bpe(self, token: str) -> str:
    word = tuple(token)              # 'Hello' → ('H', 'e', 'l', 'l', 'o')
    pairs = _get_pairs(word)         # {('H','e'), ('e','l'), ('l','l'), ('l','o')}
    while True:
        best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
        if best not in self.bpe_ranks:
            break                    # 没有可 merge 的了
        first, second = best
        new_word = []
        i = 0
        while i < len(word):
            j = word.index(first, i)
            new_word.extend(word[i:j])
            i = j
            if i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)  # merge!
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = tuple(new_word)
        if len(word) == 1:
            break
        pairs = _get_pairs(word)
    return " ".join(word)            # space-separated symbols
```

算法：
1. 把 chunk 拆成字符元组：`('H','e','l','l','o')`
2. 找出所有相邻 pair：`('H','e'), ('e','l'), ('l','l'), ('l','o')`
3. 找**优先级最高**的 pair（merge_rules 里 rank 最小的）
4. merge 那个 pair 的所有出现：`('H','e','l','l','o') → ('He','l','l','o')` 假设 `('H','e')` rank 最小
5. 重新计算 pair set，回到 3，直到没有可 merge 的

merge 优先级在 OpenAI 的 `vocab.bpe` 文件里：

```
#version: 0.2
Ġ t
Ġ a
h e
i n
r e
o n
Ġt he
Ġ s
e r
...
```

第一行（`Ġ t`）rank=0，第二行 rank=1，等等。这 50000 行规则是 GPT-2 在 WebText 上训练时统计出来的——出现频率最高的 byte pair 优先 merge。

## 4. encode / decode 主函数

```python
def encode(self, text: str) -> list[int]:
    ids = []
    for chunk in re.findall(PAT, text):
        # 1. byte-encode
        byte_str = "".join(self.byte_encoder[b] for b in chunk.encode("utf-8"))
        # 2. BPE
        merged = self._bpe(byte_str).split(" ")
        # 3. lookup
        ids.extend(self.encoder[piece] for piece in merged)
    return ids

def decode(self, ids: list[int]) -> str:
    text = "".join(self.decoder[i] for i in ids)
    return bytes(self.byte_decoder[c] for c in text).decode("utf-8", errors="replace")
```

`encoder` / `decoder` 是 `encoder.json` 文件加载来的——50,257 项的 dict，"Ġworld" → 995 这种映射。

## Vocab 文件从哪来

我没自己训 BPE merge rules——我用了 OpenAI 在 2019 年发布的官方文件：

- `encoder.json`：1.0 MB，{token_string: token_id} 字典
- `vocab.bpe`：0.5 MB，merge rules 按 rank 排序

代码自动下载：

```python
_ENCODER_URL = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json"
_VOCAB_URL   = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"

def _ensure_files():
    if not enc_path.exists():
        urllib.request.urlretrieve(_ENCODER_URL, enc_path)
    if not bpe_path.exists():
        urllib.request.urlretrieve(_VOCAB_URL, bpe_path)
```

第一次运行时下载，~1.5 MB，之后缓存在 `04_transformer/data/`。

> 为什么不自己训 BPE？两个原因：
> 1. **算法 vs 数据是两件事**。我想讲清楚 BPE 算法本身（这一篇讲了），训 BPE 是另一个独立话题。
> 2. **兼容性**。L3 训的 model + L4 SFT 后的 ckpt 都是用 GPT-2 BPE tokenize 的。如果换自家 BPE，所有数据要重 tokenize、所有 ckpt 要重训。
>
> 想看自训 BPE，[karpathy/minbpe](https://github.com/karpathy/minbpe) 是 200 行的极佳教程。

## 验证：bit-for-bit 等价 tiktoken

写完 230 行后，我最担心的就是细节漏洞。Vocab 50K 行，每一条 merge 顺序错了 token 序列就不对，model 推理结果就崩。

self-test：

```python
import tiktoken
ref = tiktoken.get_encoding("gpt2")
samples = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "ROMEO:\nO Juliet, wherefore art thou?",
    "Question: What is the capital of France?\nAnswer:",
    "  multiple   spaces   and\ttabs\n\nnewlines",
    "中文 日本語 🚀 emoji",
    "1234567890 + - * / = (test)",
]
for s in samples:
    ours = encode(s)
    theirs = ref.encode(s)
    assert ours == theirs, f"mismatch on {s!r}: {ours} vs {theirs}"
print("ALL MATCH")
```

跑：

```
✓ 'Hello, world!'                            ours=4   ref=4   match=True
✓ 'The quick brown fox jumps over the lazy ' ours=10  ref=10  match=True
✓ 'ROMEO:\nO Juliet, wherefore art thou?'    ours=12  ref=12  match=True
✓ 'Question: What is the capital of France?' ours=12  ref=12  match=True
✓ '  multiple   spaces   and\ttabs\n\nnewlines' ours=15 ref=15 match=True
✓ '中文 日本語 🚀 emoji'                       ours=14  ref=14  match=True
✓ '1234567890 + - * / = (test)'              ours=12  ref=12  match=True
ALL MATCH
```

**7/7 完全一致**。中文、日文、emoji、特殊字符——一个都不差。这给我信心：BPE 算法 + byte 映射 + regex 预切词，三个都对了。

## 这一层的"最小"在哪里

- **不自训 vocab**：用 OpenAI 公开的。换数据训 BPE 是另一个 200 行（minbpe 那种）。
- **没有 special tokens 管理**：GPT-2 只有一个 `<|endoftext|>` (id 50256)。chat 模型有十几个（`<|im_start|>`, `<|tool_call_start|>`, etc.）——在我们的简单设定里不需要。
- **没有 streaming decode**：generate 时拿一个 token id 就 decode 一个——简单粗暴。生产里要做"等到完整 unicode char 再输出"避免显示乱码（参考 [03_model/server.py](https://github.com/fxp/LLM-from-query-to-result/blob/main/03_model/server.py) 里的 `tokenizer.decode([next_id_int])`）。
- **没有 truncation / padding**：调用方自己管。

## 跟 tiktoken 的实际差距

| 维度 | tiktoken (Rust) | 我们 (Python) |
|---|---|---|
| encode 1 MB 文本 | ~30 ms | ~1.5 sec |
| decode | 一样 | 一样 |
| 准确度 | reference | bit-for-bit 一致 |
| 代码行数 | 几千行 Rust | 230 行 Python |
| 依赖 | tiktoken | regex |

**速度差 ~50×**——这就是 Rust 的代价。但对教学用例（每秒几次 encode）速度差异完全无所谓——整个莎翁语料 (1.1 MB) 我们的 BPE 几秒就 tokenize 完，等的时间还不如 GPU 启动。

## 实际场景跑得多快

L3 prepare：

```
corpus length: 1,115,394 chars
tokenized:     338,025 BPE tokens
```

整个莎翁 1.1 MB → 338K tokens 大约 **1.5-2 秒**（M1 / 5090 都差不多，纯 Python CPU 工作）。比读文件本身还慢 100×，但这是个一次性的事。

inference 时一个 prompt 通常几十 token，encode 几毫秒——根本不是性能瓶颈。瓶颈在 GPU 上的 forward。

## 下一篇

我们已经走完了从 query 到 forward 的所有 Python 代码。最后一站：在 GPU 上，那个 forward 里的每个 matmul 是怎么变成 SM 上的指令的——cuBLAS 怎么用 Tensor Core，Triton 怎么 fuse 出 flash-attention，naive 实现为什么 slow 10×。

[L1 — 一次矩阵乘在 GPU 上到底怎么跑 →](08-L5-gpu.md)
