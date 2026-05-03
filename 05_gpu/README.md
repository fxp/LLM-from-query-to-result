# L5 · GPU 层

**一句话**：L4 里的每一个 `nn.Linear` 和 `attention` 本质都是矩阵乘。这一层告诉你一次矩阵乘在 GPU 上到底怎么跑，以及为什么写对了能快 10×。

## 为什么矩阵乘是核心

跑一次 GPT-2 forward，我们做过 profiling：

| 操作 | 占比 |
|---|---|
| matmul（Linear / Q·K / attn·V） | ~92% |
| softmax, layernorm, gelu     | ~5% |
| embedding lookup, reshape     | ~3% |

**所以 LLM 推理/训练性能 ≈ matmul 性能**。想理解 GPU 上的 LLM，就得理解 GPU 上的 matmul。

## GPU 的内存层级（这是理解一切的起点）

```
     ┌─────────────────────────────────────────────┐
     │  HBM / Global Memory    40-80 GB, ~1.5 TB/s │  ← 模型权重放这
     └──────────────────┬──────────────────────────┘
                        │ load / store (慢)
     ┌──────────────────▼──────────────────────────┐
     │  L2 Cache                ~40 MB, ~4 TB/s    │
     └──────────────────┬──────────────────────────┘
                        │
     ┌──────────────────▼──────────────────────────┐
     │  Shared Memory (per SM)  ~100 KB, ~20 TB/s  │  ← 分块算时把数据搬这里
     └──────────────────┬──────────────────────────┘
                        │
     ┌──────────────────▼──────────────────────────┐
     │  Registers (per thread)  ~256×32-bit        │  ← 计算在这里发生
     └─────────────────────────────────────────────┘
```

GPU 算得比读得快太多（算力 ~300 TFLOPS vs HBM ~1.5 TB/s）。**"朴素 matmul"慢不是因为算不过来，是因为一直在读 HBM**。优化的核心就一件事：**把数据搬到 shared memory 分块复用**。

## 两个 demo

### 1. 手写 CUDA matmul：朴素 vs 分块

```
  matmul_naive.cu   朴素版：每个线程算 C[i,j] 的一个元素，读 2K 次 HBM。
  matmul_tiled.cu   分块版：一个 block 协作算 C 的一个 tile，读 2K/TILE 次。
```

在 A100 上大约 `naive ≈ 2 TFLOPS`，`tiled ≈ 10 TFLOPS`，`cuBLAS ≈ 250 TFLOPS`。
我们的 tiled 版离 cuBLAS 还差 20×——那部分差距来自 Tensor Core、double buffering、vectorized loads 等更深的优化，不在本 repo 范围。但从 naive 到 tiled 这一跳（5×）能让你抓住"内存层级"这个核心思想。

### 2. Triton 版 flash-attention 雏形

Triton 是 OpenAI 做的 Python DSL，语法像 numpy 但 compile 成 CUDA。**业界真实的推理/训练 kernel 很多就是 Triton 写的**（包括 FlashAttention 的参考实现）。

`attention_triton.py` 是一个最小的 fused attention：一次 kernel 同时做 `Q·K.T`、`softmax`、`·V`，整条 attention 只读一次 HBM。和朴素 pytorch 三段式相比省掉了中间 `[B, h, T, T]` attention matrix 的 HBM 读写——当 T=4096、h=32 时这个中间矩阵有 1GB，省它就是命。

## 目录

```
05_gpu/
├── matmul_naive.cu       # 朴素 matmul，~40 行
├── matmul_tiled.cu       # 分块 matmul with shared memory，~70 行
├── attention_triton.py   # Triton fused attention，~130 行
├── benchmark.py          # 对比上面三种 + cuBLAS
└── README.md
```

## 怎么跑

需要一块 CUDA GPU。

```bash
# 编译 CUDA 版本（需要 nvcc）
cd 05_gpu
nvcc -O3 -arch=sm_80 matmul_naive.cu -o matmul_naive
nvcc -O3 -arch=sm_80 matmul_tiled.cu -o matmul_tiled
./matmul_naive   # 自测 + 打印 GFLOPS
./matmul_tiled

# Triton + benchmark
pip install triton
python benchmark.py
```

### 实测样例

**RTX 5090 (Blackwell, sm_120, 32 GB HBM)** — 用 `nvcc -arch=sm_120` 编译：

```
matmul 2048×2048×2048, fp32:
  naive CUDA      :   2.39 ms,  7.19 TFLOPS    1.0×
  tiled CUDA      :   1.86 ms,  9.24 TFLOPS    1.3×  (tiling 在 5090 上收益小:
                                                      HBM 极快，naive 已经不太被
                                                      memory-bound 卡住)
  cuBLAS (torch)  :   0.25 ms, 68.94 TFLOPS    9.6×  (Tensor Core + TF32)

attention  B=4 H=16 T=1024 D=64, fp32:
  pytorch 三段式  :   1.02 ms              1.0×
  triton fused    :   0.12 ms              8.5×    (省掉中间 [4,16,1024,1024]
                                                    attention matrix 的 HBM 读写)
```

参考：A100 上 cuBLAS 大约 240 TFLOPS、tiled 11 TFLOPS、naive 1.8 TFLOPS。
不同 GPU 间的相对差异主要看 (a) HBM 带宽 (b) Tensor Core 代际。
关键观察：`tiled / naive` 在 A100 上 ~6×，在 5090 上只有 1.3×——5090 的 HBM
快到 naive 不再 memory-bound。但 `cuBLAS / tiled` 永远是 7-25× 因为 Tensor Core
吃的是 fp16/bf16/tf32 而不是 fp32。

## 和其他层的接口

- **往上（L4）**：L4 里的 `x @ w` 最终解析成 `cuBLAS` 或 `cutlass` 的 matmul kernel。换成我们手写的 tiled 版，结果一致，只是慢 20×。换成 Triton flash-attention，结果一致，但 HBM 读写少 10×。
- **往下（硬件）**：一个 CUDA kernel 启动后，每个 SM (Streaming Multiprocessor) 拿一组 block，每个 block 有 ~1024 个线程，每 32 个线程是一个 warp，一个周期一条指令喂给整 warp——这是 GPU"算力"的来源。

## 这一层的"最小"在哪里

- **只覆盖 matmul 和 attention**：softmax / layernorm / gelu 的 kernel 思路类似（都是 "load 一 tile、在 shared 里算、store 回去"），写出来重复。
- **不追 SOTA**：手写版和 cuBLAS 差 20× 不是我们的失败——想追近 cuBLAS 需要 Tensor Core、async copy、software pipelining，每一项都是一整篇论文。这里的目标是**让你看到为什么分块是最重要的一跳**。
- **没讲多卡通信**：NCCL、all-reduce、ring topology——那是"很多个 L5 拼起来"的话题，属于下一个 repo。
