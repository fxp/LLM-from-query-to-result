# 08 · L1：一次矩阵乘在 GPU 上到底怎么跑

> [← L4b BPE](07-L4-bpe.md) ｜ 代码：[`05_gpu/`](https://github.com/fxp/LLM-from-query-to-result/tree/main/05_gpu) ｜ [下一篇 →](09-end-to-end-trace.md)

到这一篇，我们已经从浏览器一路走到了 `model.step()` 内部的 forward pass。但最后一公里没讲：`x @ w` 这一行 PyTorch 代码，到底在 GPU 上怎么跑？为什么手写 matmul 比 cuBLAS 慢 10×？为什么 flash-attention 是命？

这一篇讲完之后，你看到一段 transformer 代码，应该能在脑子里映射到 GPU SM 上的具体指令——这是性能直觉的来源。

## 为什么 matmul 是 LLM 的核心

跑一次 GPT-2 forward 做 profiling：

| 操作 | 占 CUDA 时间 |
|---|---|
| matmul（Linear / Q·K / attn·V） | ~92% |
| softmax / layernorm / gelu / reshape | ~5% |
| embedding lookup / sampling | ~3% |

**LLM 推理性能 ≈ matmul 性能**。

每个 transformer block 有 6 个 matmul：
- `c_attn` (D → 3D)
- Q @ K.T
- (softmax results) @ V
- `c_proj` (D → D)
- `c_fc` (D → 4D)
- `c_proj` (4D → D)

12 层 × 6 = 72 个 matmul/forward。124M model + 1024 token prompt = ~1.5 TFLOP per forward。这个量在 RTX 4090 fp16 上理论 ~5 ms，实际 ~10-20 ms。

## GPU 的内存层级（理解一切的起点）

```
     ┌─────────────────────────────────────────────┐
     │  HBM / Global Memory    24-80 GB, ~1-3 TB/s │  ← 模型权重住这
     └──────────────────┬──────────────────────────┘
                        │ (慢)
     ┌──────────────────▼──────────────────────────┐
     │  L8 Cache                ~40 MB, ~4 TB/s    │
     └──────────────────┬──────────────────────────┘
                        │
     ┌──────────────────▼──────────────────────────┐
     │  Shared Memory (per SM)  ~100 KB, ~20 TB/s  │  ← 分块算时把数据搬这里
     └──────────────────┬──────────────────────────┘
                        │
     ┌──────────────────▼──────────────────────────┐
     │  Registers (per thread)  ~256 × 32-bit       │  ← 实际计算在这里
     └─────────────────────────────────────────────┘
```

GPU 算得比读得**快太多**：

| | RTX 4090 |
|---|---|
| fp32 算力 | 80 TFLOPS |
| fp16/bf16 (Tensor Core) | 330 TFLOPS |
| TF32 | 165 TFLOPS |
| HBM 带宽 | 1 TB/s |

意思是：每秒能算 80 万亿次浮点运算，但每秒只能从 HBM 读 1 TB（= 250 GFLOP fp32 数据）。算力 / 带宽 ≈ 320。**任何一次浮点运算如果只重用 < 320 次，就是 memory-bound**。

朴素 matmul：每个 C 元素读 K 个 A + K 个 B = 2K floats，做 K MAC——重用 0.5。极度 memory-bound。

## 两个 demo

### demo 1：手写 CUDA matmul，naive vs tiled

`matmul_naive.cu`（~70 行）：

```cuda
__global__ void matmul_naive(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}
```

每个 thread 算 C 的一个元素：读 K 个 A（一行）和 K 个 B（一列），乘加 K 次，写一个 C。

**问题**：`row=0` 的 1024 个 thread 都读同样的 A 第 0 行——HBM 上同一段数据被读 1024 次。这是 memory traffic 的浪费。

`matmul_tiled.cu`（~90 行）：

```cuda
__global__ void matmul_tiled(...) {
    __shared__ float As[TILE][TILE];   // 分块在 shared memory 里
    __shared__ float Bs[TILE][TILE];

    float acc = 0;
    for (int t = 0; t < K / TILE; t++) {
        // 协作加载：256 个 thread 一起填 As 和 Bs
        As[ty][tx] = A[row * K + t * TILE + tx];
        Bs[ty][tx] = B[(t * TILE + ty) * N + col];
        __syncthreads();
        // 在 shared memory 里乘加
        for (int k = 0; k < TILE; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    C[row * N + col] = acc;
}
```

关键：一个 block 的 256 个 thread **协作**，把 A 的 [TILE×TILE] 块和 B 的 [TILE×TILE] 块**一次性**搬到 shared memory，然后每个 thread 在 shared memory 里做 TILE 次乘加。

效果：A 的每行从 HBM 读 **N/TILE** 次（不是 N 次）。TILE=32 时 32× 减少。

实测（RTX 5090, fp32, 2048×2048×2048）：

```
naive CUDA   : 2.39 ms,  7.19 TFLOPS    1.0×
tiled CUDA   : 1.86 ms,  9.24 TFLOPS    1.3×    
cuBLAS torch : 0.25 ms, 68.94 TFLOPS    9.6×
```

**Tiled vs naive 在 5090 上只快 1.3×**。这反直觉——A100 上 tile 通常带 5-6× 加速。

为什么？**5090 的 HBM3e 实在太快了**。HBM3e 带宽 ~3 TB/s。Naive 版本理论 256 GFLOP / (256MB × 4 bytes) = 250 GFLOP/s memory limit——但 5090 算到了 7.19 TFLOPS 远超这个。说明实际**没那么 memory-bound**——L8 cache（5090 是 96 MB）已经吃住了大量复读，HBM 没真的被打到。

A100 的 HBM2e 带宽 1.5 TB/s 一半左右，那时候 tile 优势才显著。

但 **`cuBLAS / tiled` = 7-25×** 在所有现代 GPU 上都成立。这个 gap 是哪来的？**Tensor Core**。

cuBLAS 用 Tensor Core 跑 TF32（带宽 ~165 TFLOPS）或 fp16（~330 TFLOPS），手写 fp32 普通 SM 算力大概 80 TFLOPS。光这一项就差 2-4×。再加上 cuBLAS 还有 register tiling、async copy、software pipelining 这些更深层优化，实际见到 7-25×。

### demo 2：Triton 写 flash-attention

[Triton](https://triton-lang.org/) 是 OpenAI 做的 GPU DSL，语法像 numpy 但 compile 成 CUDA。flash-attention 的[原始论文实现](https://github.com/Dao-AILab/flash-attention)就是 Triton。

朴素 attention：

```python
# Q [B, h, T, hd], K [B, h, T, hd], V [B, h, T, hd]
scores = Q @ K.transpose(-2, -1)        # [B, h, T, T]   — kernel 1
scores = scores / sqrt(hd)
weights = softmax(scores, dim=-1)        # [B, h, T, T]   — kernel 2
out = weights @ V                        # [B, h, T, hd]  — kernel 3
```

三个 kernel，中间产生 [B, h, T, T] 的 attention matrix。当 T=4K，B=4，h=32：

```
attention matrix size = 4 × 32 × 4096 × 4096 × 2 bytes (fp16) = 4 GB
```

这 4 GB 要从 GPU 算完写回 HBM，再读出来 softmax，再写回去，再读出来乘 V。**HBM 来回 3 次** for 4 GB。HBM 带宽 1 TB/s 意味着光这一来一回就 12 ms。

Flash-attention 的核心 trick：把这三个 kernel **fuse 成一个**——Q @ K.T、softmax、@ V 在 SM 内部用 shared memory 完成，attention matrix **从来不下到 HBM**。

`05_gpu/attention_triton.py`（~130 行）：

```python
@triton.jit
def _attn_fwd(Q, K, V, Out, ...):
    # 每个 block 处理 Q 的一个 [BLOCK_M, hd] tile
    q = tl.load(Q_block_ptr)
    
    m_i = -inf  # running max for stable softmax
    l_i = 0     # running sum
    acc = 0     # accumulator for output
    
    # 遍历所有 K, V tiles
    for start_n in range(0, T, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        s = q @ k.T
        # online softmax (Flash Attention 的关键)
        m_new = maximum(m_i, max(s))
        l_new = exp(m_i - m_new) * l_i + exp(s - m_new).sum()
        acc = exp(m_i - m_new) * acc + exp(s - m_new) @ v
        m_i, l_i = m_new, l_new
    
    out = acc / l_i
    tl.store(Out_block_ptr, out)
```

"Online softmax" 是 flash-attention 论文的核心数学：把全局 softmax 拆成"按 K tile 增量更新 max + sum"，使 attention 矩阵一行可以一段一段计算，**永远不需要全部存下来**。

实测（RTX 5090, B=4 H=16 T=1024 D=64, fp32）：

```
unfused PyTorch (3 个 kernel)  : 1.02 ms
Triton fused (1 个 kernel)      : 0.12 ms     8.5× faster
```

**8.5× 加速**。当 T=8192 时差距更夸张——unfused 占的 attention matrix ~4GB，所有 HBM round-trip 加起来要十几 ms；fused 只要几 ms。

> 💡 **PyTorch 2.0 的 `F.scaled_dot_product_attention`** 在 GPU 上自动调用 flash-attention 内核（[FA2 kernel](https://github.com/Dao-AILab/flash-attention)）。所以 [L2 的 attention 实现](06-L4-transformer.md) 看起来是 3 个 op，但底层只有 1 个 kernel。这就是为什么我们的 transformer 不需要手动 fuse 也能跑得很快——PyTorch 帮我们做了。

## 实测对比表

| GPU | naive | tiled | cuBLAS | unfused attn | Triton fused | flash 加速 |
|---|---|---|---|---|---|---|
| RTX 5090 | 7.19 TFLOPS | 9.24 | 68.94 | 1.02 ms | 0.12 ms | 8.5× |
| RTX 4080 SUPER | 3.21 TFLOPS | 3.99 | 34.0 | 2.16 ms | 0.21 ms | 10.3× |
| A100 (参考) | ~1.8 | ~11.2 | ~240 | ~2.1 | ~0.6 | ~3.5× |

**几个观察**：

1. **`tiled / naive` 在 A100 上 ~6×，新卡只有 1.2-1.3×**。HBM 太快，naive 不再 memory-bound。但写出来仍然有意义——因为 Tensor Core 必须从 shared memory 喂数据，tiling 是用 Tensor Core 的前提。
2. **`cuBLAS / tiled` 永远 7-25×**。Tensor Core + lower precision 是质变。
3. **Triton flash-attn 在所有架构上都 8-10×**。这是**结构性收益**——省 HBM round-trip，跟硬件代际无关。

## SM (Streaming Multiprocessor) 内部

一个 CUDA kernel 启动后：

- **Grid**：把工作切成若干 block。如 matmul 2048² with TILE=32 → 64×64 = 4096 个 block。
- **每个 block** 分给一个 SM 跑（5090 有 170 个 SM）。
- **Block 内**：1024 个 thread，按 32 个一组叫 warp，一个 SM 同时跑多个 warp。
- **每周期一条指令喂给整 warp**——这是 SIMT (Single Instruction Multiple Thread)。

所以一个 SM 同时算 32 个 thread 的同一行代码，不同的数据。Tensor Core 在这层之上：一个 Tensor Core 指令一周期算 [16×16] × [16×16] 的 matmul 输出 16×16，**对应一个 warp 的 32 thread**。

这就是为什么 GPU 不快在"算得多"——**它快在并行度极高**。10000 个 thread 同时算同一类操作，每周期 SIMT 指令分发到一个 warp，Tensor Core 一周期出一个 16×16 matmul 结果。

## 这一层教什么

这个 repo 的 L1 不追 SOTA。手写 tiled 和 cuBLAS 差 7-25×——追这个 gap 需要 Tensor Core、async copy、register tiling、software pipelining——每一项都是一篇论文。这里的目标是：

1. **看见内存层级**。HBM → L8 → shared mem → registers，每一级带宽和延迟。
2. **看见分块复用是为什么 matmul 能快 10×**。
3. **看见 fused kernel 是为什么 flash-attention 是命**。

如果你能把这三点带走，下次面对一段慢的 GPU 代码——你脑子里会有那张内存层级图，会问"是不是 memory-bound？是不是中间结果落 HBM 了？是不是没用 Tensor Core？"。这就是性能直觉。

## 想再深入的话

- **CUTLASS**（[github](https://github.com/NVIDIA/cutlass)）：NVIDIA 开源的高性能 matmul/conv 模板库。看它就能看到 cuBLAS 那套优化怎么写的。~10x cuBLAS 代码量。
- **flash-attention**（[paper](https://arxiv.org/abs/2205.14135) + [github](https://github.com/Dao-AILab/flash-attention)）：原始论文 + 实现。Triton 版本可读，CUDA 版本快但难。
- **vllm / SGLang**（[vllm github](https://github.com/vllm-project/vllm)）：production LLM serving framework。继续 tile 思路：PagedAttention 把 KV cache 当虚拟内存分页管。
- **Triton Puzzles**（[github](https://github.com/srush/Triton-Puzzles)）：Sasha Rush 的交互式教程，写一个个小 Triton kernel 学 GPU 编程。

## 下一篇

我们走完了 7 层。最后一篇把所有层串起来，跟着 query "What is the capital of France?" 从浏览器一直追到 GPU 上的一次浮点乘法。

[端到端 trace：从一句 query 到一次浮点乘法 →](09-end-to-end-trace.md)
