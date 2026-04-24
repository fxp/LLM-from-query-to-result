"""Benchmarks for L5 kernels.

Prints TFLOPS for matmul (torch / our Triton if you add one) and ms for
attention (unfused pytorch three-pass vs our Triton fused).

Requires a CUDA GPU. For the CUDA .cu files, build them with `nvcc` and
run the binaries directly (they print their own numbers).
"""
from __future__ import annotations

import time

import torch


def bench_matmul() -> None:
    if not torch.cuda.is_available():
        print("no CUDA, skipping matmul bench")
        return
    M, N, K = 2048, 2048, 2048
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # warmup
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()
    reps = 10
    t0 = time.perf_counter()
    for _ in range(reps):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    tflops = 2 * M * N * K / dt / 1e12
    print(f"torch @ (cuBLAS)  {M}x{N}x{K}: {dt*1000:.2f} ms, {tflops:.1f} TFLOPS")


def bench_attention() -> None:
    if not torch.cuda.is_available():
        print("no CUDA, skipping attention bench")
        return
    try:
        from attention_triton import flash_attention
    except Exception as e:
        print(f"triton not available: {e}; only benchmarking pytorch.")
        flash_attention = None

    B, H, T, D = 4, 16, 1024, 64
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)

    def unfused():
        s = (q @ k.transpose(-1, -2)) * (D ** -0.5)
        mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), 1)
        s = s.masked_fill(mask, float("-inf"))
        return torch.softmax(s, dim=-1) @ v

    for _ in range(3): unfused()
    torch.cuda.synchronize()
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps): unfused()
    torch.cuda.synchronize()
    print(f"unfused pytorch  B={B} H={H} T={T} D={D}: {(time.perf_counter()-t0)/reps*1000:.2f} ms")

    if flash_attention is not None:
        for _ in range(3): flash_attention(q, k, v)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps): flash_attention(q, k, v)
        torch.cuda.synchronize()
        print(f"triton fused     B={B} H={H} T={T} D={D}: {(time.perf_counter()-t0)/reps*1000:.2f} ms")


if __name__ == "__main__":
    bench_matmul()
    bench_attention()
