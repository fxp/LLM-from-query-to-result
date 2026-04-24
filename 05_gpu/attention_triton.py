"""Flash-attention-ish kernel in Triton.

Standard (unfused) attention does three HBM round-trips:
  S = Q @ K.T          (write  [B, H, T, T] to HBM)
  P = softmax(S)       (read+write that giant matrix)
  O = P @ V            (read it again)

Fused / flash attention does the whole thing in one kernel: for each row
of Q, stream over blocks of K/V, keep a running softmax statistic (m, l) in
registers, accumulate O on the fly. The [T, T] score matrix never touches
HBM. At long context this is the difference between "fits in memory" and
"doesn't".

This is a small reference implementation — not as optimized as the real
FlashAttention, but structurally the same idea.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(
    Q, K, V, O,                      # pointers to [B*H, T, D] flat tensors
    stride_qb, stride_qt, stride_qd,
    stride_kb, stride_kt, stride_kd,
    stride_vb, stride_vt, stride_vd,
    stride_ob, stride_ot, stride_od,
    T: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr,
):
    pid_m = tl.program_id(0)     # block index along Q rows
    pid_b = tl.program_id(1)     # batch * head index

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, D)

    # --- Load a block of Q rows into registers once; we reuse across K/V blocks.
    q_ptrs = Q + pid_b * stride_qb + off_m[:, None] * stride_qt + off_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=off_m[:, None] < T, other=0.0)

    # Running softmax statistics (per row).
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # --- Stream over K/V in blocks of BLOCK_N.
    for start_n in range(0, T, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + pid_b * stride_kb + off_n[:, None] * stride_kt + off_d[None, :] * stride_kd
        v_ptrs = V + pid_b * stride_vb + off_n[:, None] * stride_vt + off_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=off_n[:, None] < T, other=0.0)
        v = tl.load(v_ptrs, mask=off_n[:, None] < T, other=0.0)

        # scores: [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * SCALE
        # causal mask
        s = tl.where(off_m[:, None] >= off_n[None, :], s, float("-inf"))

        # Online softmax update (the numerically stable trick):
        #   new m = max(old m, row max of new block)
        #   rescale accumulated stats by exp(old m - new m)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = O + pid_b * stride_ob + off_m[:, None] * stride_ot + off_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=off_m[:, None] < T)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q, k, v : [B, H, T, D]. Returns o : [B, H, T, D]. Causal."""
    B, H, T, D = q.shape
    assert D in {16, 32, 64, 128}, "BLOCK_D must match D; pick a power of 2"
    q2 = q.reshape(B * H, T, D).contiguous()
    k2 = k.reshape(B * H, T, D).contiguous()
    v2 = v.reshape(B * H, T, D).contiguous()
    o2 = torch.empty_like(q2)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(T, BLOCK_M), B * H)
    _fwd[grid](
        q2, k2, v2, o2,
        *q2.stride(), *k2.stride(), *v2.stride(), *o2.stride(),
        T=T, D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        SCALE=D ** -0.5,
    )
    return o2.view(B, H, T, D)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, T, D = 2, 4, 256, 64
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    got = flash_attention(q, k, v)
    err = (ref.float() - got.float()).abs().max().item()
    print(f"max abs err vs torch SDPA: {err:.4f}  (should be < 0.02)")
