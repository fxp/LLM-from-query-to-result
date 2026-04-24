"""GPT (GPT-2 architecture) from scratch, load-compatible with HuggingFace.

Architecture: a standard decoder-only Transformer.
  input_ids [B, T]
    -> wte + wpe embeddings -> [B, T, D]
    -> N × (LN, CausalSelfAttention, residual, LN, MLP, residual)
    -> final LN
    -> logits = x @ wte.T  [B, T, V]

Load GPT-2 small weights via `GPT.from_pretrained('gpt2')`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """Multi-head attention with a causal mask so each position only
    attends to earlier positions.

    The shape dance:
      x:         [B, T, D]
      c_attn(x): [B, T, 3D]  -> split -> q, k, v each [B, T, D]
      reshape:   q/k/v -> [B, n_head, T, head_dim]   (head_dim = D / n_head)
      att:       q @ k.T -> [B, n_head, T, T]  (masked + softmax)
      out:       att @ v -> [B, n_head, T, head_dim] -> reshape -> [B, T, D]
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        # Single linear that produces Q, K, V at once (GPT-2 layout).
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.c_attn(x)                          # [B, T, 3D]
        q, k, v = qkv.split(self.n_embd, dim=2)       # 3 × [B, T, D]
        # split into heads
        head_dim = D // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # [B, h, T, hd]
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        # scaled dot product attention, with built-in causal mask.
        # (PyTorch will call flash-attention under the hood on CUDA;
        # see 05_gpu/attention_triton.py for how that kernel is built.)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, h, T, hd]
        y = y.transpose(1, 2).contiguous().view(B, T, D)             # [B, T, D]
        return self.c_proj(y)


class MLP(nn.Module):
    """Feed-forward block: D -> 4D -> D. GELU activation."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    """One transformer layer: LN -> attn -> residual -> LN -> MLP -> residual.

    GPT-2 uses "pre-LN" (LN before the sublayer); GPT-1 used post-LN.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token embedding
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)   # learned position embedding
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        # lm_head shares weights with the embedding (tied) — saves ~40M params.

    def forward(self, input_ids: torch.Tensor, *, verbose: bool = False) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=input_ids.device)
        x = self.wte(input_ids) + self.wpe(pos)                # [B, T, D]
        if verbose:
            print(f"  x = embed(ids) + pos(ids)          shape={tuple(x.shape)}")
        for i, block in enumerate(self.h):
            if verbose:
                import time
                t0 = time.perf_counter()
            x = block(x)
            if verbose:
                dt = (time.perf_counter() - t0) * 1000
                print(f"  block {i:>2} attn+ffn                   shape={tuple(x.shape)}  {dt:.1f} ms")
        x = self.ln_f(x)
        if verbose:
            print(f"  ln_f                                shape={tuple(x.shape)}")
        logits = x @ self.wte.weight.T                         # tied [B, T, V]
        if verbose:
            print(f"  logits = x @ wte.T                 shape={tuple(logits.shape)}")
        return logits

    # ------------------------------------------------------------------
    # Load GPT-2 weights from HuggingFace. Names map 1:1 to our layout.
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, name: str = "gpt2") -> "GPT":
        from transformers import GPT2LMHeadModel

        cfg = GPTConfig()  # defaults are gpt2 small
        model = cls(cfg)
        hf = GPT2LMHeadModel.from_pretrained(name)
        sd_hf = hf.state_dict()
        sd = model.state_dict()

        # HF's c_attn / c_proj / c_fc use Conv1D (transposed) — we need
        # to transpose them to match nn.Linear's [out, in] layout.
        conv1d_keys = {"attn.c_attn.weight", "attn.c_proj.weight",
                       "mlp.c_fc.weight", "mlp.c_proj.weight"}
        for k_hf, v in sd_hf.items():
            # strip the "transformer." prefix HF uses
            k = k_hf.replace("transformer.", "")
            if k == "lm_head.weight":
                continue  # weight-tied to wte
            if any(k.endswith(suffix) for suffix in conv1d_keys):
                v = v.t()
            assert k in sd, f"unexpected key {k}"
            with torch.no_grad():
                sd[k].copy_(v)
        return model
