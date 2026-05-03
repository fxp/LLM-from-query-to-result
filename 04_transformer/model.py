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

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """If `kv_cache` is given, concat new K/V to it and skip the causal
        mask (single-query decode path). Returns (output, updated_kv).

        For training and prefill: pass kv_cache=None, get fresh causal attn.
        For autoregressive decode (one new token): pass the previous (k, v)
        and we attend the new query against past+new keys.
        """
        B, T, D = x.shape
        qkv = self.c_attn(x)                          # [B, T, 3D]
        q, k, v = qkv.split(self.n_embd, dim=2)       # 3 × [B, T, D]
        # split into heads
        head_dim = D // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # [B, h, T, hd]
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)        # [B, h, T_past + T, hd]
            v = torch.cat([past_v, v], dim=2)
        # scaled dot product attention, with built-in causal mask.
        # (PyTorch will call flash-attention under the hood on CUDA;
        # see 05_gpu/attention_triton.py for how that kernel is built.)
        # When kv_cache is used and T=1, no causal mask is needed: a single
        # query may attend all past keys. is_causal=True only makes sense
        # when q and k have the same length (prefill / training case).
        is_causal = kv_cache is None
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)  # [B, h, T, hd]
        y = y.transpose(1, 2).contiguous().view(B, T, D)                    # [B, T, D]
        return self.c_proj(y), (k, v)


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

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.ln_1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token embedding
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)   # learned position embedding
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        # lm_head shares weights with the embedding (tied) — saves ~40M params.

        # GPT-2 / nanoGPT init: N(0, 0.02) for all matmul weights & embeddings,
        # zeros for biases. Residual-output projections (c_proj at the end of
        # each sublayer) get extra scaling by 1/sqrt(2N) so the residual stream
        # variance doesn't blow up with depth. Without this, initial loss is
        # ~80 instead of ~11 and the first ~100 steps just walk it back down.
        self.apply(self._init_weights)
        std_resid = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=std_resid)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        verbose: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """If `targets` is given, returns (logits, loss) — used by 00_train.
        Otherwise returns logits — used by inference (03_model, 04_transformer)."""
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
            x, _ = block(x)  # discard kv (not caching during training/full forward)
            if verbose:
                dt = (time.perf_counter() - t0) * 1000
                print(f"  block {i:>2} attn+ffn                   shape={tuple(x.shape)}  {dt:.1f} ms")
        x = self.ln_f(x)
        if verbose:
            print(f"  ln_f                                shape={tuple(x.shape)}")
        logits = x @ self.wte.weight.T                         # tied [B, T, V]
        if verbose:
            print(f"  logits = x @ wte.T                 shape={tuple(logits.shape)}")
        if targets is None:
            return logits
        # Cross-entropy: predict targets[t] from input_ids[<=t]. Standard
        # next-token loss; ignore_index=-1 lets the caller mask padded slots.
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        return logits, loss

    # ------------------------------------------------------------------
    # Inference with KV cache. Used by L3's streaming server (which used
    # to lean on transformers.GPT2LMHeadModel for this; now it uses us).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        input_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """One forward pass with optional per-layer KV cache.

        Two modes:
          - prefill: kv_caches=None, input_ids=[B, T_prompt]. Returns
            (logits[B, T, V], cache[layer]=(K[B, h, T, hd], V[...]))
          - decode:  kv_caches=prev, input_ids=[B, 1]. Cache grows by 1.

        The position embedding offset is inferred from the cache length —
        so the caller doesn't have to track positions.
        """
        B, T = input_ids.shape
        if kv_caches is None:
            kv_caches = [None] * len(self.h)
            pos_offset = 0
        else:
            # Cache shape is [B, n_head, T_past, head_dim]; T_past = pos_offset.
            pos_offset = kv_caches[0][0].size(2)
        assert pos_offset + T <= self.cfg.block_size, (
            f"context overflow: {pos_offset}+{T} > {self.cfg.block_size}"
        )
        pos = torch.arange(pos_offset, pos_offset + T, device=input_ids.device)
        x = self.wte(input_ids) + self.wpe(pos)
        new_caches = []
        for block, kv in zip(self.h, kv_caches):
            x, new_kv = block(x, kv)
            new_caches.append(new_kv)
        x = self.ln_f(x)
        logits = x @ self.wte.weight.T
        return logits, new_caches

    # ------------------------------------------------------------------
    # Load GPT-2 weights from HuggingFace. Names map 1:1 to our layout.
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, name: str = "gpt2") -> "GPT":
        # CRITICAL: set HF_ENDPOINT BEFORE importing transformers.
        # huggingface_hub.constants.ENDPOINT is captured at import time
        # via os.getenv("HF_ENDPOINT", ...), so setting it later in env
        # is too late — the constant is already frozen.
        _probe_and_set_hf_endpoint()
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


# ---------------------------------------------------------------------------
# HuggingFace mirror auto-fallback. Some regions (e.g. CN) can't reach
# huggingface.co directly. If the user hasn't already set HF_ENDPOINT, we
# probe huggingface.co; on failure, set HF_ENDPOINT to
# `https://hf-mirror.com` (community-maintained CN mirror).
#
# CRITICAL — must run BEFORE `import transformers` / `import huggingface_hub`.
# Those modules read os.getenv("HF_ENDPOINT", "https://huggingface.co") at
# import time, into a module-level constant. After that the env var is dead
# weight: huggingface_hub uses the captured constant, not os.environ.
# ---------------------------------------------------------------------------
HF_MIRROR = "https://hf-mirror.com"


def _probe_and_set_hf_endpoint() -> None:
    """If HF_ENDPOINT isn't already set, probe huggingface.co; on failure,
    set HF_ENDPOINT to the mirror. Idempotent + cheap (~0-3 sec)."""
    import os
    if os.environ.get("HF_ENDPOINT"):
        return  # user explicitly chose; respect it
    import socket
    import urllib.error
    import urllib.request
    try:
        # 3-second probe of a tiny HF endpoint. We don't follow redirects
        # or download body — just verify the TCP+TLS handshake works.
        req = urllib.request.Request(
            "https://huggingface.co/api/models/openai-community/gpt2",
            headers={"User-Agent": "Mozilla/5.0 (LLM-from-query-to-result)"},
        )
        urllib.request.urlopen(req, timeout=3).close()
    except (urllib.error.URLError, socket.timeout, OSError):
        os.environ["HF_ENDPOINT"] = HF_MIRROR
        print(f"  HF Hub direct unreachable; using mirror: {HF_MIRROR}")
