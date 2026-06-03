"""OLMoE-compatible attention with FA4 CuTe kernels."""

from __future__ import annotations

import torch
from einops import rearrange
from flash_attn.cute import flash_attn_func

from cs336_basics.model import Linear, RMSNorm, RotaryEmbedding
from cs336_systems.flash_attention import _to_fa_dtype


class OlmoeFlashAttention(torch.nn.Module):
    """Causal MHA with Q/K RMSNorm (OLMoE) + FA4 CuTeDSL."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryEmbedding | None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(d_model, num_heads * self.d_k)
        self.k_proj = Linear(d_model, num_heads * self.d_k)
        self.v_proj = Linear(d_model, num_heads * self.d_v)
        self.output_proj = Linear(num_heads * self.d_v, d_model)
        self.q_norm = RMSNorm(d_model)
        self.k_norm = RMSNorm(d_model)
        self.positional_encoder = positional_encoder

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        *batch_dims, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        q = self.q_norm(self.q_proj(x))
        k = self.k_norm(self.k_proj(x))
        v = self.v_proj(x)

        q, k, v = (
            rearrange(t, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
            for t in (q, k, v)
        )

        if self.positional_encoder is not None:
            if token_positions is not None:
                token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
            q = self.positional_encoder(q, token_positions)
            k = self.positional_encoder(k, token_positions)

        q = _to_fa_dtype(q).transpose(-2, -3)
        k = _to_fa_dtype(k).transpose(-2, -3)
        v = _to_fa_dtype(v).transpose(-2, -3)
        attn_output, _ = flash_attn_func(q, k, v, causal=True, return_lse=True)
        attn_output = rearrange(attn_output, "batch seq heads d_v -> batch seq (heads d_v)").contiguous()
        return self.output_proj(attn_output)
