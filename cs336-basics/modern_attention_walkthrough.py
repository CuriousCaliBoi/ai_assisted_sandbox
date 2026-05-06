from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import einsum, rearrange

from cs336_basics.model import RotaryEmbedding


torch.set_printoptions(precision=4, sci_mode=False)


def show(name: str, tensor: torch.Tensor) -> None:
    print(f"\n{name}")
    print(f"shape: {tuple(tensor.shape)}")
    print(tensor)


def naive_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Readable reference attention. It materializes the full scores matrix."""
    head_dim = q.size(-1)
    scores = einsum(q, k, "batch heads query d, batch heads key d -> batch heads query key")
    scores = scores / math.sqrt(head_dim)

    query_len = q.size(-2)
    key_len = k.size(-2)
    causal_mask = torch.ones(query_len, key_len, dtype=torch.bool, device=q.device).tril()
    scores = torch.where(causal_mask, scores, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    return einsum(weights, v, "batch heads query key, batch heads key d -> batch heads query d")


def torch_sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Modern PyTorch entry point. On CUDA, PyTorch can dispatch to fused kernels."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def grouped_query_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    GQA/MQA-style attention: many query heads share fewer key/value heads.

    Example shape:
        q: (batch, 4 query_heads, seq, head_dim)
        k: (batch, 2 kv_heads,    seq, head_dim)
        v: (batch, 2 kv_heads,    seq, head_dim)

    PyTorch's enable_gqa=True repeats K/V heads logically inside the SDPA call.
    Production kernels try to avoid physically materializing repeated K/V data.
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)


def decode_one_token_with_kv_cache(
    q_new: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tiny inference-time KV cache example.

    During generation, only the newest token's Q is needed, but it attends to all
    previous cached K/V tokens plus the newest K/V. The cache avoids recomputing
    K/V for the whole prefix at every generation step.
    """
    k_cache = torch.cat((k_cache, k_new), dim=-2)
    v_cache = torch.cat((v_cache, v_new), dim=-2)

    # q_new has query length 1, so there are no future query positions to mask.
    out = F.scaled_dot_product_attention(q_new, k_cache, v_cache, is_causal=False)
    return out, k_cache, v_cache


def main() -> None:
    torch.manual_seed(0)

    batch = 1
    seq = 4
    d_model = 8
    num_heads = 2
    head_dim = d_model // num_heads

    x = torch.arange(batch * seq * d_model, dtype=torch.float32).reshape(batch, seq, d_model) / 10
    qkv_proj = torch.nn.Linear(d_model, 3 * d_model, bias=False)
    rope = RotaryEmbedding(context_length=seq + 1, dim=head_dim, theta=10_000.0)

    q, k, v = qkv_proj(x).chunk(3, dim=-1)
    q, k, v = (
        rearrange(tensor, "batch seq (heads d) -> batch heads seq d", heads=num_heads)
        for tensor in (q, k, v)
    )

    q = rope(q, pos_ids=None)
    k = rope(k, pos_ids=None)

    show("Input x", x)
    show("Q after projection, head split, and RoPE", q)

    naive = naive_causal_attention(q, k, v)
    sdpa = torch_sdpa_attention(q, k, v)

    show("Naive causal attention output", naive)
    show("PyTorch SDPA output", sdpa)
    show("Naive minus SDPA", naive - sdpa)

    print("\nKey 2026 idea:")
    print("Naive attention materializes scores/weights; SDPA can use fused memory-efficient kernels.")

    q_gqa = torch.randn(batch, 4, seq, head_dim)
    k_gqa = torch.randn(batch, 2, seq, head_dim)
    v_gqa = torch.randn(batch, 2, seq, head_dim)
    show("Grouped-query attention output", grouped_query_attention(q_gqa, k_gqa, v_gqa))

    q_new = torch.randn(batch, num_heads, 1, head_dim)
    k_new = torch.randn(batch, num_heads, 1, head_dim)
    v_new = torch.randn(batch, num_heads, 1, head_dim)
    out_new, k_cache, v_cache = decode_one_token_with_kv_cache(q_new, k_new, v_new, k, v)

    show("One-token decode output using KV cache", out_new)
    show("Updated K cache", k_cache)
    show("Updated V cache", v_cache)


if __name__ == "__main__":
    main()
