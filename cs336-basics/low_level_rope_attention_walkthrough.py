from __future__ import annotations

import math

import torch
from einops import einsum, rearrange

from cs336_basics.model import CausalMultiHeadSelfAttention, RotaryEmbedding


torch.set_printoptions(precision=4, sci_mode=False)


def show(name: str, tensor: torch.Tensor) -> None:
    print(f"\n{name}")
    print(f"shape: {tuple(tensor.shape)}")
    print(tensor)


def section(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def rope_low_level() -> None:
    """
    Low-level RoPE walkthrough.

    Goal:
    Take a query/key tensor x with shape:

        (batch, heads, seq, head_dim)

    and apply position-dependent 2D rotations to pairs of hidden dimensions.

    In this repo we changed RoPE reassembly to preserve interleaved layout:

        position 0: [1, 2, 3, 4] -> [1, 2, 3, 4]

    This differs from the original CS336 Assignment 2 staff layout, which
    returned split-half order after rotation.
    """

    section("1. RoPE Low-Level Implementation")

    context_length = 4
    head_dim = 4
    theta = 10_000.0

    # Think of this as one head's Q or K values for 4 token positions.
    x = torch.tensor(
        [
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [1.0, 2.0, 3.0, 4.0],
                    [1.0, 2.0, 3.0, 4.0],
                    [1.0, 2.0, 3.0, 4.0],
                ]
            ]
        ]
    )
    show("Input x = one Q/K vector per sequence position", x)

    # This is what RotaryEmbedding._init_cache does.
    d = torch.arange(0, head_dim, 2) / head_dim
    freqs_per_pair = torch.tensor(theta) ** -d
    positions = torch.arange(context_length)
    angles = einsum(positions, freqs_per_pair, "seq, pair -> seq pair")
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    show("Pair frequency exponents d = arange(0, head_dim, 2) / head_dim", d)
    show("Frequency per pair = theta ** -d", freqs_per_pair)
    show("Angles = position * frequency", angles)
    show("cos(angles)", cos)
    show("sin(angles)", sin)

    # This is the same split as model.py:
    # x shape:  (batch, heads, seq, head_dim)
    # x_pairs:  (batch, heads, seq, half_dim, 2)
    # x1/x2:    (batch, heads, seq, half_dim)
    x_pairs = rearrange(x, "... (half_d xy) -> ... half_d xy", xy=2)
    x1, x2 = rearrange(x, "... (half_d xy) -> xy ... half_d", xy=2).unbind(0)

    show("x viewed as adjacent 2D pairs", x_pairs)
    show("x1 = first coordinate of each pair", x1)
    show("x2 = second coordinate of each pair", x2)

    # cos/sin have shape (seq, half_dim). PyTorch broadcasts them across
    # the leading batch and heads dimensions of x1/x2.
    x1_rot = cos * x1 - sin * x2
    x2_rot = sin * x1 + cos * x2

    show("x1_rot = cos * x1 - sin * x2", x1_rot)
    show("x2_rot = sin * x1 + cos * x2", x2_rot)

    # Reassemble the rotated pair coordinates back into the original last dim.
    reassembled = rearrange(torch.stack((x1_rot, x2_rot), dim=-1), "... half_d xy -> ... (half_d xy)")
    show("Reassembled RoPE output", reassembled)

    rope = RotaryEmbedding(context_length=context_length, dim=head_dim, theta=theta)
    imported_rope_output = rope(x, pos_ids=None)
    show("Output from imported RotaryEmbedding", imported_rope_output)

    pair_norms_before = torch.linalg.vector_norm(x_pairs, dim=-1)
    pair_norms_after = torch.linalg.vector_norm(
        rearrange(reassembled, "... (half_d xy) -> ... half_d xy", xy=2),
        dim=-1,
    )
    show("2D pair norms before rotation", pair_norms_before)
    show("2D pair norms after rotation", pair_norms_after)


def attention_low_level() -> None:
    """
    Low-level causal multi-head self-attention walkthrough.

    Goal:
    Follow the same sequence as CausalMultiHeadSelfAttention.forward:

    1. Project x into Q, K, V.
    2. Split Q, K, V into heads.
    3. Apply RoPE to Q and K.
    4. Build causal mask.
    5. Compute scaled attention scores.
    6. Mask future-token scores.
    7. Softmax scores into attention weights.
    8. Weighted-sum V.
    9. Concatenate heads and apply output projection.
    """

    section("2. Causal Multi-Head Self-Attention Low-Level Implementation")

    torch.manual_seed(0)
    batch_size = 1
    seq_len = 4
    d_model = 8
    num_heads = 2
    head_dim = d_model // num_heads

    rope = RotaryEmbedding(context_length=seq_len, dim=head_dim, theta=10_000.0)
    attn = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, positional_encoder=rope)

    x = torch.arange(batch_size * seq_len * d_model, dtype=torch.float32).reshape(batch_size, seq_len, d_model) / 10
    show("Input x", x)

    q = attn.q_proj(x)
    k = attn.k_proj(x)
    v = attn.v_proj(x)
    show("Q after q_proj(x)", q)
    show("K after k_proj(x)", k)
    show("V after v_proj(x)", v)

    q_heads, k_heads, v_heads = (
        rearrange(tensor, "batch seq (heads d) -> batch heads seq d", heads=num_heads)
        for tensor in (q, k, v)
    )
    show("Q split into heads", q_heads)
    show("K split into heads", k_heads)
    show("V split into heads", v_heads)

    q_rope = rope(q_heads, pos_ids=None)
    k_rope = rope(k_heads, pos_ids=None)
    show("Q after RoPE", q_rope)
    show("K after RoPE", k_rope)

    # Causal mask: query i can read key j only when i >= j.
    iota = torch.arange(seq_len)
    qi = rearrange(iota, "query -> query 1")
    kj = rearrange(iota, "key -> 1 key")
    causal_mask = qi >= kj
    show("Causal mask", causal_mask)

    scores = einsum(q_rope, k_rope, "batch heads query d, batch heads key d -> batch heads query key") / math.sqrt(head_dim)
    masked_scores = torch.where(causal_mask, scores, float("-inf"))
    weights = torch.softmax(masked_scores, dim=-1)
    per_head_output = einsum(weights, v_heads, "batch heads query key, batch heads key d -> batch heads query d")

    show("Scaled attention scores = Q @ K.T / sqrt(head_dim)", scores)
    show("Scores after causal mask", masked_scores)
    show("Attention weights = softmax(masked_scores)", weights)
    show("Per-head output = attention_weights @ V", per_head_output)

    concatenated = rearrange(per_head_output, "batch heads seq d -> batch seq (heads d)").contiguous()
    output_manual = attn.output_proj(concatenated)
    output_module = attn(x)

    show("Concatenated heads", concatenated)
    show("Manual final output after output_proj", output_manual)
    show("Output from imported CausalMultiHeadSelfAttention", output_module)
    show("Difference between manual and module output", output_manual - output_module)


def main() -> None:
    rope_low_level()
    attention_low_level()


if __name__ == "__main__":
    main()
