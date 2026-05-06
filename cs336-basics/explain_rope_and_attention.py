from __future__ import annotations

import math

import torch
from einops import rearrange

from cs336_basics.model import CausalMultiHeadSelfAttention, RotaryEmbedding, scaled_dot_product_attention


torch.set_printoptions(precision=3, sci_mode=False)


def banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def explain_rope() -> None:
    """
    High level:
    RoPE means Rotary Positional Embedding.

    Instead of adding a learned or sinusoidal position vector to token embeddings,
    RoPE rotates pairs of query/key channels by an angle that depends on the token
    position. Position 0 gets almost no rotation. Later positions rotate more.

    Why it is useful:
    Attention compares queries and keys with dot products. If both Q and K are
    rotated according to their positions, the dot product contains information
    about relative position. The model can learn patterns like "look two tokens
    back" without storing absolute position as a separate added vector.

    Low level:
    Your implementation takes a vector like:

        [x0, x1, x2, x3]

    and treats it as two adjacent 2D vectors during the rotation:

        pair 0: (x0, x1)
        pair 1: (x2, x3)

    For each pair it applies a 2D rotation matrix:

        x1_rot = cos(position_frequency) * x1 - sin(position_frequency) * x2
        x2_rot = sin(position_frequency) * x1 + cos(position_frequency) * x2

    That changes direction but preserves the magnitude of each 2D pair.

    Repo note:
    The original Assignment 2 staff implementation returned all first coordinates
    followed by all second coordinates:

        [x0_rot, x2_rot, x1_rot, x3_rot]

    We changed this repo to restore the original interleaved layout:

        [x0_rot, x1_rot, x2_rot, x3_rot]

    That makes position 0 an identity transform, which is the easier convention
    to reason about while learning RoPE.
    """

    banner("RoPE: Rotary Positional Embedding")

    context_length = 4
    head_dim = 4
    rope = RotaryEmbedding(context_length=context_length, dim=head_dim, theta=10_000.0)

    # Shape convention matches attention after heads are split:
    # (batch, heads, sequence_length, head_dim)
    q_or_k = torch.tensor(
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

    rotated = rope(q_or_k, pos_ids=None)

    print("Input shape:   ", tuple(q_or_k.shape), " = (batch, heads, seq, head_dim)")
    print("Output shape:  ", tuple(rotated.shape))
    print("\nSame vector repeated at every position:")
    print(q_or_k[0, 0])
    print("\nAfter RoPE, each position gets a different rotation:")
    print(rotated[0, 0])

    cos_sin_cache = rope._freq_cis_cache
    print("\nCached cos values by position and pair:")
    print(cos_sin_cache[0])
    print("\nCached sin values by position and pair:")
    print(cos_sin_cache[1])

    before_norms = torch.linalg.vector_norm(q_or_k[0, 0], dim=-1)
    after_norms = torch.linalg.vector_norm(rotated[0, 0], dim=-1)
    print("\nVector norm before RoPE:", before_norms)
    print("Vector norm after RoPE: ", after_norms)
    print("The norm is preserved; RoPE changes direction, not magnitude.")


def explain_scaled_dot_product_attention() -> None:
    """
    High level:
    Scaled dot-product attention answers:

        For each query token, which key tokens should I read from?

    It does this in three steps:

    1. Compare Q against K with dot products.
    2. Softmax those scores into attention weights.
    3. Use the weights to take a weighted average of V.

    Low level:
    If Q has shape (seq, d_k), K has shape (seq, d_k), and V has shape
    (seq, d_v), then:

        scores  = Q @ K.T / sqrt(d_k)       # shape: (query_seq, key_seq)
        weights = softmax(scores, dim=-1)   # rows sum to 1
        output  = weights @ V               # shape: (query_seq, d_v)

    The causal mask blocks attention to future tokens. Query position 2 can look
    at key positions 0, 1, and 2, but not 3, 4, ...
    """

    banner("SDPA: Scaled Dot-Product Attention")

    q = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    k = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    v = torch.tensor(
        [
            [10.0, 0.0],
            [0.0, 20.0],
            [30.0, 30.0],
        ]
    )

    seq_len = q.shape[0]
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    scores = q @ k.T / math.sqrt(k.shape[-1])
    masked_scores = torch.where(causal_mask, scores, float("-inf"))
    weights = torch.softmax(masked_scores, dim=-1)
    output_manual = weights @ v
    output_from_project = scaled_dot_product_attention(q, k, v, causal_mask)

    print("Q shape:", tuple(q.shape), "K shape:", tuple(k.shape), "V shape:", tuple(v.shape))
    print("\nRaw attention scores = Q @ K.T / sqrt(d_k):")
    print(scores)
    print("\nCausal mask, True means this query may read this key:")
    print(causal_mask)
    print("\nScores after masking future positions:")
    print(masked_scores)
    print("\nAttention weights after softmax. Each row sums to 1:")
    print(weights)
    print("Row sums:", weights.sum(dim=-1))
    print("\nOutput = attention_weights @ V:")
    print(output_manual)
    print("\nOutput from cs336_basics.model.scaled_dot_product_attention:")
    print(output_from_project)


def explain_causal_multi_head_self_attention() -> None:
    """
    High level:
    Causal multi-head self-attention is SDPA wrapped in learnable projections.

    Self-attention:
    The same input x creates Q, K, and V. Tokens are comparing themselves against
    other tokens in the same sequence.

    Causal:
    A token can only read itself and earlier tokens. This is what lets a language
    model predict the next token without cheating by looking ahead.

    Multi-head:
    Instead of one attention operation over the full d_model vector, the model
    splits the representation into several heads. Each head gets its own Q/K/V
    subspace and can learn a different kind of relationship.

    Low level in your class:
    1. x has shape (batch, seq, d_model).
    2. q_proj, k_proj, v_proj make Q, K, V with shape (batch, seq, d_model).
    3. rearrange splits d_model into (heads, head_dim):
           (batch, seq, d_model) -> (batch, heads, seq, head_dim)
    4. RoPE rotates Q and K, if enabled.
    5. Causal SDPA returns one output per head.
    6. Heads are concatenated back to d_model.
    7. output_proj mixes the heads.
    """

    banner("Causal Multi-Head Self-Attention")

    torch.manual_seed(0)
    batch_size = 1
    seq_len = 4
    d_model = 8
    num_heads = 2
    head_dim = d_model // num_heads

    rope = RotaryEmbedding(context_length=seq_len, dim=head_dim, theta=10_000.0)
    attn = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, positional_encoder=rope)

    x = torch.arange(batch_size * seq_len * d_model, dtype=torch.float32).reshape(batch_size, seq_len, d_model) / 10

    print("Input x shape:", tuple(x.shape), "= (batch, seq, d_model)")
    print("Input x:")
    print(x)

    q = attn.q_proj(x)
    k = attn.k_proj(x)
    v = attn.v_proj(x)
    print("\nAfter q/k/v projections:")
    print("Q:", tuple(q.shape), "K:", tuple(k.shape), "V:", tuple(v.shape))

    q_heads, k_heads, v_heads = (
        rearrange(tensor, "batch seq (heads d) -> batch heads seq d", heads=num_heads)
        for tensor in (q, k, v)
    )
    print("\nAfter splitting into heads:")
    print("Q heads:", tuple(q_heads.shape), "= (batch, heads, seq, head_dim)")

    q_rope = rope(q_heads, pos_ids=None)
    k_rope = rope(k_heads, pos_ids=None)
    print("\nHead 0 Q before RoPE:")
    print(q_heads[0, 0])
    print("\nHead 0 Q after RoPE:")
    print(q_rope[0, 0])

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    head0_scores = q_rope[0, 0] @ k_rope[0, 0].T / math.sqrt(head_dim)
    head0_masked_scores = torch.where(causal_mask, head0_scores, float("-inf"))
    head0_weights = torch.softmax(head0_masked_scores, dim=-1)
    print("\nHead 0 causal attention weights:")
    print(head0_weights)

    output = attn(x)
    print("\nFinal module output shape:", tuple(output.shape), "= (batch, seq, d_model)")
    print("Final module output:")
    print(output)


def main() -> None:
    explain_rope()
    explain_scaled_dot_product_attention()
    explain_causal_multi_head_self_attention()


if __name__ == "__main__":
    main()
