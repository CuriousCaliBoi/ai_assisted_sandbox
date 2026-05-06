from __future__ import annotations

import math

import torch
from einops import einsum, rearrange


torch.set_printoptions(precision=4, sci_mode=False)


def show(name: str, tensor: torch.Tensor) -> None:
    print(f"\n{name}")
    print(f"shape: {tuple(tensor.shape)}")
    print(tensor)


def section(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def softmax_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Standard causal softmax attention.

    q, k: (seq, d)
    v:    (seq, d_v)
    """
    scores = q @ k.T / math.sqrt(q.size(-1))

    seq = q.size(0)
    causal_mask = torch.ones(seq, seq, dtype=torch.bool).tril()
    masked_scores = torch.where(causal_mask, scores, float("-inf"))

    weights = torch.softmax(masked_scores, dim=-1)
    return weights @ v


def sliding_window_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Local/sliding-window causal attention.

    Token i can read only:
        max(0, i - window_size + 1), ..., i

    This keeps the softmax attention behavior, but reduces the number of keys each
    query can inspect.
    """
    scores = q @ k.T / math.sqrt(q.size(-1))

    seq = q.size(0)
    query_positions = rearrange(torch.arange(seq), "query -> query 1")
    key_positions = rearrange(torch.arange(seq), "key -> 1 key")

    is_past_or_current = key_positions <= query_positions
    is_inside_window = key_positions >= query_positions - window_size + 1
    mask = is_past_or_current & is_inside_window

    masked_scores = torch.where(mask, scores, float("-inf"))
    weights = torch.softmax(masked_scores, dim=-1)
    return weights @ v


def positive_feature_map(x: torch.Tensor) -> torch.Tensor:
    """
    A simple educational feature map for linear attention.

    Real papers/kernels use more careful feature maps depending on the method
    (Performer/FAVOR+, cosFormer, etc.). ELU + 1 is common in simple demos because
    it is positive and easy to inspect.
    """
    return torch.nn.functional.elu(x) + 1


def linear_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Causal linear attention.

    Softmax attention uses:
        softmax(q_i @ k_j) over all j <= i

    Linear attention replaces the softmax kernel with:
        similarity(q_i, k_j) = phi(q_i)^T phi(k_j)

    Then:
        out_i = sum_{j <= i} phi(q_i)^T phi(k_j) v_j
                ---------------------------------------
                sum_{j <= i} phi(q_i)^T phi(k_j)

    The trick:
        sum_{j <= i} phi(k_j) v_j
        and
        sum_{j <= i} phi(k_j)

    can be maintained with prefix sums, so we do not build a full seq x seq
    attention matrix.
    """
    phi_q = positive_feature_map(q)
    phi_k = positive_feature_map(k)

    # Per token, build the outer product phi(k_j) v_j.
    # Shape: (seq, feature_dim, value_dim)
    kv_outer = einsum(phi_k, v, "seq f, seq dv -> seq f dv")

    # Causal prefix summaries:
    # prefix_kv[i] = sum_{j <= i} phi(k_j) v_j
    # prefix_k[i]  = sum_{j <= i} phi(k_j)
    prefix_kv = torch.cumsum(kv_outer, dim=0)
    prefix_k = torch.cumsum(phi_k, dim=0)

    numerator = einsum(phi_q, prefix_kv, "seq f, seq f dv -> seq dv")
    denominator = einsum(phi_q, prefix_k, "seq f, seq f -> seq").unsqueeze(-1)
    return numerator / denominator.clamp_min(1e-8)


def linear_causal_attention_loop(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Same as linear_causal_attention, but written as a token-by-token loop."""
    phi_q = positive_feature_map(q)
    phi_k = positive_feature_map(k)

    feature_dim = phi_k.size(-1)
    value_dim = v.size(-1)
    running_kv = torch.zeros(feature_dim, value_dim)
    running_k = torch.zeros(feature_dim)

    outputs = []
    for i in range(q.size(0)):
        running_kv = running_kv + torch.outer(phi_k[i], v[i])
        running_k = running_k + phi_k[i]

        numerator = phi_q[i] @ running_kv
        denominator = phi_q[i] @ running_k
        outputs.append(numerator / denominator.clamp_min(1e-8))

    return torch.stack(outputs)


def grouped_query_attention_demo() -> None:
    """
    Grouped-query attention is not a new scoring rule.

    It changes the head layout:
        many Q heads
        fewer K/V heads

    That reduces inference KV-cache memory. Here we manually repeat K/V heads so
    the ordinary attention formula can run.
    """
    section("4. Grouped-Query Attention Shape Trick")

    batch = 1
    seq = 3
    query_heads = 4
    kv_heads = 2
    head_dim = 2

    q = torch.arange(batch * query_heads * seq * head_dim, dtype=torch.float32).reshape(
        batch, query_heads, seq, head_dim
    )
    k = torch.arange(batch * kv_heads * seq * head_dim, dtype=torch.float32).reshape(batch, kv_heads, seq, head_dim)
    v = k + 100

    show("Q has more heads", q)
    show("K has fewer heads", k)
    show("V has fewer heads", v)

    repeats = query_heads // kv_heads
    k_repeated = k.repeat_interleave(repeats, dim=1)
    v_repeated = v.repeat_interleave(repeats, dim=1)

    show("K repeated to match Q heads", k_repeated)
    show("V repeated to match Q heads", v_repeated)

    scores = einsum(q, k_repeated, "batch heads query d, batch heads key d -> batch heads query key")
    scores = scores / math.sqrt(head_dim)

    causal_mask = torch.ones(seq, seq, dtype=torch.bool).tril()
    weights = torch.softmax(torch.where(causal_mask, scores, float("-inf")), dim=-1)
    output = einsum(weights, v_repeated, "batch heads query key, batch heads key d -> batch heads query d")
    show("Grouped-query output after repeating K/V", output)


def main() -> None:
    section("1. Standard Causal Softmax Attention")

    q = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    k = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ]
    )
    v = torch.tensor(
        [
            [10.0, 0.0],
            [0.0, 20.0],
            [30.0, 30.0],
            [40.0, 10.0],
        ]
    )

    show("Q", q)
    show("K", k)
    show("V", v)

    scores = q @ k.T / math.sqrt(q.size(-1))
    show("All QK scores before masking", scores)

    softmax_output = softmax_attention_reference(q, k, v)
    show("Standard causal softmax attention output", softmax_output)

    section("2. Sliding-Window Causal Attention")

    window_size = 2
    sliding_output = sliding_window_attention(q, k, v, window_size=window_size)
    show(f"Sliding-window output, window_size={window_size}", sliding_output)

    print("\nInterpretation:")
    print("Token 3 can read only tokens 2 and 3, not tokens 0 and 1.")

    section("3. Linear Causal Attention")

    phi_q = positive_feature_map(q)
    phi_k = positive_feature_map(k)
    show("phi(Q) = ELU(Q) + 1", phi_q)
    show("phi(K) = ELU(K) + 1", phi_k)

    kv_outer = einsum(phi_k, v, "seq f, seq dv -> seq f dv")
    prefix_kv = torch.cumsum(kv_outer, dim=0)
    prefix_k = torch.cumsum(phi_k, dim=0)

    show("Per-token outer products phi(K_j) outer V_j", kv_outer)
    show("Prefix sum of phi(K_j) outer V_j", prefix_kv)
    show("Prefix sum of phi(K_j)", prefix_k)

    linear_output = linear_causal_attention(q, k, v)
    linear_loop_output = linear_causal_attention_loop(q, k, v)

    show("Linear causal attention output", linear_output)
    show("Linear causal attention loop output", linear_loop_output)
    show("Vectorized minus loop", linear_output - linear_loop_output)

    print("\nImportant difference:")
    print("Linear attention is not exactly softmax attention. It changes the similarity function.")
    print("Its attraction is that causal attention can be computed with prefix sums instead of a full seq x seq matrix.")

    grouped_query_attention_demo()


if __name__ == "__main__":
    main()
