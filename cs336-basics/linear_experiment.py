from __future__ import annotations

import torch
from einops import einsum

from cs336_basics.model import Embedding, Linear


def show_parameter(module_name: str, parameter_name: str, parameter: torch.nn.Parameter) -> None:
    print(f"{module_name}.{parameter_name}")
    print(f"  type: {type(parameter).__name__}")
    print(f"  shape: {tuple(parameter.shape)}")
    print(f"  requires_grad: {parameter.requires_grad}")


def main() -> None:
    torch.manual_seed(0)

    print("=== Linear ===")
    linear = Linear(d_in=3, d_out=2)
    show_parameter("linear", "weight", linear.weight)
    print(f"registered parameters: {list(linear._parameters)}")
    print(f"state_dict keys: {list(linear.state_dict())}")

    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    y = linear(x)
    y_by_hand = einsum(x, linear.weight, "batch d_in, d_out d_in -> batch d_out")

    print(f"\nx: {tuple(x.shape)} -> linear(x): {tuple(y.shape)}")
    print(f"matches manual einsum? {torch.allclose(y, y_by_hand)}")

    print("\n=== Embedding ===")
    embedding = Embedding(vocab_size=5, d_model=3)
    show_parameter("embedding", "weight", embedding.weight)
    print(f"registered parameters: {list(embedding._parameters)}")
    print(f"state_dict keys: {list(embedding.state_dict())}")

    token_ids = torch.tensor(
        [
            [0, 2, 4],
            [3, 2, 1],
        ]
    )
    embedded = embedding(token_ids)

    print(f"\ntoken_ids: {tuple(token_ids.shape)} -> embedding(token_ids): {tuple(embedded.shape)}")
    print(token_ids)
    print("token id 2 appears twice, so both positions select embedding.weight[2]:")
    print(f"embedded[0, 1] == embedded[1, 1]? {torch.allclose(embedded[0, 1], embedded[1, 1])}")


if __name__ == "__main__":
    main()
