from __future__ import annotations

import torch

from cs336_basics.model import BasicsTransformerLM


def main() -> None:
    torch.manual_seed(1)

    model = BasicsTransformerLM(
        vocab_size=128,
        context_length=16,
        d_model=32,
        num_layers=3,
        num_heads=4,
        d_ff=64,
        rope_theta=10_000.0,
    )

    token_ids = torch.randint(low=0, high=model.config["vocab_size"], size=(2, 10))
    logits = model(token_ids)

    print(model)
    print(f"\nInput token IDs shape: {tuple(token_ids.shape)}")
    print(f"Output logits shape: {tuple(logits.shape)}")
    print(f"Total parameters: {model.get_num_params():,}")


if __name__ == "__main__":
    main()
