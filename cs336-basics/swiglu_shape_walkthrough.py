from __future__ import annotations

import torch

from cs336_basics.model import SwiGLU, silu


def describe(name: str, tensor: torch.Tensor) -> None:
    print(f"{name:<24} shape={tuple(tensor.shape)}")


def main() -> None:
    torch.manual_seed(0)

    batch_size = 2
    sequence_length = 4
    d_model = 6
    d_ff = 16

    x = torch.randn(batch_size, sequence_length, d_model)
    swiglu = SwiGLU(d_model=d_model, d_ff=d_ff)

    print("SwiGLU formula:")
    print("  output = W2(SiLU(W1(x)) * W3(x))")
    print()

    describe("input x", x)

    w1_x = swiglu.w1(x)
    describe("W1(x)", w1_x)

    gate = silu(w1_x)
    describe("SiLU(W1(x)) gate", gate)

    value = swiglu.w3(x)
    describe("W3(x) value", value)

    gated_hidden = gate * value
    describe("gate * value", gated_hidden)

    output = swiglu.w2(gated_hidden)
    describe("W2(gated_hidden)", output)

    print()
    print(f"Full module output matches manual pass? {torch.allclose(output, swiglu(x))}")


if __name__ == "__main__":
    main()
