from __future__ import annotations

import torch

from cs336_basics.model import SwiGLU, silu


def main() -> None:
    torch.manual_seed(0)

    batch_size = 2
    sequence_length = 3
    d_model = 4
    d_ff = 8

    swiglu = SwiGLU(d_model=d_model, d_ff=d_ff)
    x = torch.randn(batch_size, sequence_length, d_model)

    # SwiGLU(x) = W2(SiLU(W1 x) * W3 x)
    gate = silu(swiglu.w1(x))
    value = swiglu.w3(x)
    hidden = gate * value
    y_manual = swiglu.w2(hidden)
    y_module = swiglu(x)

    print("=== SwiGLU shape flow ===")
    print(f"x:              {tuple(x.shape)}")
    print(f"W1(x) gate:     {tuple(gate.shape)}")
    print(f"W3(x) value:    {tuple(value.shape)}")
    print(f"gate * value:   {tuple(hidden.shape)}")
    print(f"W2(hidden):     {tuple(y_module.shape)}")
    print()
    print(f"manual formula matches module? {torch.allclose(y_manual, y_module)}")


if __name__ == "__main__":
    main()
