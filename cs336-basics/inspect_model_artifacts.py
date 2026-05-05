from __future__ import annotations

import torch
from torch import nn

from cs336_basics.model import BasicsTransformerLM


def summarize_tensor(name: str, tensor: torch.Tensor) -> str:
    return f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)


def main() -> None:
    torch.manual_seed(0)

    model = BasicsTransformerLM(
        vocab_size=32,
        context_length=8,
        d_model=16,
        num_layers=2,
        num_heads=4,
        d_ff=32,
        rope_theta=10_000.0,
    )

    print_section("1. The model object")
    print(model)
    print(f"\nTraining mode? {model.training}")
    print(f"Config saved by the model: {model.config}")

    print_section("2. Immediate child modules")
    for name, child in model.named_children():
        print(f"{name}: {child.__class__.__name__}")

    print_section("3. ModuleList: repeated transformer blocks")
    print(f"type(model.layers): {type(model.layers)}")
    print(f"len(model.layers): {len(model.layers)}")
    for layer_index, layer in enumerate(model.layers):
        print(f"model.layers[{layer_index}]: {layer.__class__.__name__}")

    print_section("4. All named modules")
    for name, module in model.named_modules():
        display_name = name or "<root model>"
        print(f"{display_name}: {module.__class__.__name__}")

    print_section("5. Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        total_params += parameter.numel()
        print(
            f"{name}: shape={tuple(parameter.shape)}, "
            f"requires_grad={parameter.requires_grad}, numel={parameter.numel()}"
        )
    print(f"\nTotal parameters: {total_params:,}")

    print_section("6. Buffers")
    buffers = list(model.named_buffers())
    if not buffers:
        print("No buffers registered.")
    for name, buffer in buffers:
        print(summarize_tensor(name, buffer))
    print("\nNote: this model's RoPE cache is a non-persistent buffer, so it appears in named_buffers()")
    print("but is intentionally omitted from state_dict().")

    print_section("7. state_dict")
    state_dict = model.state_dict()
    print(f"state_dict type: {type(state_dict)}")
    print(f"number of entries: {len(state_dict)}")
    for name, tensor in state_dict.items():
        print(summarize_tensor(name, tensor))

    print_section("8. Forward pass artifacts")
    token_ids = torch.randint(low=0, high=model.config["vocab_size"], size=(2, 5))
    logits = model(token_ids)
    print(summarize_tensor("token_ids", token_ids))
    print(summarize_tensor("logits", logits))
    print("Expected logits shape: (batch_size, sequence_length, vocab_size)")

    print_section("9. train/eval mode")
    model.eval()
    print(f"After model.eval(), training mode? {model.training}")
    print(f"First block training mode? {model.layers[0].training}")

    model.train()
    print(f"After model.train(), training mode? {model.training}")
    print(f"First block training mode? {model.layers[0].training}")

    print_section("10. What to look for in PyTorch model files")
    artifacts = [
        ("nn.Module subclass", "the Python class that owns layers and forward()"),
        ("__init__", "where submodules, parameters, and buffers are registered"),
        ("forward()", "the computation graph used when calling model(inputs)"),
        ("nn.ModuleList", "a registered list of repeated submodules"),
        ("named_modules()", "the tree of modules inside the model"),
        ("named_parameters()", "learned tensors updated by the optimizer"),
        ("named_buffers()", "non-parameter tensors; persistent buffers are saved in state_dict()"),
        ("state_dict()", "serializable mapping of parameter and buffer names to tensors"),
        ("train()/eval()", "mode switches that propagate through submodules"),
    ]
    for artifact, meaning in artifacts:
        print(f"{artifact}: {meaning}")

    assert isinstance(model.layers, nn.ModuleList)


if __name__ == "__main__":
    main()
