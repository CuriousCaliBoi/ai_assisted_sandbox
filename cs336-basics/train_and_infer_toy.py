from __future__ import annotations

import time

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import clip_gradient, cross_entropy
from cs336_basics.optimizer import AdamW


def decode(token_ids: torch.Tensor) -> str:
    return " ".join(str(token_id) for token_id in token_ids.tolist())


def tensor_summary(tensor: torch.Tensor) -> str:
    return f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def configure_device(device: str) -> None:
    if device != "cuda":
        return

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def inspect_model_artifacts(model: BasicsTransformerLM, sample_input: torch.Tensor) -> None:
    print("\n=== Model Artifacts ===")
    print(f"Config: {model.config}")
    print(f"Top-level modules: {list(model._modules)}")
    print(f"Transformer blocks in ModuleList: {len(model.layers)}")

    print("\nFirst transformer block:")
    for name, module in model.layers[0].named_children():
        print(f"  {name}: {module.__class__.__name__}")

    state_dict = model.state_dict()
    trainable_parameters = dict(model.named_parameters())
    print(f"\nstate_dict tensors: {len(state_dict)}")
    print("Representative state_dict entries:")
    for idx, (name, tensor) in enumerate(state_dict.items()):
        if idx == 12:
            remaining = len(state_dict) - idx
            print(f"  ... {remaining} more tensors")
            break
        artifact_type = "parameter" if name in trainable_parameters else "buffer"
        print(f"  {name}: {tensor_summary(tensor)} [{artifact_type}]")

    with torch.no_grad():
        logits = model(sample_input)
    print(f"\nSample input: {tensor_summary(sample_input)}")
    print(f"Sample logits: {tensor_summary(logits)}")
    print("Logits are raw next-token scores: one vocab-sized vector per input token.")


@torch.no_grad()
def generate_greedy(model: BasicsTransformerLM, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    model.eval()
    generated = prompt
    for _ in range(max_new_tokens):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=prompt.device.type == "cuda"):
            logits = model(generated[:, -model.context_length :])
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=-1)
    return generated[:, prompt.size(1) :]


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    configure_device(device)

    vocab_size = 16
    context_length = 8
    batch_size = 32
    train_steps = 100
    max_new_tokens = 64

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=32,
        num_layers=2,
        num_heads=4,
        d_ff=64,
        rope_theta=10_000.0,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    # A tiny deterministic language-modeling dataset: 0, 1, 2, ..., 15, 0, 1, ...
    dataset = (np.arange(4096, dtype=np.int64) % vocab_size).astype(np.uint16)

    print(f"Training on {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print("Using CUDA bfloat16 autocast and TF32 matmul")
    print(f"Total parameters: {model.get_num_params():,}")
    sample_input = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long, device=device)
    inspect_model_artifacts(model, sample_input)

    model.train()
    synchronize(device)
    train_start = time.perf_counter()
    for step in range(1, train_steps + 1):
        x, y = get_batch(dataset, batch_size=batch_size, context_length=context_length, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
        loss = cross_entropy(logits.float(), y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_gradient(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % 20 == 0:
            print(f"step {step:03d} | loss {loss.item():.4f}")

    synchronize(device)
    train_elapsed = time.perf_counter() - train_start
    train_tokens = train_steps * batch_size * context_length
    print(f"Training throughput: {train_tokens / train_elapsed:,.0f} tokens/sec")

    model.eval()
    prompt = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)
    synchronize(device)
    inference_start = time.perf_counter()
    generated = generate_greedy(model, prompt, max_new_tokens=max_new_tokens)
    synchronize(device)
    inference_elapsed = time.perf_counter() - inference_start
    full_sequence = torch.cat((prompt, generated), dim=-1).squeeze(0).cpu()

    generated_tokens = generated.numel()
    print("\nPrompt:   ", decode(prompt.squeeze(0).cpu()))
    print("Generated:", decode(generated.squeeze(0).cpu()))
    print("Full:     ", decode(full_sequence))
    print(f"Inference throughput: {generated_tokens / inference_elapsed:,.0f} tokens/sec")


if __name__ == "__main__":
    main()
