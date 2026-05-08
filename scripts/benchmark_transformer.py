from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from timeit import default_timer

import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM


"""
Offline-first helper for CS336 Assignment 2.

This is not meant to replace the CUDA/Triton implementations you still need for
the assignment. It gives one local entry point for the sections you can practice
on a Mac without Wi-Fi, and it labels the CUDA-only sections so it is clear where
to switch to an NVIDIA machine.
"""


MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "tiny": {"d_model": 128, "d_ff": 512, "num_layers": 2, "num_heads": 4},
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}

ASSIGNMENT_COVERAGE: dict[str, str] = {
    "2.1.1 setup": "Verified by any task that imports cs336_basics and builds BasicsTransformerLM.",
    "2.1.2 model sizing": "--model-size uses tiny/small/medium/large/xl/10B configs.",
    "2.1.3 benchmarking_script": "--task transformer-benchmark times forward/backward/optimizer.",
    "2.1.4 nsys_profile": "Use --task transformer-benchmark on CUDA under nsys; Mac cannot run Nsight/CUDA kernels.",
    "2.1.5 mixed precision": "--dtype plus --task mixed-precision-accumulation and toy-autocast-dtypes.",
    "2.1.6 memory_profiling": "--profile-memory and --profile-saved-tensors; CUDA snapshot requires NVIDIA.",
    "3.1 autograd residuals": "--profile-saved-tensors estimates tensors saved for backward.",
    "3.2 gradient_checkpointing": "--checkpoint-every on transformer-benchmark compares checkpoint scopes.",
    "4.1 pytorch_attention": "--task attention-benchmark sweeps vanilla PyTorch attention.",
    "4.2 torch_compile": "--compile-model and --compile-attention compare compiled vs uncompiled.",
    "4.2.2 flash_forward": "Implementation belongs in cs336_systems/tests adapters; CUDA/Triton required for final.",
    "4.2.3 flash_backward": "Implementation belongs in cs336_systems/tests adapters; CUDA/Triton required for final.",
    "4.2 flash_benchmarking": "Use attention-benchmark as the PyTorch baseline; Triton timing needs CUDA.",
    "5.1 distributed communication": "--task distributed-all-reduce runs local Gloo CPU all-reduce benchmarks.",
    "5.2 naive_ddp": "Implement in cs336_systems and run tests/test_ddp.py; local Gloo can debug logic.",
    "5.3 ddp overlap": "Implement in cs336_systems; Nsight overlap screenshots require CUDA/NCCL.",
    "6 optimizer state sharding": "Implement in cs336_systems and run tests/test_sharded_optimizer.py.",
    "7 fsdp": "Implement in cs336_systems and run tests/test_fsdp.py; final accounting needs CUDA/NCCL.",
    "8 parallelism strategies": "--task parallelism-calculator prints the core communication/compute formulas.",
    "9 leaderboard": "--task leaderboard-config prints the target setup; real benchmark requires two B200 GPUs.",
}


@dataclass
class MemorySample:
    timestamp_s: float
    rss_mib: float
    accelerator_allocated_mib: float | None
    accelerator_driver_mib: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline benchmark and memory sampler for CS336 Assignment 2.")
    parser.add_argument(
        "--task",
        choices=[
            "transformer-benchmark",
            "mixed-precision-accumulation",
            "toy-autocast-dtypes",
            "residual-stream-size",
            "attention-benchmark",
            "distributed-all-reduce",
            "parallelism-calculator",
            "leaderboard-config",
        ],
        default="transformer-benchmark",
        help="Assignment section helper to run. The default preserves the original benchmark behavior.",
    )
    parser.add_argument("--list-assignment-coverage", action="store_true", help="Print the section-to-task map and exit.")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="tiny")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train"], default="train")
    parser.add_argument("--compile-model", action="store_true", help="Section 4.2: compile the Transformer model.")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Section 3.2: checkpoint every N layers; 0 disables.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-grad-forward", action="store_true", help="Use torch.no_grad() for forward-only inference.")
    parser.add_argument("--profile-memory", action="store_true", help="Sample RSS and accelerator memory during measured steps.")
    parser.add_argument(
        "--profile-saved-tensors",
        action="store_true",
        help="Use autograd saved_tensors_hooks to estimate tensors saved for backward.",
    )
    parser.add_argument("--memory-sample-interval", type=float, default=0.01)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--attention-d-models", default="16,32,64,128", help="Section 4.1 sweep values.")
    parser.add_argument("--attention-seq-lengths", default="256,1024", help="Section 4.1 sweep values; keep small on Mac.")
    parser.add_argument("--attention-batch-size", type=int, default=8)
    parser.add_argument("--compile-attention", action="store_true", help="Section 4.2: compile vanilla attention.")
    parser.add_argument("--world-size", type=int, default=2, help="Section 5.1 local Gloo process count.")
    parser.add_argument("--all-reduce-sizes-mb", default="1,10", help="Section 5.1 tensor sizes in MiB.")
    parser.add_argument("--parallel-batch", type=int, default=1024, help="Section 8 variable B.")
    parser.add_argument("--parallel-d-model", type=int, default=4096, help="Section 8 variable D.")
    parser.add_argument("--parallel-d-ff", type=int, default=11008, help="Section 8 variable D_FF.")
    parser.add_argument("--parallel-devices", type=int, default=8, help="Section 8 variable N.")
    parser.add_argument("--parallel-bandwidth-gbps", type=float, default=400.0, help="Section 8 egress bandwidth W.")
    parser.add_argument("--parallel-compute-tflops", type=float, default=1000.0, help="Section 8 accelerator speed C.")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
    return device


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def accelerator_memory_mib(device: torch.device) -> tuple[float | None, float | None]:
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 1024**2, torch.cuda.memory_reserved() / 1024**2
    if device.type == "mps":
        allocated = torch.mps.current_allocated_memory() / 1024**2 if hasattr(torch.mps, "current_allocated_memory") else None
        driver = torch.mps.driver_allocated_memory() / 1024**2 if hasattr(torch.mps, "driver_allocated_memory") else None
        return allocated, driver
    return None, None


@contextlib.contextmanager
def memory_sampler(device: torch.device, interval_s: float) -> Iterator[list[MemorySample]]:
    samples: list[MemorySample] = []
    stop = threading.Event()
    process = psutil.Process(os.getpid())
    start = time.monotonic()

    def sample_loop() -> None:
        while not stop.is_set():
            allocated, driver = accelerator_memory_mib(device)
            samples.append(
                MemorySample(
                    timestamp_s=time.monotonic() - start,
                    rss_mib=process.memory_info().rss / 1024**2,
                    accelerator_allocated_mib=allocated,
                    accelerator_driver_mib=driver,
                )
            )
            stop.wait(interval_s)

    thread = threading.Thread(target=sample_loop, daemon=True)
    thread.start()
    try:
        yield samples
    finally:
        stop.set()
        thread.join()


def autocast_context(device: torch.device, dtype: torch.dtype):
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_batch(batch_size: int, context_length: int, vocab_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return x, y


class CheckpointedTransformerLM(BasicsTransformerLM):
    """Section 3.2: same model, but layer groups can be recomputed during backward."""

    def __init__(self, *args, checkpoint_every: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_every = checkpoint_every

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        if self.checkpoint_every <= 0:
            for layer in self.layers:
                x = layer(x)
        else:
            for start in range(0, len(self.layers), self.checkpoint_every):
                group = self.layers[start : start + self.checkpoint_every]

                def run_group(hidden: torch.Tensor, group=group) -> torch.Tensor:
                    for layer in group:
                        hidden = layer(hidden)
                    return hidden

                x = checkpoint(run_group, x, use_reentrant=False)
        x = self.ln_final(x)
        return self.lm_head(x)


def run_step(
    model: BasicsTransformerLM,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    no_grad_forward: bool,
) -> dict[str, float]:
    timings: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)

    grad_context = torch.no_grad() if mode == "forward" and no_grad_forward else contextlib.nullcontext()
    with grad_context:
        synchronize(device)
        t0 = default_timer()
        with autocast_context(device, dtype):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        synchronize(device)
        timings["forward_s"] = default_timer() - t0

    if mode in {"forward_backward", "train"}:
        synchronize(device)
        t0 = default_timer()
        loss.backward()
        synchronize(device)
        timings["backward_s"] = default_timer() - t0

    if mode == "train":
        synchronize(device)
        t0 = default_timer()
        optimizer.step()
        synchronize(device)
        timings["optimizer_s"] = default_timer() - t0

    timings["total_s"] = sum(timings.values())
    return timings


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_s": 0.0, "stdev_s": 0.0}
    return {
        "mean_s": statistics.fmean(values),
        "stdev_s": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def parse_int_list(csv: str) -> list[int]:
    return [int(value.strip()) for value in csv.split(",") if value.strip()]


def emit_json(result: dict, output_json: str | None = None) -> None:
    print(json.dumps(result, indent=2))
    if output_json:
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)


def print_assignment_coverage() -> None:
    print(json.dumps(ASSIGNMENT_COVERAGE, indent=2))


def run_mixed_precision_accumulation(device: torch.device, output_json: str | None = None) -> None:
    # Section 2.1.5: reproduces the accumulation precision experiment from the PDF.
    results = {}
    for name, accumulator_dtype, addend_dtype, cast_addend in [
        ("fp32_accumulator_fp32_addend", torch.float32, torch.float32, False),
        ("fp16_accumulator_fp16_addend", torch.float16, torch.float16, False),
        ("fp32_accumulator_fp16_addend", torch.float32, torch.float16, False),
        ("fp32_accumulator_fp16_addend_cast_to_fp32", torch.float32, torch.float16, True),
    ]:
        s = torch.tensor(0, dtype=accumulator_dtype, device=device)
        for _ in range(1000):
            x = torch.tensor(0.01, dtype=addend_dtype, device=device)
            s += x.to(torch.float32) if cast_addend else x
        results[name] = {"value": float(s.cpu()), "dtype": str(s.dtype)}
    emit_json({"assignment_section": "2.1.5 mixed_precision_accumulation", "results": results}, output_json)


def run_toy_autocast_dtypes(device: torch.device, dtype: torch.dtype, output_json: str | None = None) -> None:
    # Section 2.1.5: observes autocast dtypes through Linear, LayerNorm, logits, loss, and gradients.
    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(16, 10, bias=False)
            self.ln = nn.LayerNorm(10)
            self.fc2 = nn.Linear(10, 4, bias=False)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu(self.fc1(x))
            x = self.ln(x)
            return self.fc2(x)

    model = ToyModel().to(device)
    seen: dict[str, str] = {}

    def capture(name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            seen[name] = str(output.dtype)

        return hook

    model.fc1.register_forward_hook(capture("fc1_output"))
    model.ln.register_forward_hook(capture("layernorm_output"))
    model.fc2.register_forward_hook(capture("logits"))
    x = torch.randn(8, 16, device=device)
    y = torch.randint(0, 4, (8,), device=device)
    with autocast_context(device, dtype):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
    loss.backward()
    result = {
        "assignment_section": "2.1.5 benchmarking_mixed_precision",
        "requested_autocast_dtype": str(dtype),
        "parameter_dtype_inside_context": str(next(model.parameters()).dtype),
        **seen,
        "loss_dtype": str(loss.dtype),
        "gradient_dtype": str(model.fc1.weight.grad.dtype),
    }
    emit_json(result, output_json)


def run_residual_stream_size(args: argparse.Namespace) -> None:
    # Section 2.1.6(d): activation tensor size = batch * context * d_model * bytes_per_element.
    dtype = dtype_from_name(args.dtype)
    d_model = MODEL_CONFIGS[args.model_size]["d_model"]
    bytes_total = args.batch_size * args.context_length * d_model * torch.tensor([], dtype=dtype).element_size()
    emit_json(
        {
            "assignment_section": "2.1.6(d) residual_stream_activation_size",
            "model_size": args.model_size,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "d_model": d_model,
            "dtype": str(dtype),
            "mib": bytes_total / 1024**2,
            "formula": "batch_size * context_length * d_model * element_size / 1024**2",
        },
        args.output_json,
    )


def vanilla_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.size(-1))
    if is_causal:
        mask = torch.ones(scores.size(-2), scores.size(-1), device=scores.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, -1e6)
    return torch.softmax(scores, dim=-1) @ v


def run_attention_benchmark(args: argparse.Namespace, device: torch.device) -> None:
    # Sections 4.1 and 4.2: vanilla PyTorch attention baseline, optionally torch.compile'd.
    dtype = dtype_from_name(args.dtype)
    attention_fn = vanilla_attention
    if args.compile_attention:
        attention_fn = torch.compile(attention_fn)

    results = []
    for d_model in parse_int_list(args.attention_d_models):
        for seq_len in parse_int_list(args.attention_seq_lengths):
            try:
                q = torch.randn(args.attention_batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
                k = torch.randn(args.attention_batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
                v = torch.randn(args.attention_batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

                for _ in range(args.warmup):
                    out = attention_fn(q, k, v, True)
                    out.sum().backward()
                    q.grad = k.grad = v.grad = None
                    synchronize(device)

                forward_times = []
                backward_times = []
                for _ in range(args.steps):
                    synchronize(device)
                    t0 = default_timer()
                    out = attention_fn(q, k, v, True)
                    synchronize(device)
                    forward_times.append(default_timer() - t0)

                    t0 = default_timer()
                    out.sum().backward()
                    synchronize(device)
                    backward_times.append(default_timer() - t0)
                    q.grad = k.grad = v.grad = None

                results.append(
                    {
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "status": "ok",
                        "forward": summarize(forward_times),
                        "backward": summarize(backward_times),
                    }
                )
            except RuntimeError as error:
                results.append({"d_model": d_model, "seq_len": seq_len, "status": f"runtime_error: {error}"})

    emit_json(
        {
            "assignment_sections": ["4.1 pytorch_attention", "4.2 torch_compile"],
            "device": str(device),
            "dtype": args.dtype,
            "compiled": args.compile_attention,
            "results": results,
        },
        args.output_json,
    )


def _distributed_all_reduce_worker(rank: int, world_size: int, sizes_mb: list[int], steps: int, warmup: int, port: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    output = []
    for size_mb in sizes_mb:
        numel = size_mb * 1024**2 // torch.tensor([], dtype=torch.float32).element_size()
        tensor = torch.ones(numel, dtype=torch.float32)
        for _ in range(warmup):
            dist.all_reduce(tensor, async_op=False)
        timings = []
        for _ in range(steps):
            t0 = default_timer()
            dist.all_reduce(tensor, async_op=False)
            timings.append(default_timer() - t0)
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, summarize(timings))
        if rank == 0:
            output.append({"size_mib": size_mb, "rank_summaries": gathered})
    if rank == 0:
        print(json.dumps({"assignment_section": "5.1 distributed_communication_single_node", "backend": "gloo", "results": output}, indent=2))
    dist.destroy_process_group()


def run_distributed_all_reduce(args: argparse.Namespace) -> None:
    # Section 5.1: local CPU/Gloo version for offline development. Use NCCL/CUDA for final GPU numbers.
    port = str(29500 + (os.getpid() % 1000))
    mp.spawn(
        _distributed_all_reduce_worker,
        args=(args.world_size, parse_int_list(args.all_reduce_sizes_mb), args.steps, args.warmup, port),
        nprocs=args.world_size,
        join=True,
    )


def run_parallelism_calculator(args: argparse.Namespace) -> None:
    # Section 8: formula helper for DP/FSDP/TP/2D communication-vs-compute calculations.
    b = args.parallel_batch
    d = args.parallel_d_model
    d_ff = args.parallel_d_ff
    n = args.parallel_devices
    bytes_per_elem = 2
    bandwidth = args.parallel_bandwidth_gbps * 1e9 / 8
    compute = args.parallel_compute_tflops * 1e12
    weight_bytes = 3 * d * d_ff * bytes_per_elem
    dp_backward_flops = 6 * b * d * d_ff / n
    ring_all_reduce_s = 2 * (n - 1) / n * weight_bytes / bandwidth
    dp_backward_compute_s = dp_backward_flops / compute
    result = {
        "assignment_section": "8 parallelism strategies",
        "inputs": {"B": b, "D": d, "D_FF": d_ff, "N": n, "bytes_per_elem": bytes_per_elem},
        "ring_collectives": {
            "all_gather_or_reduce_scatter_time": "(N - 1) / N * S / W",
            "all_reduce_time": "2 * (N - 1) / N * S / W",
        },
        "data_parallel_example": {
            "backward_flops_formula": "6 * B * D * D_FF / N_DP",
            "all_reduce_bytes_formula": "3 * D * D_FF * bytes_per_elem",
            "estimated_backward_compute_s": dp_backward_compute_s,
            "estimated_gradient_all_reduce_s": ring_all_reduce_s,
            "communication_bottlenecked": ring_all_reduce_s > dp_backward_compute_s,
        },
        "notes": [
            "FSDP replaces gradient all-reduce with weight all-gathers plus gradient reduce-scatters.",
            "TP adds activation collectives; the FFN layout in the PDF all-reduces the row-parallel output.",
            "2D FSDP+TP compares the max of overlapped FSDP-axis and TP-axis collective costs.",
        ],
    }
    emit_json(result, args.output_json)


def run_leaderboard_config(output_json: str | None = None) -> None:
    # Section 9: local reminder of the target benchmark; this cannot be run faithfully on a Mac.
    emit_json(
        {
            "assignment_section": "9 leaderboard",
            "requires": "two B200 GPUs, BF16, causal masking",
            "target_config": {
                "ctx_len": 32768,
                "vocab_size": 151936,
                "d_model": 4096,
                "d_ff": 11008,
                "num_layers": 34,
                "num_heads": 32,
                "batch_size": 2,
                "dtype": "torch.bfloat16",
            },
            "suggested_local_use": "Use tiny/small transformer-benchmark and attention-benchmark to debug timing paths offline.",
        },
        output_json,
    )


def main() -> None:
    args = parse_args()
    if args.list_assignment_coverage:
        print_assignment_coverage()
        return

    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    dtype = dtype_from_name(args.dtype)
    if args.task == "mixed-precision-accumulation":
        run_mixed_precision_accumulation(device, args.output_json)
        return
    if args.task == "toy-autocast-dtypes":
        run_toy_autocast_dtypes(device, dtype, args.output_json)
        return
    if args.task == "residual-stream-size":
        run_residual_stream_size(args)
        return
    if args.task == "attention-benchmark":
        run_attention_benchmark(args, device)
        return
    if args.task == "distributed-all-reduce":
        run_distributed_all_reduce(args)
        return
    if args.task == "parallelism-calculator":
        run_parallelism_calculator(args)
        return
    if args.task == "leaderboard-config":
        run_leaderboard_config(args.output_json)
        return

    config = MODEL_CONFIGS[args.model_size] | {"vocab_size": args.vocab_size, "context_length": args.context_length}
    model_cls = CheckpointedTransformerLM if args.checkpoint_every > 0 else BasicsTransformerLM
    model = model_cls(**config, checkpoint_every=args.checkpoint_every).to(device) if args.checkpoint_every > 0 else model_cls(**config).to(device)
    if args.compile_model:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    x, y = make_batch(args.batch_size, args.context_length, args.vocab_size, device)

    for _ in range(args.warmup):
        run_step(model, x, y, optimizer, args.mode, device, dtype, args.no_grad_forward)

    saved_tensor_bytes = 0
    saved_tensor_count = 0

    def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal saved_tensor_bytes, saved_tensor_count
        if not isinstance(tensor, torch.nn.Parameter):
            saved_tensor_bytes += tensor.numel() * tensor.element_size()
            saved_tensor_count += 1
        return tensor

    def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    step_timings: list[dict[str, float]] = []
    hook_context = (
        torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)
        if args.profile_saved_tensors
        else contextlib.nullcontext()
    )
    sampler_context = memory_sampler(device, args.memory_sample_interval) if args.profile_memory else contextlib.nullcontext([])

    with sampler_context as memory_samples:
        with hook_context:
            for _ in range(args.steps):
                step_timings.append(run_step(model, x, y, optimizer, args.mode, device, dtype, args.no_grad_forward))

    active_sections = [
        section
        for section in [
            "2.1.3 benchmarking_script",
            "2.1.5 benchmarking_mixed_precision" if args.dtype != "float32" else None,
            "2.1.6 memory_profiling" if args.profile_memory else None,
            "3.1 autograd residuals" if args.profile_saved_tensors else None,
            "3.2 gradient_checkpointing" if args.checkpoint_every > 0 else None,
            "4.2 torch_compile" if args.compile_model else None,
        ]
        if section is not None
    ]

    summary = {
        "assignment_sections": active_sections,
        "device": str(device),
        "dtype": args.dtype,
        "mode": args.mode,
        "model_size": args.model_size,
        "config": config,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup,
        "measurement_steps": args.steps,
        "compiled_model": args.compile_model,
        "checkpoint_every": args.checkpoint_every,
        "timings": {key: summarize([step[key] for step in step_timings if key in step]) for key in step_timings[0]},
        "saved_tensors": {
            "count": saved_tensor_count,
            "total_mib": saved_tensor_bytes / 1024**2,
            "average_per_step_mib": saved_tensor_bytes / max(args.steps, 1) / 1024**2,
        }
        if args.profile_saved_tensors
        else None,
        "memory": None,
    }

    if args.profile_memory:
        accelerator_allocated = [s.accelerator_allocated_mib for s in memory_samples if s.accelerator_allocated_mib is not None]
        accelerator_driver = [s.accelerator_driver_mib for s in memory_samples if s.accelerator_driver_mib is not None]
        summary["memory"] = {
            "samples": [asdict(sample) for sample in memory_samples],
            "peak_rss_mib": max((sample.rss_mib for sample in memory_samples), default=0.0),
            "peak_accelerator_allocated_mib": max(accelerator_allocated) if accelerator_allocated else None,
            "peak_accelerator_driver_mib": max(accelerator_driver) if accelerator_driver else None,
        }

    emit_json(summary, args.output_json)


if __name__ == "__main__":
    main()
