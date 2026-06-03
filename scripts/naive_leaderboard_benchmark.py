"""Leaderboard training-step timing harness for CS336 Assignment 2.

Faithful port of the official leaderboard test_timing_forward_backward() code.
Uses unmodified cs336_basics BasicsTransformerLM + AdamW.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton.testing
from torch.profiler import ProfilerActivity, profile

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.fsdp import FSDP, fsdp_on_after_backward
from cs336_systems.fused_ce import fused_lm_head_cross_entropy
from cs336_systems.model import build_flash_basics_transformer_lm
from tests.adapters import get_fsdp


class Config:
    ctx_len = 32768
    vocab_size = 151936
    d_model = 4096
    d_ff = 11008
    num_layers = 34
    num_heads = 32
    torch_dtype = torch.bfloat16
    is_causal = True
    batch_size = 2


@dataclass
class TrainStepContext:
    model: torch.nn.Module
    optimizer: AdamW
    labels: torch.Tensor
    targets: torch.Tensor
    device: torch.device
    use_fsdp: bool = False
    fused_ce: bool = False
    fused_ce_chunk_size: int = 512
    forward_hidden_fn: object | None = None


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, FSDP) else model


def compute_loss(ctx: TrainStepContext) -> torch.Tensor:
    if ctx.fused_ce:
        base = _unwrap_model(ctx.model)
        if not hasattr(base, "forward_hidden"):
            raise ValueError("fused CE requires a model with forward_hidden() (use --attention cute)")
        forward_hidden = ctx.forward_hidden_fn or base.forward_hidden
        hidden = forward_hidden(ctx.labels)
        loss = fused_lm_head_cross_entropy(
            hidden,
            base.lm_head,
            ctx.targets,
            chunk_size=ctx.fused_ce_chunk_size,
        )
    else:
        logits = ctx.model(ctx.labels)
        loss = cross_entropy(logits, ctx.targets)
    return loss.sum()


def gib(bytes_val: int | float) -> float:
    return float(bytes_val) / 1024**3


def cuda_memory_snapshot(device: torch.device, label: str) -> dict:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "label": label,
        "allocated_gib": gib(torch.cuda.memory_allocated(device)),
        "reserved_gib": gib(torch.cuda.memory_reserved(device)),
        "active_gib": gib(torch.cuda.memory_stats(device)["active_bytes.all.current"]),
        "inactive_gib": gib(torch.cuda.memory_stats(device)["inactive_split_bytes.all.current"]),
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
        "num_alloc_retries": torch.cuda.memory_stats(device)["num_alloc_retries"],
        "num_ooms": torch.cuda.memory_stats(device)["num_ooms"],
        "device_index": idx,
    }


def profile_train_step_memory(ctx: TrainStepContext, top_k: int = 20) -> dict:
    """Run one training step under torch.profiler with profile_memory=True."""
    device = ctx.device

    def train_step() -> None:
        ctx.optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(ctx)
        loss.backward()
        ctx.optimizer.step()

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    phases: list[dict] = []

    def record(label: str) -> None:
        torch.cuda.synchronize(device)
        phases.append(cuda_memory_snapshot(device, label))

    record("before_step")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        ctx.optimizer.zero_grad(set_to_none=True)
        record("after_zero_grad")
        if ctx.fused_ce:
            base = _unwrap_model(ctx.model)
            forward_hidden = ctx.forward_hidden_fn or base.forward_hidden
            hidden = forward_hidden(ctx.labels)
            record("after_forward")
            loss = fused_lm_head_cross_entropy(
                hidden,
                base.lm_head,
                ctx.targets,
                chunk_size=ctx.fused_ce_chunk_size,
            ).sum()
            record("after_fused_loss")
        else:
            res = ctx.model(ctx.labels)
            record("after_forward")
            loss = cross_entropy(res, ctx.targets).sum()
        record("after_loss")
        loss.backward()
        if ctx.use_fsdp:
            fsdp_on_after_backward(ctx.model, ctx.optimizer)
        record("after_backward")
        ctx.optimizer.step()
        record("after_optimizer_step")
        torch.cuda.synchronize(device)

    events = prof.key_averages()
    by_cuda_memory = sorted(
        (
            {
                "name": evt.key,
                "self_device_memory_gib": gib(evt.self_device_memory_usage),
                "device_memory_gib": gib(evt.device_memory_usage),
                "count": evt.count,
            }
            for evt in events
            if evt.self_device_memory_usage > 0
        ),
        key=lambda row: row["self_device_memory_gib"],
        reverse=True,
    )[:top_k]

    by_cuda_time = sorted(
        (
            {
                "name": evt.key,
                "self_device_time_ms": evt.self_device_time_total / 1000,
                "device_time_ms": evt.device_time_total / 1000,
                "count": evt.count,
            }
            for evt in events
            if evt.self_device_time_total > 0
        ),
        key=lambda row: row["self_device_time_ms"],
        reverse=True,
    )[:top_k]

    summary_lines = prof.key_averages().table(
        sort_by="self_device_memory_usage",
        row_limit=top_k,
        top_level_events_only=True,
    )

    return {
        "phases": phases,
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
        "top_by_self_device_memory": by_cuda_memory,
        "top_by_self_device_time_ms": by_cuda_time,
        "table_self_device_memory": summary_lines,
        "memory_summary": torch.cuda.memory_summary(device=device, abbreviated=False),
    }


def dump_cuda_memory_snapshot_pickle(path: str, ctx: TrainStepContext) -> None:
    """Dump a PyTorch CUDA memory snapshot for https://pytorch.org/memory_viz."""
    if not hasattr(torch.cuda.memory, "_record_memory_history"):
        raise RuntimeError("torch.cuda.memory._record_memory_history is unavailable in this PyTorch build.")

    snapshot_dir = os.path.dirname(path)
    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)

    device = ctx.device
    torch.cuda.synchronize(device)
    torch.cuda.memory._record_memory_history(max_entries=200_000)
    try:
        ctx.optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(ctx)
        loss.backward()
        if ctx.use_fsdp:
            fsdp_on_after_backward(ctx.model, ctx.optimizer)
        ctx.optimizer.step()
        torch.cuda.synchronize(device)
        torch.cuda.memory._dump_snapshot(path)
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


def init_distributed(world_size: int, rank: int) -> torch.device:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return torch.device(f"cuda:{local_rank}")


def shard_batch(
    labels: torch.Tensor,
    targets: torch.Tensor,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if world_size == 1:
        return labels, targets
    local_bs = labels.shape[0] // world_size
    start = rank * local_bs
    stop = start + local_bs
    return labels[start:stop], targets[start:stop]


def build_train_context(
    cfg: Config,
    device: torch.device,
    attention: str,
    checkpoint_every: int,
    compile_model: bool,
    compile_mode: str,
    use_fsdp: bool,
    fused_ce: bool = False,
    fused_ce_chunk_size: int = 512,
) -> TrainStepContext:
    labels = torch.randint(
        high=cfg.vocab_size,
        size=(cfg.batch_size, cfg.ctx_len),
        device=device,
    )
    targets = torch.randint(
        high=cfg.vocab_size,
        size=(cfg.batch_size, cfg.ctx_len),
        device=device,
    )
    if use_fsdp:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        labels, targets = shard_batch(labels, targets, rank, world_size)

    model = build_model(cfg, device, attention=attention, checkpoint_every=checkpoint_every)
    if use_fsdp:
        model = get_fsdp(model, compute_dtype=cfg.torch_dtype)

    forward_hidden_fn = None
    base = _unwrap_model(model)
    if compile_model:
        if fused_ce and hasattr(base, "forward_hidden"):
            print(f"Compiling forward_hidden with torch.compile(mode={compile_mode!r})...")
            forward_hidden_fn = torch.compile(base.forward_hidden, mode=compile_mode)
        else:
            print(f"Compiling model with torch.compile(mode={compile_mode!r})...")
            model = torch.compile(model, mode=compile_mode)
    elif fused_ce and hasattr(base, "forward_hidden"):
        forward_hidden_fn = base.forward_hidden

    optimizer = AdamW(model.parameters())
    return TrainStepContext(
        model=model,
        optimizer=optimizer,
        labels=labels,
        targets=targets,
        device=device,
        use_fsdp=use_fsdp,
        fused_ce=fused_ce,
        fused_ce_chunk_size=fused_ce_chunk_size,
        forward_hidden_fn=forward_hidden_fn,
    )


def build_model(cfg: Config, device: torch.device, attention: str, checkpoint_every: int) -> BasicsTransformerLM:
    kwargs = dict(
        vocab_size=cfg.vocab_size,
        context_length=cfg.ctx_len,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
    )
    if attention == "cute":
        model = build_flash_basics_transformer_lm(checkpoint_every=checkpoint_every, **kwargs)
    elif attention == "naive":
        model = BasicsTransformerLM(**kwargs)
    else:
        raise ValueError(f"Unknown attention backend: {attention}")
    return model.to(device=device, dtype=cfg.torch_dtype)


def test_timing_forward_backward(
    cfg: Config,
    device: torch.device,
    warmup_ms: int = 10_000,
    rep_ms: int = 30_000,
    *,
    compile_model: bool = False,
    compile_mode: str = "default",
    attention: str = "naive",
    checkpoint_every: int = 0,
    use_fsdp: bool = False,
    fused_ce: bool = False,
    fused_ce_chunk_size: int = 512,
) -> tuple[float, TrainStepContext]:
    ctx = build_train_context(
        cfg,
        device,
        attention=attention,
        checkpoint_every=checkpoint_every,
        compile_model=compile_model,
        compile_mode=compile_mode,
        use_fsdp=use_fsdp,
        fused_ce=fused_ce,
        fused_ce_chunk_size=fused_ce_chunk_size,
    )

    def train_step() -> None:
        ctx.optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(ctx)
        loss.backward()
        if ctx.use_fsdp:
            fsdp_on_after_backward(ctx.model, ctx.optimizer)
        ctx.optimizer.step()

    # One explicit step before do_bench so we fail fast on OOM.
    train_step()
    torch.cuda.synchronize()

    timing_results = triton.testing.do_bench(train_step, rep=rep_ms, warmup=warmup_ms)
    return timing_results, ctx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leaderboard training-step timing harness.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ctx-len", type=int, default=None, help="Override ctx_len for smoke tests.")
    parser.add_argument("--warmup-ms", type=int, default=10_000)
    parser.add_argument("--rep-ms", type=int, default=30_000)
    parser.add_argument("--quick", action="store_true", help="Use warmup=1000ms, rep=3000ms.")
    parser.add_argument(
        "--attention",
        choices=["naive", "cute"],
        default="naive",
        help="Attention backend: naive cs336_basics or FA4 CuTeDSL (cute).",
    )
    parser.add_argument("--compile", action="store_true", help="Wrap model with torch.compile before benchmarking.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Gradient-checkpoint every N layers (0 disables). Needed for cute @ ctx=32768.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    )
    parser.add_argument(
        "--fused-ce",
        action="store_true",
        help="Fuse LM head with cross-entropy (chunked; avoids full logits tensor). Requires --attention cute.",
    )
    parser.add_argument(
        "--fused-ce-chunk-size",
        type=int,
        default=512,
        help="Token chunk size for fused cross-entropy.",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Use 2-GPU FSDP (launch with torchrun --nproc_per_node=2, or auto-spawn).",
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Run torch.profiler with profile_memory=True on one training step after timing.",
    )
    parser.add_argument(
        "--memory-snapshot",
        type=str,
        default=None,
        help="Optional path to dump a PyTorch CUDA memory snapshot pickle (view at pytorch.org/memory_viz).",
    )
    parser.add_argument(
        "--profiler-top-k",
        type=int,
        default=20,
        help="Number of top profiler rows to include in JSON output.",
    )
    return parser.parse_args()


def _distributed_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    device = init_distributed(world_size, rank)
    if rank == 0:
        torch.cuda.empty_cache()
        gc.collect()
    dist.barrier()
    run_benchmark(args, device, rank=rank, world_size=world_size)


def run_benchmark(
    args: argparse.Namespace,
    device: torch.device,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    warmup_ms = 1_000 if args.quick else args.warmup_ms
    rep_ms = 3_000 if args.quick else args.rep_ms

    cfg = Config()
    if args.ctx_len is not None:
        cfg.ctx_len = args.ctx_len

    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    is_main = rank == 0
    if is_main:
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        if world_size > 1:
            print(f"distributed: {world_size} GPUs, rank {rank}, backend=nccl")
        print(f"Config: batch={cfg.batch_size}, ctx_len={cfg.ctx_len}, dtype={cfg.torch_dtype}")
        print(f"do_bench warmup={warmup_ms}ms, rep={rep_ms}ms")
        print(f"attention: {args.attention}")
        if args.checkpoint_every:
            print(f"checkpoint_every: {args.checkpoint_every}")
        print(f"fsdp: {args.fsdp or world_size > 1}")
        print(f"fused_ce: {args.fused_ce}" + (f" (chunk={args.fused_ce_chunk_size})" if args.fused_ce else ""))
        print(f"torch.compile: {args.compile}" + (f" (mode={args.compile_mode})" if args.compile else ""))

    if args.fused_ce and args.attention != "cute":
        raise ValueError("--fused-ce requires --attention cute")

    checkpoint_every = args.checkpoint_every
    if is_main and args.fsdp and checkpoint_every == 0 and not args.fused_ce:
        print("fsdp: no checkpointing — may OOM at ctx=32768 without fused CE")

    timing_ms, train_ctx = test_timing_forward_backward(
        cfg,
        device,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        attention=args.attention,
        checkpoint_every=checkpoint_every,
        use_fsdp=args.fsdp or world_size > 1,
        fused_ce=args.fused_ce,
        fused_ce_chunk_size=args.fused_ce_chunk_size,
    )

    memory_profile = None
    if args.profile_memory and is_main:
        print("\nRunning PyTorch memory profiler on one training step...")
        memory_profile = profile_train_step_memory(train_ctx, top_k=args.profiler_top_k)

    if args.memory_snapshot and is_main:
        dump_cuda_memory_snapshot_pickle(args.memory_snapshot, train_ctx)

    alloc = torch.cuda.max_memory_allocated(device) / 1024**3
    reserved = torch.cuda.max_memory_reserved(device) / 1024**3

    if is_main:
        if memory_profile:
            print("\n=== Memory by phase ===")
            for phase in memory_profile["phases"]:
                print(
                    f"  {phase['label']:24s}  allocated={phase['allocated_gib']:.2f} GiB  "
                    f"peak={phase['peak_allocated_gib']:.2f} GiB"
                )
        print(f"\n=== Training step time: {timing_ms:.1f} ms ({timing_ms / 1000:.3f} s) ===")
        print(f"Peak GPU memory (rank 0): allocated={alloc:.2f} GiB, reserved={reserved:.2f} GiB")

        result = {
            "training_step_ms": timing_ms,
            "training_step_s": timing_ms / 1000,
            "warmup_ms": warmup_ms,
            "rep_ms": rep_ms,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "world_size": world_size,
            "fsdp": args.fsdp or world_size > 1,
            "peak_memory_allocated_gib": alloc,
            "peak_memory_reserved_gib": reserved,
            "compiled": args.compile,
            "compile_mode": args.compile_mode if args.compile else None,
            "attention": args.attention,
            "checkpoint_every": checkpoint_every,
            "fused_ce": args.fused_ce,
            "fused_ce_chunk_size": args.fused_ce_chunk_size if args.fused_ce else None,
            "memory_profile": memory_profile,
            "memory_snapshot_path": args.memory_snapshot,
            "config": {
                "ctx_len": cfg.ctx_len,
                "vocab_size": cfg.vocab_size,
                "d_model": cfg.d_model,
                "d_ff": cfg.d_ff,
                "num_layers": cfg.num_layers,
                "num_heads": cfg.num_heads,
                "batch_size": cfg.batch_size,
                "dtype": str(cfg.torch_dtype),
                "is_causal": cfg.is_causal,
            },
            "leaderboard_reference_ms": {
                "naive_baseline": 10_000,
                "top_keshav": 3_837,
                "median_daphne": 6_606,
            },
        }
        print(json.dumps(result, indent=2))
        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if args.fsdp:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            run_benchmark(args, device, rank=rank, world_size=world_size)
        else:
            world_size = min(args.world_size, torch.cuda.device_count())
            if world_size < 2:
                raise RuntimeError("FSDP requires at least 2 GPUs.")
            mp.spawn(_distributed_worker, args=(world_size, args), nprocs=world_size, join=True)
        return

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    gc.collect()
    run_benchmark(args, device)


if __name__ == "__main__":
    main()
