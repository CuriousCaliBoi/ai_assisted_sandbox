"""SonicMoE ~8B training-step timing harness."""

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
from torch.nn.parallel import DistributedDataParallel as DDP

from cs336_basics.nn_utils import cross_entropy
from cs336_moe.model import SonicMoETransformerLM, build_olmoe_7b_lm
from cs336_systems.fused_adamw import build_adamw
from cs336_systems.fused_ce import fused_lm_head_cross_entropy
from sonicmoe import KernelBackendMoE


class Config:
    ctx_len = 32768
    vocab_size = 151936
    batch_size = 2
    torch_dtype = torch.bfloat16
    aux_loss_coef = 0.01

    # OLMoE-style ~7-8B preset (SonicMoE blog benchmark)
    hidden_size = 2048
    num_layers = 16
    num_heads = 16
    intermediate_size = 1024
    num_experts = 64
    num_experts_per_tok = 8


@dataclass
class TrainStepContext:
    model: SonicMoETransformerLM
    optimizer: torch.optim.Optimizer
    labels: torch.Tensor
    targets: torch.Tensor
    device: torch.device
    aux_loss_coef: float
    fused_ce: bool = False
    fused_ce_chunk_size: int = 512
    forward_hidden_fn: object | None = None
    use_ddp: bool = False


def _unwrap_model(model: torch.nn.Module) -> SonicMoETransformerLM:
    if isinstance(model, DDP):
        return model.module
    return model


def init_distributed(world_size: int, rank: int) -> torch.device:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
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


def compute_loss(ctx: TrainStepContext) -> torch.Tensor:
    base = _unwrap_model(ctx.model)
    if ctx.fused_ce:
        if ctx.use_ddp:
            hidden, aux = ctx.model(ctx.labels, return_hidden=True, return_aux_loss=True)
        else:
            forward_hidden = ctx.forward_hidden_fn or base.forward_hidden
            hidden = forward_hidden(ctx.labels)
            aux = base.load_balance_loss()
        ce = fused_lm_head_cross_entropy(
            hidden,
            base.lm_head,
            ctx.targets,
            chunk_size=ctx.fused_ce_chunk_size,
        )
        return (ce + ctx.aux_loss_coef * aux).sum()

    logits, aux_loss = ctx.model(ctx.labels, return_aux_loss=True)
    ce = cross_entropy(logits, ctx.targets)
    return (ce + ctx.aux_loss_coef * aux_loss).sum()


def build_train_context(
    cfg: Config,
    device: torch.device,
    *,
    kernel_backend: KernelBackendMoE = KernelBackendMoE.sonicmoe,
    checkpoint_every: int = 0,
    fused_ce: bool = False,
    fused_ce_chunk_size: int = 512,
    compile_model: bool = False,
    compile_mode: str = "default",
    use_ddp: bool = False,
    fused_adamw: bool = True,
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
    if use_ddp:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        labels, targets = shard_batch(labels, targets, rank, world_size)

    model = build_olmoe_7b_lm(
        vocab_size=cfg.vocab_size,
        context_length=cfg.ctx_len,
        kernel_backend_moe=kernel_backend,
        checkpoint_every=checkpoint_every,
    ).to(device=device, dtype=cfg.torch_dtype)
    model.train()

    forward_hidden_fn = None
    if compile_model:
        if fused_ce and not use_ddp:
            forward_hidden_fn = torch.compile(model.forward_hidden, mode=compile_mode)
        else:
            model = torch.compile(model, mode=compile_mode)
    elif fused_ce:
        forward_hidden_fn = model.forward_hidden

    if use_ddp:
        model = DDP(model, device_ids=[device.index])

    optimizer = build_adamw(model.parameters(), fused=fused_adamw)
    return TrainStepContext(
        model=model,
        optimizer=optimizer,
        labels=labels,
        targets=targets,
        device=device,
        aux_loss_coef=cfg.aux_loss_coef,
        fused_ce=fused_ce,
        fused_ce_chunk_size=fused_ce_chunk_size,
        forward_hidden_fn=forward_hidden_fn,
        use_ddp=use_ddp,
    )


def warmup_kernels(ctx: TrainStepContext, steps: int = 3) -> None:
    """Prime SonicMoE / QuACK autotuning before timed runs."""
    for _ in range(steps):
        ctx.optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(ctx)
        loss.backward()
        ctx.optimizer.step()
    torch.cuda.synchronize(ctx.device)


def test_timing_forward_backward(
    cfg: Config,
    device: torch.device,
    warmup_ms: int = 10_000,
    rep_ms: int = 30_000,
    *,
    kernel_backend: KernelBackendMoE = KernelBackendMoE.sonicmoe,
    checkpoint_every: int = 0,
    fused_ce: bool = False,
    fused_ce_chunk_size: int = 512,
    compile_model: bool = False,
    compile_mode: str = "default",
    kernel_warmup_steps: int = 3,
    use_ddp: bool = False,
    fused_adamw: bool = True,
) -> tuple[float, TrainStepContext]:
    ctx = build_train_context(
        cfg,
        device,
        kernel_backend=kernel_backend,
        checkpoint_every=checkpoint_every,
        fused_ce=fused_ce,
        fused_ce_chunk_size=fused_ce_chunk_size,
        compile_model=compile_model,
        compile_mode=compile_mode,
        use_ddp=use_ddp,
        fused_adamw=fused_adamw,
    )
    warmup_kernels(ctx, steps=kernel_warmup_steps)

    def train_step() -> None:
        ctx.optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(ctx)
        loss.backward()
        ctx.optimizer.step()

    train_step()
    torch.cuda.synchronize(device)

    timing_ms = triton.testing.do_bench(train_step, rep=rep_ms, warmup=warmup_ms)
    return timing_ms, ctx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SonicMoE ~8B training-step timing harness.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ctx-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Global batch size (must divide world size under DDP).")
    parser.add_argument("--warmup-ms", type=int, default=10_000)
    parser.add_argument("--rep-ms", type=int, default=30_000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--kernel-backend",
        choices=["sonicmoe", "torch"],
        default="sonicmoe",
        help="SonicMoE FFN kernel backend.",
    )
    parser.add_argument("--aux-loss-coef", type=float, default=None)
    parser.add_argument(
        "--fused-ce",
        action="store_true",
        help="Fuse LM head with cross-entropy (chunked; avoids full logits tensor).",
    )
    parser.add_argument("--fused-ce-chunk-size", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    )
    parser.add_argument(
        "--kernel-warmup-steps",
        type=int,
        default=3,
        help="Untimed train steps before do_bench (SonicMoE autotune).",
    )
    parser.add_argument(
        "--fused-adamw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use PyTorch fused CUDA AdamW (default: on).",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Use 2-GPU DDP (auto-spawns 2 processes, splits batch across GPUs).",
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--output-json", type=str, default=None)
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
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.aux_loss_coef is not None:
        cfg.aux_loss_coef = args.aux_loss_coef

    kernel_backend = KernelBackendMoE[args.kernel_backend]
    use_ddp = args.ddp or world_size > 1
    if use_ddp and cfg.batch_size % world_size != 0:
        raise ValueError(
            f"batch_size={cfg.batch_size} must be divisible by world_size={world_size} for DDP"
        )
    is_main = rank == 0

    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    if is_main:
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        if world_size > 1:
            print(f"distributed: {world_size} GPUs, rank {rank}, backend=nccl (DDP)")
        print("Preset: OLMoE-style ~7-8B SonicMoE + FA4 CuTeDSL attention")
        print(f"Config: batch={cfg.batch_size}, ctx_len={cfg.ctx_len}, dtype={cfg.torch_dtype}")
        print(
            f"Model: layers={cfg.num_layers}, H={cfg.hidden_size}, I={cfg.intermediate_size}, "
            f"E={cfg.num_experts}, top_k={cfg.num_experts_per_tok}, heads={cfg.num_heads}"
        )
        print(f"attention: cute (FA4), kernel_backend: {args.kernel_backend}")
        print(f"aux_loss_coef: {cfg.aux_loss_coef}")
        print(f"fused_ce: {args.fused_ce}" + (f" (chunk={args.fused_ce_chunk_size})" if args.fused_ce else ""))
        if args.checkpoint_every:
            print(f"checkpoint_every: {args.checkpoint_every}")
        print(f"ddp: {use_ddp}")
        print(f"fused_adamw: {args.fused_adamw}")
        print(f"torch.compile: {args.compile}" + (f" (mode={args.compile_mode})" if args.compile else ""))
        print(f"kernel_warmup_steps: {args.kernel_warmup_steps}")
        print(f"do_bench warmup={warmup_ms}ms, rep={rep_ms}ms")

    if is_main and args.compile and args.fused_ce:
        print(f"Compiling forward_hidden with torch.compile(mode={args.compile_mode!r})...")
    elif is_main and args.compile:
        print(f"Compiling model with torch.compile(mode={args.compile_mode!r})...")

    timing_ms, ctx = test_timing_forward_backward(
        cfg,
        device,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
        kernel_backend=kernel_backend,
        checkpoint_every=args.checkpoint_every,
        fused_ce=args.fused_ce,
        fused_ce_chunk_size=args.fused_ce_chunk_size,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        kernel_warmup_steps=args.kernel_warmup_steps,
        use_ddp=use_ddp,
        fused_adamw=args.fused_adamw,
    )

    alloc = torch.cuda.max_memory_allocated(device) / 1024**3
    reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    num_params_b = _unwrap_model(ctx.model).get_num_params() / 1e9

    if is_main:
        print(f"\n=== Training step time: {timing_ms:.1f} ms ({timing_ms / 1000:.3f} s) ===")
        tokens_per_step = cfg.batch_size * cfg.ctx_len
        tok_per_s = tokens_per_step / (timing_ms / 1000)
        print(f"Throughput: {tok_per_s:,.0f} tok/s ({tokens_per_step:,} tok/step)")
        print(f"Model params: {num_params_b:.2f}B (replicated per GPU under DDP)")
        print(f"Peak GPU memory (rank 0): allocated={alloc:.2f} GiB, reserved={reserved:.2f} GiB")

        result = {
            "training_step_ms": timing_ms,
            "training_step_s": timing_ms / 1000,
            "tokens_per_step": cfg.batch_size * cfg.ctx_len,
            "throughput_tok_per_s": (cfg.batch_size * cfg.ctx_len) / (timing_ms / 1000),
            "warmup_ms": warmup_ms,
            "rep_ms": rep_ms,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "world_size": world_size,
            "ddp": use_ddp,
            "peak_memory_allocated_gib": alloc,
            "peak_memory_reserved_gib": reserved,
            "model_type": "sonic_moe",
            "preset": "olmoe-7b",
            "num_params_b": num_params_b,
            "attention": "cute",
            "kernel_backend": args.kernel_backend,
            "aux_loss_coef": cfg.aux_loss_coef,
            "fused_ce": args.fused_ce,
            "fused_ce_chunk_size": args.fused_ce_chunk_size if args.fused_ce else None,
            "checkpoint_every": args.checkpoint_every,
            "compiled": args.compile,
            "compile_mode": args.compile_mode if args.compile else None,
            "fused_adamw": args.fused_adamw,
            "config": {
                "ctx_len": cfg.ctx_len,
                "vocab_size": cfg.vocab_size,
                "hidden_size": cfg.hidden_size,
                "intermediate_size": cfg.intermediate_size,
                "num_layers": cfg.num_layers,
                "num_heads": cfg.num_heads,
                "num_experts": cfg.num_experts,
                "num_experts_per_tok": cfg.num_experts_per_tok,
                "batch_size": cfg.batch_size,
                "dtype": str(cfg.torch_dtype),
            },
            "single_gpu_reference_ms": {
                "fused_ce_compile_ctx32768_quick": 1146,
                "fused_ce_ctx32768_quick": 1322,
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

    if args.ddp:
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
                raise RuntimeError("DDP requires at least 2 GPUs.")
            mp.spawn(_distributed_worker, args=(world_size, args), nprocs=world_size, join=True)
        return

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    gc.collect()
    run_benchmark(args, device)


if __name__ == "__main__":
    main()
