#!/usr/bin/env python3
"""Fine-tune / continue-train OLMoE with SonicMoE FFN + FA4 attention."""

from __future__ import annotations

import argparse
import gc
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from cs336_basics.optimizer import AdamW
from cs336_moe.benchmark import (
    Config,
    compute_loss,
    init_distributed,
    shard_batch,
    warmup_kernels,
    TrainStepContext,
)
from cs336_moe.load_hf import OLMOE_7B_INSTRUCT_REPO, OLMOE_7B_REPO, build_olmoe_7b_from_hf, olmoe_hf_config
from cs336_moe.model import build_olmoe_7b_lm
from cs336_systems.fused_adamw import build_adamw
from sonicmoe import KernelBackendMoE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train OLMoE with SonicMoE kernels.")
    p.add_argument(
        "--hf-repo",
        default=OLMOE_7B_REPO,
        help=f"HF repo (default: {OLMOE_7B_REPO}; instruct: {OLMOE_7B_INSTRUCT_REPO})",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ctx-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--aux-loss-coef", type=float, default=0.01)
    p.add_argument("--kernel-backend", choices=["sonicmoe", "torch"], default="sonicmoe")
    p.add_argument("--fused-ce", action="store_true")
    p.add_argument("--fused-ce-chunk-size", type=int, default=512)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    )
    p.add_argument(
        "--fused-adamw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use PyTorch fused CUDA AdamW (default: on).",
    )
    p.add_argument(
        "--ddp",
        action="store_true",
        help="Use 2-GPU DDP (auto-spawns 2 processes, splits batch across GPUs).",
    )
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--kernel-warmup-steps", type=int, default=2)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--skip-load", action="store_true", help="Random init only (smoke test).")
    return p.parse_args()


def build_train_context(
    args: argparse.Namespace,
    cfg: Config,
    device: torch.device,
    *,
    rank: int = 0,
    world_size: int = 1,
    use_ddp: bool = False,
) -> TrainStepContext:
    dtype = cfg.torch_dtype
    kernel_backend = KernelBackendMoE[args.kernel_backend]

    if args.skip_load:
        model = build_olmoe_7b_lm(
            vocab_size=cfg.vocab_size,
            context_length=cfg.ctx_len,
            kernel_backend_moe=kernel_backend,
            checkpoint_every=args.checkpoint_every,
            olmoe_attention=True,
        ).to(device=device, dtype=dtype)
    else:
        hf_cfg = olmoe_hf_config(args.hf_repo)
        model = build_olmoe_7b_from_hf(
            args.hf_repo,
            context_length=max(cfg.ctx_len, hf_cfg.get("max_position_embeddings", 4096)),
            kernel_backend_moe=kernel_backend,
            checkpoint_every=args.checkpoint_every,
            device=device,
            dtype=dtype,
        )

    model.train()

    forward_hidden_fn = None
    if args.compile:
        if args.fused_ce and not use_ddp:
            forward_hidden_fn = torch.compile(model.forward_hidden, mode=args.compile_mode)
        else:
            model = torch.compile(model, mode=args.compile_mode)
    elif args.fused_ce:
        forward_hidden_fn = model.forward_hidden

    if use_ddp:
        model = DDP(model, device_ids=[device.index])

    labels = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.ctx_len), device=device)
    targets = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.ctx_len), device=device)
    if use_ddp:
        labels, targets = shard_batch(labels, targets, rank, world_size)

    optimizer = build_adamw(model.parameters(), lr=args.lr, fused=args.fused_adamw)
    return TrainStepContext(
        model=model,
        optimizer=optimizer,
        labels=labels,
        targets=targets,
        device=device,
        aux_loss_coef=cfg.aux_loss_coef,
        fused_ce=args.fused_ce,
        fused_ce_chunk_size=args.fused_ce_chunk_size,
        forward_hidden_fn=forward_hidden_fn,
        use_ddp=use_ddp,
    )


def run_training(
    args: argparse.Namespace,
    device: torch.device,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    use_ddp = args.ddp or world_size > 1
    is_main = rank == 0

    hf_cfg = olmoe_hf_config(args.hf_repo)
    cfg = Config()
    cfg.ctx_len = args.ctx_len
    cfg.batch_size = args.batch_size
    cfg.vocab_size = hf_cfg["vocab_size"]
    cfg.aux_loss_coef = args.aux_loss_coef

    if is_main:
        print(f"HF repo: {args.hf_repo}")
        print(f"vocab={cfg.vocab_size}, ctx={cfg.ctx_len}, batch={cfg.batch_size}")
        print(
            f"kernel={args.kernel_backend}, fused_ce={args.fused_ce}, "
            f"fused_adamw={args.fused_adamw}, compile={args.compile}, ddp={use_ddp}"
        )
        if world_size > 1:
            print(f"distributed: {world_size} GPUs, rank {rank}, backend=nccl (DDP)")

    if is_main and args.skip_load:
        print("Using random init (--skip-load).")
    elif is_main:
        print("Loading HF weights (first run may take several minutes)...")

    torch.cuda.set_device(device)
    if is_main:
        torch.cuda.empty_cache()
        gc.collect()
    if use_ddp:
        dist.barrier()

    ctx = build_train_context(
        args,
        cfg,
        device,
        rank=rank,
        world_size=world_size,
        use_ddp=use_ddp,
    )

    if is_main and args.compile and args.fused_ce and not use_ddp:
        print(f"Compiling forward_hidden with torch.compile(mode={args.compile_mode!r})...")
    elif is_main and args.compile:
        print(f"Compiling model with torch.compile(mode={args.compile_mode!r})...")

    if args.kernel_warmup_steps > 0 and is_main:
        print(f"Kernel warmup ({args.kernel_warmup_steps} steps)...")
    if args.kernel_warmup_steps > 0:
        warmup_kernels(ctx, steps=args.kernel_warmup_steps)

    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        ctx.optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(ctx)
        loss.backward()
        ctx.optimizer.step()
        if is_main and step % args.log_every == 0:
            print(f"step {step}/{args.steps}  loss={loss.item():.4f}")

    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    if is_main:
        print(f"Done. Peak VRAM (rank 0): {peak:.2f} GiB")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _distributed_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    device = init_distributed(world_size, rank)
    if rank == 0:
        torch.cuda.empty_cache()
        gc.collect()
    dist.barrier()
    run_training(args, device, rank=rank, world_size=world_size)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    if args.ddp:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            run_training(args, device, rank=rank, world_size=world_size)
        else:
            world_size = min(args.world_size, torch.cuda.device_count())
            if world_size < 2:
                raise RuntimeError("DDP requires at least 2 GPUs.")
            mp.spawn(_distributed_worker, args=(world_size, args), nprocs=world_size, join=True)
        return

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    run_training(args, device)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
