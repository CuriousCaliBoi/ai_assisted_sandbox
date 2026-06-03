"""SonicMoE ~7B inference timing harness (prefill + single-step decode)."""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass

import torch
import triton.testing

from cs336_moe.model import SonicMoETransformerLM, build_olmoe_7b_lm
from sonicmoe import KernelBackendMoE


class Config:
    ctx_len = 32768
    vocab_size = 151936
    batch_size = 1
    torch_dtype = torch.bfloat16


@dataclass
class InferenceContext:
    model: SonicMoETransformerLM
    input_ids: torch.Tensor
    device: torch.device
    include_lm_head: bool


def build_context(
    cfg: Config,
    device: torch.device,
    *,
    seq_len: int,
    batch_size: int,
    kernel_backend: KernelBackendMoE,
    compile_model: bool = False,
    compile_mode: str = "default",
    include_lm_head: bool = True,
) -> InferenceContext:
    input_ids = torch.randint(
        high=cfg.vocab_size,
        size=(batch_size, seq_len),
        device=device,
    )
    model = build_olmoe_7b_lm(
        vocab_size=cfg.vocab_size,
        context_length=max(cfg.ctx_len, seq_len),
        kernel_backend_moe=kernel_backend,
    ).to(device=device, dtype=cfg.torch_dtype)
    model.eval()

    if compile_model:
        if include_lm_head:
            model = torch.compile(model, mode=compile_mode)
        else:
            model.forward_hidden = torch.compile(model.forward_hidden, mode=compile_mode)

    return InferenceContext(
        model=model,
        input_ids=input_ids,
        device=device,
        include_lm_head=include_lm_head,
    )


def warmup(ctx: InferenceContext, steps: int = 5) -> None:
    with torch.inference_mode():
        for _ in range(steps):
            if ctx.include_lm_head:
                ctx.model(ctx.input_ids)
            else:
                ctx.model.forward_hidden(ctx.input_ids)
    torch.cuda.synchronize(ctx.device)


def bench_forward(ctx: InferenceContext, warmup_ms: int, rep_ms: int) -> float:
    def run() -> None:
        with torch.inference_mode():
            if ctx.include_lm_head:
                ctx.model(ctx.input_ids)
            else:
                ctx.model.forward_hidden(ctx.input_ids)

    warmup(ctx, steps=3)
    torch.cuda.synchronize(ctx.device)
    return triton.testing.do_bench(run, warmup=warmup_ms, rep=rep_ms)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SonicMoE inference benchmark.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--max-ctx", type=int, default=32768)
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--rep-ms", type=int, default=3000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--kernel-backend", choices=["sonicmoe", "torch"], default="sonicmoe")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    )
    parser.add_argument(
        "--no-lm-head",
        action="store_true",
        help="Benchmark transformer hidden states only (skip vocab projection).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run prefill sweep at seq 512, 2048, 8192, 32768.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def run_one(
    args: argparse.Namespace,
    device: torch.device,
    *,
    seq_len: int,
    batch_size: int,
    include_lm_head: bool,
    warmup_ms: int,
    rep_ms: int,
) -> dict:
    cfg = Config()
    cfg.ctx_len = args.max_ctx
    kernel_backend = KernelBackendMoE[args.kernel_backend]

    torch.cuda.reset_peak_memory_stats(device)
    ctx = build_context(
        cfg,
        device,
        seq_len=seq_len,
        batch_size=batch_size,
        kernel_backend=kernel_backend,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        include_lm_head=include_lm_head,
    )

    latency_ms = bench_forward(ctx, warmup_ms=warmup_ms, rep_ms=rep_ms)
    tokens = batch_size * seq_len
    tok_per_sec = tokens / (latency_ms / 1000)
    alloc = torch.cuda.max_memory_allocated(device) / 1024**3

    label = "prefill+lm_head" if include_lm_head else "prefill_hidden_only"
    if seq_len == 1:
        label = "decode_step" + ("+lm_head" if include_lm_head else "_hidden_only")

    return {
        "mode": label,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_forward": tokens,
        "latency_ms": latency_ms,
        "tok_per_sec": tok_per_sec,
        "ms_per_token": latency_ms / max(tokens, 1),
        "peak_memory_allocated_gib": alloc,
        "include_lm_head": include_lm_head,
        "compiled": args.compile,
    }


def run_benchmark(args: argparse.Namespace, device: torch.device) -> None:
    warmup_ms = 1_000 if args.quick else args.warmup_ms
    rep_ms = 3_000 if args.quick else args.rep_ms
    include_lm_head = not args.no_lm_head

    torch.cuda.set_device(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("SonicMoE OLMoE-7B inference (model.eval(), MoE is_inference_mode=True)")
    print(f"batch={args.batch_size}, seq={args.seq_len}, lm_head={include_lm_head}, compile={args.compile}")

    results: list[dict] = []

    if args.sweep:
        for seq in [512, 2048, 8192, 32768]:
            if seq > args.max_ctx:
                continue
            print(f"\n--- seq_len={seq} ---")
            row = run_one(
                args,
                device,
                seq_len=seq,
                batch_size=args.batch_size,
                include_lm_head=include_lm_head,
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
            )
            results.append(row)
            print(
                f"  {row['latency_ms']:.2f} ms  |  {row['tok_per_sec']/1000:.1f}K tok/s  |  "
                f"{row['peak_memory_allocated_gib']:.1f} GiB"
            )
        # Single-token step (no KV cache — not production decode, but shows per-step floor)
        print("\n--- decode-style seq_len=1 (no KV cache) ---")
        decode_row = run_one(
            args,
            device,
            seq_len=1,
            batch_size=args.batch_size,
            include_lm_head=include_lm_head,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        )
        results.append(decode_row)
        print(
            f"  {decode_row['latency_ms']:.2f} ms/step  |  "
            f"{1000/decode_row['latency_ms']:.0f} steps/s  |  "
            f"{decode_row['peak_memory_allocated_gib']:.1f} GiB"
        )
    else:
        row = run_one(
            args,
            device,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            include_lm_head=include_lm_head,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        )
        results.append(row)
        print(f"\nLatency: {row['latency_ms']:.2f} ms")
        print(f"Throughput: {row['tok_per_sec']/1000:.1f}K tok/s")
        print(f"Peak VRAM: {row['peak_memory_allocated_gib']:.1f} GiB")

    payload = {
        "gpu_name": torch.cuda.get_device_name(device),
        "kernel_backend": args.kernel_backend,
        "results": results,
    }
    print(json.dumps(payload, indent=2))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    gc.collect()
    run_benchmark(args, device)


if __name__ == "__main__":
    main()
