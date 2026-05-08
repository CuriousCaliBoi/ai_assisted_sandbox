# Offline Mac Guide for CS336 Assignment 2 Systems

This repo is already usable offline on this Mac:

```sh
uv run --offline python - <<'PY'
import torch
print(torch.__version__)
print("cuda", torch.cuda.is_available())
print("mps", torch.backends.mps.is_available())
PY
```

On this laptop, the check returned PyTorch `2.11.0`, CUDA `False`, and MPS `True`. That means you can run local forward passes, backward passes, optimizer steps, timing, approximate memory sampling, CPU/Gloo distributed demos, and the math helpers without Wi-Fi. You cannot run NVIDIA-only CUDA/Nsight/Triton deliverables on this Mac unless you use a CUDA machine.

## One-Time Prep Before Going Offline

Run these while you still have Wi-Fi if this is a fresh clone or fresh machine:

```sh
cd /Users/kingzuko/projects/ai_assisted_sandbox
uv sync
uv run python -c "import torch, cs336_basics; print(torch.__version__)"
```

Then verify the cached environment works without the network:

```sh
uv run --offline python -c "import torch, cs336_basics; print('offline ok')"
```

If that command succeeds, you can disconnect from Wi-Fi and keep using `uv run --offline ...`.

## Flight Quickstart

When you are offline, start every session from the repo root:

```sh
cd /Users/kingzuko/projects/ai_assisted_sandbox
```

Then run this quick check:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 32 \
  --warmup 1 \
  --steps 1 \
  --mode train
```

If that works, use the rest of this section as a menu.

### Example 1: Get Forward/Backward/Optimizer Timings

Use this when answering the Section 2.1.3 benchmarking questions:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 128 \
  --warmup 5 \
  --steps 10 \
  --mode train
```

Look in the JSON output under:

- `timings.forward_s.mean_s`
- `timings.backward_s.mean_s`
- `timings.optimizer_s.mean_s`
- `timings.total_s.mean_s`

To compare warmup behavior, rerun the same command with `--warmup 0`, then with `--warmup 1`, then with `--warmup 2`.

### Example 2: Save Results To Files

Make an output folder once:

```sh
mkdir -p outputs
```

Then save each run:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 128 \
  --warmup 5 \
  --steps 10 \
  --mode train \
  --output-json outputs/tiny_ctx128_train.json
```

Use descriptive filenames like `tiny_ctx128_train.json`, `tiny_ctx128_forward.json`, and `tiny_ctx256_train_bf16.json`.

### Example 3: Compare Forward-Only vs Full Training Memory

Forward-only inference-style memory:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 256 \
  --warmup 2 \
  --steps 5 \
  --mode forward \
  --no-grad-forward \
  --profile-memory \
  --output-json outputs/tiny_ctx256_forward_memory.json
```

Full training memory:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 256 \
  --warmup 2 \
  --steps 5 \
  --mode train \
  --profile-memory \
  --profile-saved-tensors \
  --output-json outputs/tiny_ctx256_train_memory.json
```

Compare:

- `memory.peak_rss_mib`
- `memory.peak_accelerator_allocated_mib`
- `saved_tensors.average_per_step_mib`

### Example 4: Sweep Context Lengths

Use this to build a small table:

```sh
for ctx in 128 256 512; do
  uv run --offline python scripts/benchmark_transformer.py \
    --model-size tiny \
    --batch-size 1 \
    --context-length "$ctx" \
    --warmup 2 \
    --steps 5 \
    --mode train \
    --profile-memory \
    --profile-saved-tensors \
    --output-json "outputs/tiny_ctx${ctx}_train_memory.json"
done
```

If `512` is slow or crashes, stop there and use `128` and `256`.

### Example 5: Compare FP32 vs BF16

FP32:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 128 \
  --warmup 5 \
  --steps 10 \
  --mode train \
  --dtype float32 \
  --output-json outputs/tiny_fp32.json
```

BF16:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 128 \
  --warmup 5 \
  --steps 10 \
  --mode train \
  --dtype bfloat16 \
  --output-json outputs/tiny_bf16.json
```

Compare `timings.total_s.mean_s` and peak memory if you also add `--profile-memory`.

### Example 6: Compare Checkpointing

No checkpointing:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 256 \
  --mode train \
  --profile-saved-tensors \
  --output-json outputs/tiny_no_checkpoint.json
```

Checkpoint every layer:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 256 \
  --mode train \
  --checkpoint-every 1 \
  --profile-saved-tensors \
  --output-json outputs/tiny_checkpoint_every1.json
```

Checkpointing should reduce saved tensors but often makes backward slower because it recomputes work.

### Example 7: Attention Benchmarks

Run a small vanilla attention sweep:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --task attention-benchmark \
  --attention-d-models 16,32,64 \
  --attention-seq-lengths 128,256,512 \
  --attention-batch-size 1 \
  --warmup 2 \
  --steps 5 \
  --output-json outputs/attention_small.json
```

Then compare compiled attention:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --task attention-benchmark \
  --compile-attention \
  --attention-d-models 16,32,64 \
  --attention-seq-lengths 128,256,512 \
  --attention-batch-size 1 \
  --warmup 2 \
  --steps 5 \
  --output-json outputs/attention_small_compiled.json
```

### Example 8: Distributed All-Reduce On The Laptop

This uses CPU/Gloo, not GPU/NCCL, but it helps you understand Section 5:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --task distributed-all-reduce \
  --world-size 2 \
  --all-reduce-sizes-mb 1,10 \
  --warmup 2 \
  --steps 5
```

### Example 9: Written Math Helpers

Residual stream size:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --task residual-stream-size \
  --model-size xl \
  --batch-size 4 \
  --context-length 2048 \
  --dtype float32
```

Parallelism formulas:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --task parallelism-calculator
```

### If Something Goes Wrong

If a run is too slow or crashes:

1. Reduce `--context-length`.
2. Reduce `--batch-size`.
3. Use `--model-size tiny`.
4. Remove `--compile-model` or `--compile-attention`.
5. Remove `--profile-memory` if you only need timing.

Remember: CUDA/Nsight/Triton screenshots, CUDA memory snapshots, NCCL timing, and leaderboard numbers still need an NVIDIA GPU later. On the flight, focus on script correctness, small-model tables, memory intuition, checkpointing intuition, attention scaling, and written math.

## Script Coverage Map

First print the script’s own assignment-section map:

```sh
uv run --offline python scripts/benchmark_transformer.py --list-assignment-coverage
```

Use this as the source of truth for what the script covers locally and what still needs CUDA/NVIDIA later.

## Smoke Test

Start small so you know the script and environment work:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 32 \
  --warmup 1 \
  --steps 2 \
  --mode train \
  --profile-memory \
  --profile-saved-tensors
```

This prints JSON with timing means/stdevs, sampled process RSS, sampled MPS memory, and an estimate of tensors saved by autograd for backward.

## Section 2: Profiling And Benchmarking

### 2.1.1 Setup

Any script run imports `cs336_basics` and builds the assignment 1 Transformer. For a minimal import-only check:

```sh
uv run --offline python -c "import cs336_basics.model; print('basics import ok')"
```

### 2.1.2 Model Sizing

The script supports the PDF table names: `small`, `medium`, `large`, `xl`, and `10B`. It also adds `tiny` for laptop-safe debugging.

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 128 \
  --steps 1
```

### 2.1.3 End-To-End Benchmarking

Forward only:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 128 \
  --warmup 5 \
  --steps 10 \
  --mode forward
```

For inference-style forward memory, add `--no-grad-forward` so PyTorch does not save activations for backward.

Forward plus backward:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 128 \
  --warmup 5 \
  --steps 10 \
  --mode forward_backward
```

Full training step:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 128 \
  --warmup 5 \
  --steps 10 \
  --mode train
```

The assignment table uses `small`, `medium`, `large`, `xl`, and `10B`, but many of those are too large for a laptop. Use `tiny` while developing the workflow, then try `small` with shorter context lengths:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size small \
  --batch-size 1 \
  --context-length 128 \
  --warmup 2 \
  --steps 5 \
  --mode train
```

If the process runs out of memory, reduce `--context-length`, then `--batch-size`, then use `--model-size tiny`.

### 2.1.4 Nsight Systems Profiler

Nsight itself is CUDA-only, but the same benchmark command is designed to be wrapped by `nsys` later:

```sh
uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx \
  --pytorch=functions-trace,autograd-shapes-nvtx \
  -- python scripts/benchmark_transformer.py \
  --device cuda \
  --model-size small \
  --batch-size 4 \
  --context-length 512 \
  --warmup 1 \
  --steps 1 \
  --mode train
```

On the Mac, use the same command without `nsys` and with `--device mps` or `--device auto` to debug logic offline.

### 2.1.5 Mixed Precision

Run the accumulation experiment from the PDF:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --task mixed-precision-accumulation
```

Inspect autocast dtypes for the toy model:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --task toy-autocast-dtypes \
  --dtype bfloat16
```

Benchmark the Transformer under BF16 autocast:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 128 \
  --warmup 5 \
  --steps 10 \
  --mode train \
  --dtype bfloat16
```

Compare it to the same command with `--dtype float32`. MPS BF16/FP16 behavior can differ from NVIDIA Tensor Cores, so treat the Mac result as local intuition, not a replacement for final CUDA numbers.

### 2.1.6 Memory Profiling

For local Mac memory profiling, use:

```sh
mkdir -p outputs
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 256 \
  --warmup 2 \
  --steps 5 \
  --mode train \
  --profile-memory \
  --profile-saved-tensors \
  --output-json outputs/tiny_train_mps_memory.json
```

Read these JSON fields first:

- `memory.peak_rss_mib`: peak process resident memory.
- `memory.peak_accelerator_allocated_mib`: peak MPS tensor allocation sampled during the run.
- `memory.peak_accelerator_driver_mib`: memory held by the MPS driver.
- `saved_tensors.average_per_step_mib`: approximate autograd residuals saved per measured step.

This is not the same artifact as `torch.cuda.memory._dump_snapshot(...)`, but it is enough to study how memory changes across context length, batch size, model size, forward-only, and full training.

Compute the residual-stream activation size from part 2.1.6(d):

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --task residual-stream-size \
  --model-size xl \
  --batch-size 4 \
  --context-length 2048 \
  --dtype float32
```

## Context-Length Sweep

Run the same configuration at increasing context lengths:

```sh
for ctx in 128 256 512; do
  uv run --offline python scripts/benchmark_transformer.py \
    --model-size tiny \
    --batch-size 1 \
    --context-length "$ctx" \
    --warmup 2 \
    --steps 5 \
    --mode train \
    --profile-memory \
    --profile-saved-tensors \
    --output-json "outputs/tiny_ctx_${ctx}_train.json"
done
```

For the assignment’s memory questions, repeat the pattern with `--mode forward` and `--mode train`, then compare peak memory. On the Mac, use `tiny` or `small`; on a CUDA machine, use the requested `xl` model and context lengths.

## Section 3: Single-GPU Memory

### 3.1 Autograd Residuals

Use `--profile-saved-tensors` to estimate saved tensors:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 128 \
  --warmup 1 \
  --steps 2 \
  --mode train \
  --profile-saved-tensors
```

### 3.2 Activation Checkpointing

Compare uncheckpointed versus checkpointed layer groups:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 256 \
  --mode train \
  --profile-memory \
  --profile-saved-tensors
```

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 256 \
  --mode train \
  --checkpoint-every 1 \
  --profile-memory \
  --profile-saved-tensors
```

For the PDF’s `xl` checkpointing deliverable, use CUDA later and increase `--model-size`, `--batch-size`, and `--context-length`.

## Section 4: GPU Kernels And Attention

### 4.1 PyTorch Attention Benchmarking

Run a laptop-safe PyTorch attention sweep:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --task attention-benchmark \
  --attention-d-models 16,32,64,128 \
  --attention-seq-lengths 256,1024 \
  --warmup 2 \
  --steps 5
```

The PDF asks for sequence lengths up to `16384`; those may be too large on a Mac. Use the same flag later on CUDA:

```sh
uv run python scripts/benchmark_transformer.py \
  --device cuda \
  --task attention-benchmark \
  --attention-d-models 16,32,64,128 \
  --attention-seq-lengths 256,1024,4096,8192,16384 \
  --warmup 5 \
  --steps 100
```

### 4.2 Torch Compile

Compile the full Transformer benchmark:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --model-size tiny \
  --batch-size 1 \
  --context-length 128 \
  --mode train \
  --compile-model
```

Compile the vanilla attention baseline:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --task attention-benchmark \
  --compile-attention \
  --attention-d-models 16,32 \
  --attention-seq-lengths 256,1024
```

### 4.2.2-4.2.3 FlashAttention-2

The script labels these sections but does not implement the required Triton kernels. The implementation belongs in `cs336_systems` and is wired through `tests/adapters.py`.

Use these tests as milestones:

```sh
uv run pytest -k test_flash_forward_pass_pytorch
uv run pytest -k test_flash_forward_pass_triton
uv run pytest -k test_flash_backward
```

On Mac, the pure PyTorch FlashAttention forward can be debugged if you write it device-agnostically. The Triton forward/backward and final benchmarking need CUDA.

## Section 5: Distributed Data Parallel Training

### 5.1 Distributed Communication

Run local CPU/Gloo all-reduce offline:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --task distributed-all-reduce \
  --world-size 2 \
  --all-reduce-sizes-mb 1,10 \
  --warmup 2 \
  --steps 5
```

For the actual PDF deliverable, repeat the idea on a CUDA machine with NCCL and 2/4/6 GPUs.

### 5.2-5.3 DDP Implementations

The implementation work belongs in `cs336_systems`, with adapters in `tests/adapters.py`.

```sh
uv run --offline pytest tests/test_ddp.py
```

Local Gloo is useful for logic; final performance and Nsight overlap screenshots require CUDA/NCCL.

## Section 6: Optimizer State Sharding

Implement the sharded optimizer in `cs336_systems`, expose it through `tests/adapters.py`, then run:

```sh
uv run --offline pytest tests/test_sharded_optimizer.py
```

The memory accounting can use the script’s `--profile-memory` locally for tiny/small sanity checks, but final `xl` 2-GPU numbers need CUDA.

## Section 7: Fully-Sharded Data Parallel

Implement FSDP in `cs336_systems`, expose it through `tests/adapters.py`, then run:

```sh
uv run --offline pytest tests/test_fsdp.py
```

Local testing can catch correctness issues. The PDF’s all-gather timing and Nsight screenshots need CUDA/NCCL.

## Section 8: Analyzing Parallelism Strategies

Use the formula helper to keep the symbols straight while answering the written calculations:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --task parallelism-calculator \
  --parallel-batch 1024 \
  --parallel-d-model 4096 \
  --parallel-d-ff 11008 \
  --parallel-devices 8 \
  --parallel-bandwidth-gbps 400 \
  --parallel-compute-tflops 1000
```

This prints the ring collective formulas and a concrete DP example. Use the PDF’s equations for the final symbolic answers.

## Section 9: Leaderboard

Print the target leaderboard configuration:

```sh
uv run --offline python scripts/benchmark_transformer.py \
  --task leaderboard-config
```

The real leaderboard benchmark requires two B200 GPUs. Use the local script only to debug the timing harness and small-model behavior.

## CUDA-Only Work Later

These assignment pieces require an NVIDIA GPU:

- `torch.cuda.synchronize()` timing behavior.
- `nsys profile ...` and Nsight Systems timelines.
- CUDA kernel summaries.
- `torch.cuda.memory._record_memory_history(...)` and `_dump_snapshot(...)`.
- `pytorch.org/memory_viz` snapshots from CUDA allocations.

When you get CUDA access, run the same script with `--device cuda`. For example:

```sh
uv run python scripts/benchmark_transformer.py \
  --device cuda \
  --model-size small \
  --batch-size 4 \
  --context-length 512 \
  --warmup 5 \
  --steps 10 \
  --mode train
```

Then layer Nsight on top of a short run:

```sh
uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx \
  --pytorch=functions-trace,autograd-shapes-nvtx \
  -- python scripts/benchmark_transformer.py \
  --device cuda \
  --model-size small \
  --batch-size 4 \
  --context-length 512 \
  --warmup 1 \
  --steps 1 \
  --mode train
```

Use the Mac workflow to debug the script and generate preliminary tables offline; use CUDA later for the exact assignment profiler screenshots and kernel-level answers.
