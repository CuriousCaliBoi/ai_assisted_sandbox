# CS336 Assignment 2 Command Cheat Sheet

Run everything from the repo root:

```sh
cd /Users/kingzuko/projects/ai_assisted_sandbox
```

The script now creates `outputs/` automatically when you pass `--output-json outputs/...`.

For conceptual explanations and writeup scaffolding, read `docs/assignment2_concepts_and_questions.md`.
For analysis of the generated Mac results, read `docs/assignment2_mac_results_analysis.md`.

Run all Mac-supported sections in one batch:

```sh
./scripts/run_mac_offline_assignment2.sh
```

## Offline Sanity Checks

```sh
uv run --offline python -c "import torch, cs336_basics; print(torch.__version__); print(torch.backends.mps.is_available())"
```

```sh
uv run --offline python scripts/benchmark_transformer.py --list-assignment-coverage
```

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 32 --warmup 1 --steps 1 --mode train
```

## 2.1.3 End-To-End Benchmarking

Forward only:

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 5 --steps 10 --mode forward --no-grad-forward --output-json outputs/bench_tiny_forward.json
```

Forward + backward:

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 5 --steps 10 --mode forward_backward --output-json outputs/bench_tiny_forward_backward.json
```

Forward + backward + optimizer:

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 5 --steps 10 --mode train --output-json outputs/bench_tiny_train.json
```

Warmup comparison:

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 0 --steps 10 --mode train --output-json outputs/bench_tiny_train_warmup0.json
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 1 --steps 10 --mode train --output-json outputs/bench_tiny_train_warmup1.json
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 2 --steps 10 --mode train --output-json outputs/bench_tiny_train_warmup2.json
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 5 --steps 10 --mode train --output-json outputs/bench_tiny_train_warmup5.json
```

Try the smallest real PDF model if your laptop can handle it:

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size small --batch-size 1 --context-length 128 --warmup 2 --steps 5 --mode train --output-json outputs/bench_small_ctx128_train.json
```

## 2.1.4 Nsight Systems Profiling

Mac/offline dry run:

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 1 --steps 1 --mode train --output-json outputs/nsight_dry_run_tiny.json
```

Skip on Mac: the real Nsight command requires an NVIDIA GPU, CUDA PyTorch, and `nsys` installed. It is commented out so you do not accidentally run it on the laptop.

```sh
# uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx -- python scripts/benchmark_transformer.py --device cuda --model-size small --batch-size 4 --context-length 512 --warmup 1 --steps 1 --mode train
```

## 2.1.5 Mixed Precision

Accumulation experiment:

```sh
uv run --offline python scripts/benchmark_transformer.py --task mixed-precision-accumulation
```

Toy autocast dtype experiment:

```sh
uv run --offline python scripts/benchmark_transformer.py --task toy-autocast-dtypes --dtype bfloat16
```

FP32 timing:

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 5 --steps 10 --mode train --dtype float32 --output-json outputs/mp_tiny_fp32.json
```

BF16 timing:

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 5 --steps 10 --mode train --dtype bfloat16 --output-json outputs/mp_tiny_bf16.json
```

## 2.1.6 Memory Profiling

Forward-only memory:

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 256 --warmup 2 --steps 5 --mode forward --no-grad-forward --profile-memory --output-json outputs/memory_tiny_ctx256_forward.json
```

Full training memory:

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 256 --warmup 2 --steps 5 --mode train --profile-memory --profile-saved-tensors --output-json outputs/memory_tiny_ctx256_train.json
```

Context length sweep:

```sh
for ctx in 128 256 512; do uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length "$ctx" --warmup 2 --steps 5 --mode train --profile-memory --profile-saved-tensors --output-json "outputs/memory_tiny_ctx${ctx}_train.json"; done
```

Residual stream size for the PDF's `xl` config:

```sh
uv run --offline python scripts/benchmark_transformer.py --task residual-stream-size --model-size xl --batch-size 4 --context-length 2048 --dtype float32
```

Skip on Mac: CUDA memory snapshots with `torch.cuda.memory._dump_snapshot(...)` require an NVIDIA GPU.

## 3.1 Autograd Residuals

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 1 --steps 2 --mode train --profile-saved-tensors --output-json outputs/residuals_tiny_train.json
```

## 3.2 Activation Checkpointing

No checkpointing:

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 256 --warmup 2 --steps 5 --mode train --profile-memory --profile-saved-tensors --output-json outputs/checkpoint_none.json
```

Checkpoint every layer:

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 256 --warmup 2 --steps 5 --mode train --checkpoint-every 1 --profile-memory --profile-saved-tensors --output-json outputs/checkpoint_every1.json
```

Checkpoint every 2 layers:

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 256 --warmup 2 --steps 5 --mode train --checkpoint-every 2 --profile-memory --profile-saved-tensors --output-json outputs/checkpoint_every2.json
```

## 4.1 PyTorch Attention Benchmarking

Laptop-safe sweep:

```sh
uv run --offline python scripts/benchmark_transformer.py --task attention-benchmark --attention-batch-size 1 --attention-d-models 16,32,64,128 --attention-seq-lengths 128,256,512 --warmup 2 --steps 5 --output-json outputs/attention_pytorch_small.json
```

Larger sweep, still maybe okay:

```sh
uv run --offline python scripts/benchmark_transformer.py --task attention-benchmark --attention-batch-size 1 --attention-d-models 16,32,64,128 --attention-seq-lengths 256,1024 --warmup 2 --steps 5 --output-json outputs/attention_pytorch_medium.json
```

Skip on Mac: CUDA-sized sweep. This uses CUDA and much larger tensor sizes than the laptop-safe sweeps above.

```sh
# uv run python scripts/benchmark_transformer.py --device cuda --task attention-benchmark --attention-batch-size 8 --attention-d-models 16,32,64,128 --attention-seq-lengths 256,1024,4096,8192,16384 --warmup 5 --steps 100 --output-json outputs/attention_pytorch_cuda.json
```

## 4.2 Torch Compile

Compile full Transformer:

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 2 --steps 5 --mode train --compile-model --output-json outputs/compile_transformer_tiny.json
```

Compile attention:

```sh
uv run --offline python scripts/benchmark_transformer.py --task attention-benchmark --compile-attention --attention-batch-size 1 --attention-d-models 16,32,64 --attention-seq-lengths 128,256,512 --warmup 2 --steps 5 --output-json outputs/compile_attention_small.json
```

## 4.2.2-4.2.3 FlashAttention Tests

Pure PyTorch forward test after implementation:

```sh
uv run --offline pytest -k test_flash_forward_pass_pytorch
```

Skip on Mac: Triton forward test requires CUDA/Triton GPU execution.

```sh
# uv run pytest -k test_flash_forward_pass_triton
```

Backward test:

```sh
uv run pytest -k test_flash_backward
```

## 5.1 Distributed Communication

Local CPU/Gloo all-reduce:

```sh
uv run --offline python scripts/benchmark_transformer.py --task distributed-all-reduce --world-size 2 --all-reduce-sizes-mb 1,10 --warmup 2 --steps 5
```

Try a larger local tensor:

```sh
uv run --offline python scripts/benchmark_transformer.py --task distributed-all-reduce --world-size 2 --all-reduce-sizes-mb 1,10,100 --warmup 2 --steps 5
```

## 5.2-5.3 DDP

Run DDP tests after implementation:

```sh
uv run --offline pytest tests/test_ddp.py
```

Run multiple times:

```sh
for i in 1 2 3 4 5; do uv run --offline pytest tests/test_ddp.py || break; done
```

## 6 Optimizer State Sharding

Run sharded optimizer tests after implementation:

```sh
uv run --offline pytest tests/test_sharded_optimizer.py
```

Run multiple times:

```sh
for i in 1 2 3 4 5; do uv run --offline pytest tests/test_sharded_optimizer.py || break; done
```

## 7 FSDP

Run FSDP tests after implementation:

```sh
uv run --offline pytest tests/test_fsdp.py
```

Run multiple times:

```sh
for i in 1 2 3 4 5; do uv run --offline pytest tests/test_fsdp.py || break; done
```

## 8 Parallelism Math

Default helper:

```sh
uv run --offline python scripts/benchmark_transformer.py --task parallelism-calculator
```

Custom values:

```sh
uv run --offline python scripts/benchmark_transformer.py --task parallelism-calculator --parallel-batch 1024 --parallel-d-model 4096 --parallel-d-ff 11008 --parallel-devices 8 --parallel-bandwidth-gbps 400 --parallel-compute-tflops 1000
```

## 9 Leaderboard

Print target config:

```sh
uv run --offline python scripts/benchmark_transformer.py --task leaderboard-config
```

Local tiny full-step timing:

```sh
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 5 --steps 10 --mode train --dtype bfloat16 --output-json outputs/leaderboard_tiny_local.json
```

## Reading Output JSON

Most useful fields:

- `timings.forward_s.mean_s`
- `timings.backward_s.mean_s`
- `timings.optimizer_s.mean_s`
- `timings.total_s.mean_s`
- `memory.peak_rss_mib`
- `memory.peak_accelerator_allocated_mib`
- `saved_tensors.average_per_step_mib`

If a command crashes or is too slow, lower `--context-length`, lower `--batch-size`, and use `--model-size tiny`.
