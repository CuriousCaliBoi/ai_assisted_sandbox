#!/usr/bin/env bash
set -euo pipefail

# Mac/offline runner for the parts of CS336 Assignment 2 that do not require
# NVIDIA CUDA, Nsight Systems, NCCL, or Triton GPU kernels.

mkdir -p outputs

echo "== Sanity check =="
uv run --offline python -c "import torch, cs336_basics; print(torch.__version__); print('mps_available', torch.backends.mps.is_available())"

echo "== 2.1.3 End-to-end benchmarking =="
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 5 --steps 10 --mode forward --no-grad-forward --output-json outputs/mac_2_1_3_forward.json
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 5 --steps 10 --mode forward_backward --output-json outputs/mac_2_1_3_forward_backward.json
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 5 --steps 10 --mode train --output-json outputs/mac_2_1_3_train.json

echo "== 2.1.3 Warmup comparison =="
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 0 --steps 10 --mode train --output-json outputs/mac_2_1_3_warmup0.json
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 1 --steps 10 --mode train --output-json outputs/mac_2_1_3_warmup1.json
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 2 --steps 10 --mode train --output-json outputs/mac_2_1_3_warmup2.json

echo "== 2.1.5 Mixed precision =="
uv run --offline python scripts/benchmark_transformer.py --task mixed-precision-accumulation --output-json outputs/mac_2_1_5_accumulation.json || true
uv run --offline python scripts/benchmark_transformer.py --task toy-autocast-dtypes --dtype bfloat16 --output-json outputs/mac_2_1_5_toy_autocast.json || true
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 5 --steps 10 --mode train --dtype float32 --output-json outputs/mac_2_1_5_fp32.json
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 5 --steps 10 --mode train --dtype bfloat16 --output-json outputs/mac_2_1_5_bf16.json

echo "== 2.1.6 Memory profiling and residual-stream size =="
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 256 --warmup 2 --steps 5 --mode forward --no-grad-forward --profile-memory --output-json outputs/mac_2_1_6_forward_memory.json
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 256 --warmup 2 --steps 5 --mode train --profile-memory --profile-saved-tensors --output-json outputs/mac_2_1_6_train_memory.json
uv run --offline python scripts/benchmark_transformer.py --task residual-stream-size --model-size xl --batch-size 4 --context-length 2048 --dtype float32 --output-json outputs/mac_2_1_6_residual_stream_size.json || true

echo "== 3.1 Autograd residuals =="
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 1 --steps 2 --mode train --profile-saved-tensors --output-json outputs/mac_3_1_residuals.json

echo "== 3.2 Activation checkpointing =="
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 256 --warmup 2 --steps 5 --mode train --profile-memory --profile-saved-tensors --output-json outputs/mac_3_2_checkpoint_none.json
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 256 --warmup 2 --steps 5 --mode train --checkpoint-every 1 --profile-memory --profile-saved-tensors --output-json outputs/mac_3_2_checkpoint_every1.json
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 256 --warmup 2 --steps 5 --mode train --checkpoint-every 2 --profile-memory --profile-saved-tensors --output-json outputs/mac_3_2_checkpoint_every2.json

echo "== 4.1 PyTorch attention benchmarking =="
uv run --offline python scripts/benchmark_transformer.py --task attention-benchmark --attention-batch-size 1 --attention-d-models 16,32,64,128 --attention-seq-lengths 128,256,512 --warmup 2 --steps 5 --output-json outputs/mac_4_1_attention_pytorch.json

echo "== 4.2 Torch compile smoke tests =="
uv run --offline python scripts/benchmark_transformer.py --model-size tiny --batch-size 1 --context-length 128 --warmup 1 --steps 2 --mode train --compile-model --output-json outputs/mac_4_2_compile_transformer.json || true
uv run --offline python scripts/benchmark_transformer.py --task attention-benchmark --compile-attention --attention-batch-size 1 --attention-d-models 16,32 --attention-seq-lengths 128,256 --warmup 1 --steps 2 --output-json outputs/mac_4_2_compile_attention.json || true

echo "== 5.1 Local Gloo all-reduce =="
uv run --offline python scripts/benchmark_transformer.py --task distributed-all-reduce --world-size 2 --all-reduce-sizes-mb 1,10 --warmup 2 --steps 5

echo "== 8 Parallelism math helper =="
uv run --offline python scripts/benchmark_transformer.py --task parallelism-calculator --output-json outputs/mac_8_parallelism_calculator.json || true

echo "== 9 Leaderboard target config =="
uv run --offline python scripts/benchmark_transformer.py --task leaderboard-config --output-json outputs/mac_9_leaderboard_config.json || true

echo "Done. JSON outputs are in ./outputs."
