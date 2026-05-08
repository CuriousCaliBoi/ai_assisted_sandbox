# CS336 Assignment 2 Concepts And Question Guide

This is a Mac/offline companion to the assignment PDF. It explains what each section is trying to teach, what each question is asking you to reason about, what you can measure locally, and what you must mark as CUDA/NVIDIA-only.

Use alongside:

- `docs/assignment2_commands.md` for exact commands.
- `scripts/run_mac_offline_assignment2.sh` to run all Mac-supported measurements.
- `outputs/mac_*.json` for measured local results.
- `docs/assignment2_mac_results_analysis.md` for observations from the generated Mac results.

## Big Picture

Assignment 2 is about systems constraints in Transformer training:

- Time: where forward, backward, optimizer, kernels, and communication spend wall-clock time.
- Memory: where parameters, activations, gradients, optimizer state, and temporary attention matrices live.
- IO: how much data moves between memory levels or between devices.
- Parallelism: how data parallelism, optimizer sharding, FSDP, and tensor parallelism trade compute for communication.

On your Mac, you can study the shape of these ideas with CPU/MPS. You cannot collect final CUDA kernel traces, Nsight screenshots, NCCL timings, Triton kernel benchmarks, or leaderboard numbers.

## 1 Assignment Overview

The assignment expects both code and a written report. The code pieces mostly live in `cs336_systems` and are exposed through `tests/adapters.py`. The profiling scripts can live under `scripts/`.

Main implementation themes:

- Benchmarking and profiling harness.
- Activation checkpointing.
- FlashAttention-2 in PyTorch/Triton.
- DDP.
- Optimizer state sharding.
- FSDP.

Main written themes:

- Explain measured timing and memory behavior.
- Connect memory usage to tensor shapes.
- Explain communication costs with formulas.
- Compare strategies under bottleneck assumptions.

## 2 Profiling And Benchmarking

### 2.1.1 Setup: Importing The Basics Transformer

Concept: Assignment 2 reuses the assignment 1 Transformer as a baseline. Before profiling anything, you need to know the model imports and instantiates correctly.

Question asks: Can your assignment 2 environment import `cs336_basics.model` and build `BasicsTransformerLM`?

Mac evidence: Any successful run of `scripts/benchmark_transformer.py` proves this.

What to say in writeup: Briefly mention you used random data and random weights because the profiling questions care about runtime and memory, not model quality.

### 2.1.2 Model Sizing

Concept: Runtime and memory scale strongly with `d_model`, `d_ff`, number of layers, number of heads, batch size, and context length.

Important configs:

- `small`: 768 model width, 12 layers.
- `medium`: 1024 model width, 24 layers.
- `large`: 1280 model width, 36 layers.
- `xl`: 2560 model width, 32 layers.
- `10B`: 4608 model width, 50 layers.

Question asks: Use consistent model sizes so timings and memory are comparable.

Mac limitation: The real configs may be too large locally. Use `tiny` or `small` on Mac for practice, and clearly label these as local exploratory measurements.

### 2.1.3 End-To-End Benchmarking

Concept: Measure high-level training step costs:

- Forward pass.
- Backward pass.
- Optimizer step.

Why warmup matters: First iterations include setup costs such as lazy initialization, memory allocation, graph/kernel caching, and MPS/PyTorch startup effects. With no warmup, the mean and standard deviation are often worse.

Question asks:

- Can you time forward-only, forward+backward, and full training?
- How do mean and standard deviation change with model size?
- How does omitting warmup affect measurements?

Mac evidence:

- `outputs/mac_2_1_3_forward.json`
- `outputs/mac_2_1_3_forward_backward.json`
- `outputs/mac_2_1_3_train.json`
- `outputs/mac_2_1_3_warmup0.json`
- `outputs/mac_2_1_3_warmup1.json`
- `outputs/mac_2_1_3_warmup2.json`

How to analyze:

- Forward-only is the inference cost.
- Backward usually costs more than forward because autograd computes gradients for many intermediates and parameters.
- Optimizer cost can be visible even for small models because AdamW touches every parameter and optimizer state.
- If `warmup=0` has higher mean/stdev than `warmup>0`, explain that first-iteration setup polluted the measurement.

Mac caveat: CUDA timing requires `torch.cuda.synchronize()`. On Mac, the script uses MPS synchronization where available. The concept is the same: synchronize asynchronous accelerator work before stopping the timer.

### 2.1.4 Nsight Systems Profiler

Concept: End-to-end timings say how long the whole step took, but not which GPU kernels or CPU calls caused the time. Nsight breaks execution into CUDA API calls, GPU kernels, library calls, and NVTX ranges.

Question asks:

- Does Nsight total time match Python timing?
- Which CUDA kernels dominate?
- How much time is matmul versus softmax/normalization/other kernels?
- How does kernel mix change in full training versus forward-only?

Mac status: Skip locally. This requires NVIDIA Nsight Systems and CUDA. You can only do a dry run of the Python benchmark on Mac.

How to write later: When you get CUDA results, compare Python timing to Nsight range timing. Expect matmul/GEMM kernels to dominate FLOPs, but non-matmul kernels can still matter due to memory bandwidth, launch overhead, and unfused elementwise operations.

### 2.1.5 Mixed Precision

Concept: Lower precision reduces memory bandwidth and can speed up matrix multiplications on hardware with special low-precision units. FP16 has small dynamic range; BF16 has FP32-like exponent range and is usually more stable.

Autocast concept:

- Parameters usually remain FP32.
- Some operations run in BF16/FP16.
- Numerically sensitive operations may stay FP32.
- Gradients usually accumulate into FP32 parameters.

Questions ask:

- Why does FP16 accumulation drift?
- What dtypes appear in a toy model under autocast?
- How do full precision and BF16 timings compare?

Mac evidence:

- `outputs/mac_2_1_5_accumulation.json`
- `outputs/mac_2_1_5_toy_autocast.json`
- `outputs/mac_2_1_5_fp32.json`
- `outputs/mac_2_1_5_bf16.json`

How to analyze:

- FP16 accumulation of `0.01` repeatedly is inaccurate because the value and intermediate sums are rounded at low precision.
- Keeping the accumulator in FP32 helps, but if the addend is already rounded to FP16, some input error remains.
- In the toy autocast result, layernorm may stay FP32 while linear outputs/logits may be BF16.
- On MPS, BF16 timing is not necessarily representative of NVIDIA Tensor Core speedups.

### 2.1.6 Memory Profiling

Concept: Training memory includes:

- Parameters.
- Activations saved for backward.
- Gradients.
- Optimizer state.
- Temporary tensors.

Forward-only inference can avoid saving backward activations with `torch.no_grad()`. Full training must save or recompute data needed by backward.

Questions ask:

- What do memory timelines look like for forward-only versus full training?
- What is peak memory at different context lengths?
- Does mixed precision reduce memory?
- How large is one residual-stream activation tensor?
- Which large allocations dominate?

Mac evidence:

- `outputs/mac_2_1_6_forward_memory.json`
- `outputs/mac_2_1_6_train_memory.json`
- `outputs/mac_2_1_6_residual_stream_size.json`

Important formula:

```text
residual_stream_MiB = batch_size * context_length * d_model * bytes_per_element / 1024**2
```

For `xl`, batch 4, context 2048, d_model 2560, FP32:

```text
4 * 2048 * 2560 * 4 / 1024**2 = 80 MiB
```

Mac caveat: PyTorch CUDA memory snapshots and `pytorch.org/memory_viz` require CUDA snapshots. The Mac script gives sampled RSS/MPS memory and saved-tensor estimates, not the official CUDA snapshot artifact.

## 3 Single-GPU Memory

### 3.1 Autograd Residuals

Concept: Backward pass needs information from forward pass. PyTorch saves tensors, called residuals or saved tensors, for use during gradient computation.

Important lesson: A simple-looking operation like RMSNorm can save multiple full-size tensors if implemented as many separate PyTorch ops. Fusing operations can reduce saved tensors.

Question asks: Understand what tensors are saved and how much memory they use.

Mac evidence:

- `outputs/mac_3_1_residuals.json`
- `saved_tensors.average_per_step_mib`

How to analyze:

- Large saved tensors usually scale like activation shapes: batch x sequence x hidden dimension.
- Attention can be worse because attention scores scale like batch x heads x sequence x sequence.
- More granular operations may save more intermediates than a fused operation.

### 3.2 Activation Checkpointing

Concept: Activation checkpointing trades compute for memory. Instead of saving every intermediate during forward, it saves selected checkpoint inputs and recomputes the missing forward work during backward.

Question asks:

- What checkpointing strategy minimizes peak activation memory if compute is ignored?
- For one level of checkpointing, what checkpoint block size works best?
- How do measured peak memory and runtime change?

Mac evidence:

- `outputs/mac_3_2_checkpoint_none.json`
- `outputs/mac_3_2_checkpoint_every1.json`
- `outputs/mac_3_2_checkpoint_every2.json`

How to analyze:

- Smaller checkpoint scopes reduce peak saved activations inside each group, but more checkpoint boundaries save more group inputs.
- Larger checkpoint scopes save fewer boundaries, but recomputation materializes more intermediate residuals during backward.
- Runtime usually increases because backward includes recomputation.
- In asymptotic discussion, recursive checkpointing can push peak activation memory below linear in number of layers, at the cost of extra compute.

## 4 GPU Kernels

### 4.1 PyTorch Attention Benchmarking

Concept: Vanilla attention materializes an attention matrix with shape roughly:

```text
batch_size * num_heads * seq_len * seq_len
```

That quadratic `seq_len**2` term dominates memory at long contexts.

Question asks:

- Benchmark attention over head dimension and sequence length.
- Identify out-of-memory thresholds.
- Explain saved memory growth with sequence length.
- Explain what would eliminate this memory cost.

Mac evidence:

- `outputs/mac_4_1_attention_pytorch.json`

How to analyze:

- Runtime and memory should grow sharply with sequence length.
- The attention score/probability matrix is the main problem.
- FlashAttention avoids materializing the full attention matrix in HBM by tiling, online softmax, fusion, and recomputation.

### 4.2 Torch Compile

Concept: `torch.compile` traces/compiles PyTorch graphs and may fuse operations or generate better kernels.

Questions ask:

- Does compiled attention beat uncompiled attention?
- Does compiled Transformer improve forward/backward/full-step time?

Mac evidence:

- `outputs/mac_4_2_compile_transformer.json`
- `outputs/mac_4_2_compile_attention.json`

How to analyze:

- Compilation has startup overhead, so separate compile cost from steady-state timing.
- On Mac/MPS, performance may differ from CUDA.
- `torch.compile` can obscure attribution in profilers because multiple source operations may become fused kernels.

### 4.2.1 Weighted Sum Triton Example

Concept: Triton kernels operate on blocks/tiles. You manually load data, compute, and store outputs. Autograd integration uses `torch.autograd.Function`.

Question purpose: This is a tutorial for how the FlashAttention Triton implementation should be structured.

Mac status: Triton GPU kernels are not runnable locally. You can still read the code pattern:

- Forward saves tensors in `ctx`.
- Backward receives upstream gradients.
- Tiled kernels use program IDs, strides, block pointers, and boundary checks.

### 4.2.2 FlashAttention-2 Forward Pass

Concepts:

- Tiling: process blocks of queries and keys.
- Online softmax: maintain row-wise running max `m` and denominator proxy `l`.
- Fusion: compute score, softmax, and value matmul in one kernel.
- Reduced memory IO: avoid writing full attention matrices to HBM.

Question asks:

- Implement pure PyTorch FA2 forward as a debugging reference.
- Implement Triton FA2 forward.
- Add causal masking.

How to reason:

- Pure PyTorch tiled implementation helps verify output and logsumexp.
- Triton kernel should write output `O` and logsumexp `L`.
- Causal masking zeros out positions where key index is greater than query index.

Mac status: Pure PyTorch reference may be developed locally if implemented device-agnostically. Triton final tests require CUDA.

### 4.2.3 FlashAttention Backward

Concept: Instead of saving the full attention probability matrix `P`, FlashAttention saves smaller data such as `Q`, `K`, `V`, `O`, and logsumexp `L`, then recomputes attention probabilities during backward.

Important backward pieces:

```text
D = rowsum(dO * O)
S = QK^T / sqrt(d)
P = exp(S - L)
dV = P^T dO
dP = dO V^T
dS = P * (dP - D)
dQ = dS K / sqrt(d)
dK = dS^T Q / sqrt(d)
```

Question asks:

- Implement backward using PyTorch and `torch.compile`.
- Benchmark Triton/pure PyTorch FlashAttention versus vanilla attention.

Mac status: PyTorch logic can be reasoned about locally. Triton performance and final benchmark require CUDA.

## 5 Distributed Data Parallel Training

### 5.1 Distributed Communication

Concept: Distributed training uses collectives such as all-reduce, all-gather, reduce-scatter, and broadcast.

Important terms:

- Rank: process ID.
- World size: number of participating processes.
- Backend: Gloo for CPU/local debugging, NCCL for CUDA GPUs.
- All-reduce: every rank starts with a tensor and ends with the reduced result.

Question asks:

- Benchmark all-reduce over tensor sizes and GPU counts.
- Aggregate timing across ranks.
- Explain how data size and device count affect communication.

Mac evidence:

- Local Gloo output from `scripts/run_mac_offline_assignment2.sh`.
- This is conceptually useful but not final NCCL/GPU evidence.

How to analyze:

- Larger tensors take longer because more bytes move.
- More ranks can increase total communication and synchronization overhead.
- Timings vary across ranks; report aggregate/mean and variability.

### 5.2 Naive DDP

Concept: Data parallel training replicates the model on every rank, shards the batch, computes local gradients, all-reduces gradients, then every rank runs the optimizer step.

Question asks:

- Implement minimal DDP by all-reducing gradients after backward.
- Verify against single-process training.
- Benchmark step time and communication time.

How to reason:

- Parameters must start identical, usually via broadcast from rank 0.
- Gradients must be averaged, not just summed, unless loss scaling already accounts for world size.
- All ranks must apply the same optimizer update.

Mac status: Gloo can test logic. Final performance requires CUDA/NCCL.

### 5.3 Improved DDP

Concept: Naive DDP has two overheads:

- One all-reduce per parameter has many small communication calls.
- Waiting until backward finishes misses overlap opportunities.

Improvements:

- Flatten gradients and all-reduce one large tensor.
- Use gradient hooks and async all-reduce as soon as individual gradients are ready.

Question asks:

- Compare individual all-reduces versus flat all-reduce.
- Implement overlapped individual-parameter DDP.
- Show overlap in Nsight.

How to reason:

- Flat all-reduce reduces launch/collective overhead but cannot overlap as naturally.
- Async hooks can overlap communication of later-layer gradients while earlier-layer backward computation continues.
- You must wait for all async handles before `optimizer.step()`.

Mac status: Logic tests can run locally; Nsight screenshots require CUDA.

## 6 Optimizer State Sharding

Concept: AdamW stores extra state per parameter, usually first and second moments. With normal DDP, every rank stores all optimizer state, duplicating memory.

Optimizer state sharding idea:

- Each rank owns only a subset of parameters for optimizer updates.
- After local shard updates, updated parameters are broadcast/gathered so all ranks have synchronized model weights.

Question asks:

- Implement a wrapper optimizer.
- Handle parameter groups.
- Synchronize updated parameters after each step.
- Compare memory and speed with/without sharding.
- Compare to ZeRO stage 1.

How to reason:

- Memory saved: optimizer state is approximately divided by world size.
- Communication added: updated parameter shards must be shared after each step.
- Difference from ZeRO stage 1: real ZeRO has specific partitioning and communication schedules; this assignment's simplified sharding may broadcast updated parameters more directly.

Mac status: Tests can run locally. Final memory/timing needs GPU scale.

## 7 Fully-Sharded Data Parallel

Concept: DDP replicates weights. Optimizer sharding shards optimizer state, but weights are still duplicated. FSDP shards weights, gradients, and optimizer states.

How FSDP works:

- Each rank stores a shard of a parameter.
- Before a layer uses the parameter, ranks all-gather full weights.
- After use, full gathered weights can be freed.
- During backward, gradients are reduce-scattered so each rank keeps only its shard.
- Optional compute dtype sends/uses lower-precision weights while keeping master weights FP32.

Question asks:

- Implement FSDP wrappers/hooks for Linear/Embedding-like modules.
- Prefetch/all-gather weights before use.
- Free gathered weights after use.
- Reduce-scatter gradients.
- Account for memory savings and communication timing.

How to reason:

- FSDP saves parameter and optimizer memory, roughly scaling down with world size.
- It adds communication before forward/backward layers.
- Effective FSDP overlaps communication with compute so all-gathers finish before the layer needs weights.
- Small layers like norms may not be worth sharding.

Mac status: Correctness tests can run locally. Real all-gather timing and Nsight evidence require CUDA/NCCL.

## 8 Analyzing Parallelism Strategies

This section is mostly symbolic math. The key pattern is:

```text
communication bottleneck when communication_time > compute_time
```

### 8.1 Communication Primitives

Ring all-gather:

```text
T = (N - 1) / N * S / W
```

Ring reduce-scatter:

```text
T = (N - 1) / N * S / W
```

Ring all-reduce as reduce-scatter + all-gather:

```text
T = 2 * (N - 1) / N * S / W
```

Alternate all-reduce question: If each step sends a full-size `S` tensor for `N - 1` steps, time is approximately:

```text
T = (N - 1) * S / W
```

The one-sentence justification: each device sends `S` bytes per step for `N - 1` sequential ring steps.

### 8.2 Data Parallel Calculations

Concept: Batch is sharded, weights are replicated. Forward has no collectives. Backward needs all-reduce of weight gradients.

FFN backward matmuls from the PDF:

- `dy W3^T`
- `dx1 W1^T`
- `dx2 W2^T`
- `z^T dy`
- `x^T dx2`
- `x^T dx1`

Ignoring elementwise ops, the backward compute per DP rank is proportional to:

```text
6 * (B / N_DP) * D * D_FF
```

Gradient communication size for three FFN matrices is:

```text
3 * D * D_FF * bytes_per_element
```

All-reduce time uses ring all-reduce.

### 8.3 FSDP Calculations

Concept: Batch is sharded like DP, but weights and gradients are sharded too.

Forward:

- All-gather weights.
- Run local batch-sharded compute.

Backward:

- All-gather weights as needed.
- Compute local gradients.
- Reduce-scatter gradients instead of all-reduce.

Question asks:

- Compute forward/backward FLOPs per rank.
- Compute forward/backward communication time.
- Solve for when communication exceeds compute.

Key distinction from DP: FSDP communicates weights in forward/backward and reduce-scatters gradients, while DP mainly all-reduces gradients.

### 8.4 Tensor Parallel Calculations

Concept: Weight matrices are split along hidden dimensions, so each device computes part of the matmul.

For the FFN layout in the PDF:

- `W1` and `W2` are column-parallel.
- `W3` is row-parallel.
- The final row-parallel output needs an all-reduce across TP ranks.

Question asks:

- Write the backward pass with sharded weights/activations.
- Compute forward/backward FLOPs.
- Compute forward/backward communication.
- Solve bottleneck inequalities.

How to reason:

- TP reduces compute per device.
- TP introduces activation communication.
- Unlike DP/FSDP, communication depends heavily on batch and activation size, not just weight size.

### 8.5 2D Parallelism: FSDP + TP

Concept: Use a grid of devices:

```text
N = N_FSDP * N_TP
```

FSDP shards across one axis; TP shards across another. This can scale farther because it splits both weight/optimizer memory and matmul work.

Question asks:

- Compute forward FLOPs.
- Compute communication time when FSDP-axis and TP-axis collectives can overlap.
- Solve optimal `N_FSDP`, `N_TP`.
- Repeat when collectives cannot overlap.

Key idea:

- If collectives overlap, communication time is the max of the FSDP-axis cost and TP-axis cost.
- If collectives cannot overlap, communication time is the sum of the two costs.
- Optimal split balances the two communication terms against compute.

Mac evidence:

- `outputs/mac_8_parallelism_calculator.json`

## 9 Leaderboard

Concept: Combine all systems tricks to make one full training step fast and fit in memory:

- BF16.
- FlashAttention.
- Activation checkpointing if needed.
- FSDP / sharding.
- Possibly fused LM head + cross entropy.
- Possibly fused AdamW.
- Careful compile/autotune choices.

Question asks: Report best wall-clock time on the specified B200 setup.

Mac status: You cannot do this faithfully locally. Use Mac runs only to debug small timing harnesses and understand which pieces matter.

## Suggested Mac Writeup Framing

For sections requiring CUDA/NVIDIA, say something like:

```text
I could not collect the final CUDA/Nsight/NCCL/Triton measurement on my Mac because it does not have an NVIDIA GPU. I used the local MPS/CPU runs to validate the benchmark harness and study relative behavior on small models. The final CUDA-only deliverables would need to be rerun on the required GPU environment.
```

For Mac-supported sections, cite local JSON outputs and report:

- Mean and standard deviation for timings.
- Peak RSS/MPS memory where available.
- Saved tensor estimates for residual/checkpointing comparisons.
- Qualitative trends rather than pretending Mac numbers equal B200/CUDA numbers.

## Quick Mapping: Question To Local Artifact

| PDF area | Local artifact |
| --- | --- |
| 2.1.3 timing | `outputs/mac_2_1_3_*.json` |
| warmup effect | `outputs/mac_2_1_3_warmup*.json` |
| mixed precision accumulation | `outputs/mac_2_1_5_accumulation.json` |
| autocast dtypes | `outputs/mac_2_1_5_toy_autocast.json` |
| FP32 vs BF16 timing | `outputs/mac_2_1_5_fp32.json`, `outputs/mac_2_1_5_bf16.json` |
| memory forward/train | `outputs/mac_2_1_6_*_memory.json` |
| residual stream size | `outputs/mac_2_1_6_residual_stream_size.json` |
| autograd residuals | `outputs/mac_3_1_residuals.json` |
| checkpointing | `outputs/mac_3_2_checkpoint_*.json` |
| PyTorch attention | `outputs/mac_4_1_attention_pytorch.json` |
| torch.compile | `outputs/mac_4_2_compile_*.json` |
| local all-reduce | terminal output from `scripts/run_mac_offline_assignment2.sh` |
| parallelism math | `outputs/mac_8_parallelism_calculator.json` |
| leaderboard config | `outputs/mac_9_leaderboard_config.json` |
