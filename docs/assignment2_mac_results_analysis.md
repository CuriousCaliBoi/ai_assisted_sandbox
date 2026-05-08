# CS336 Assignment 2 Mac Results Analysis

This note analyzes the results in:

```text
outputs/mac_assignment2_all_results.json
```

These are Mac/MPS exploratory results. They are useful for understanding concepts and drafting answers, but they are not substitutes for CUDA/Nsight/NCCL/Triton measurements required by parts of the assignment.

## High-Level Takeaways

- The benchmark harness works offline on MPS and produces stable results after warmup.
- Backward is consistently more expensive than forward because it computes parameter/input gradients and uses saved activations.
- AdamW optimizer time is visible even on the tiny model because the optimizer touches every parameter and maintains/update state.
- Warmup matters a lot: `warmup=0` is much slower and noisier than `warmup=1` or `warmup=2`.
- BF16 autocast is slower than FP32 on this MPS run, which is a good reminder that mixed precision speedups are hardware-dependent.
- Full training uses far more accelerator memory than inference-style forward-only because training keeps activations for backward.
- Activation checkpointing reduced saved-tensor estimates but increased backward time, matching the expected memory/compute tradeoff.

## 2.1.3 End-To-End Benchmarking

Files:

```text
outputs/mac_2_1_3_forward.json
outputs/mac_2_1_3_forward_backward.json
outputs/mac_2_1_3_train.json
```

Measured means:

| Run | Forward mean | Backward mean | Optimizer mean | Total mean |
| --- | ---: | ---: | ---: | ---: |
| forward only | 0.002824 s | n/a | n/a | 0.002824 s |
| forward + backward | 0.002239 s | 0.002999 s | n/a | 0.005238 s |
| train | 0.002526 s | 0.003177 s | 0.002528 s | 0.008231 s |

Noticeable details:

- Backward is about `1.26x` the training-run forward time here (`0.003177 / 0.002526`).
- Optimizer time is roughly the same size as forward time on this tiny model.
- Full training is about `2.9x` forward-only total time, because it adds backward plus optimizer work.

Conceptual interpretation:

- Forward-only inference only evaluates the model.
- Training has to retain or recompute intermediate values, traverse the graph backward, compute gradients, and update parameters.
- On larger models, matmuls and activation memory dominate more strongly; on tiny models, fixed overheads and optimizer overhead are more visible.

Possible writeup sentence:

```text
On the local tiny MPS run, forward-only took about 2.8 ms, while a full training step took about 8.2 ms. The backward pass was slightly more expensive than the forward pass, and AdamW added another forward-pass-sized cost because it updates every parameter and optimizer state.
```

## Warmup Effect

Files:

```text
outputs/mac_2_1_3_warmup0.json
outputs/mac_2_1_3_warmup1.json
outputs/mac_2_1_3_warmup2.json
```

Measured total means:

| Warmup steps | Total mean | Comment |
| ---: | ---: | --- |
| 0 | 0.018866 s | much slower/noisier |
| 1 | 0.008354 s | close to steady state |
| 2 | 0.008620 s | close to steady state |

Noticeable details:

- No-warmup total time is more than `2x` the warmed-up total time.
- No-warmup also had much higher variance in the raw JSON.
- One or two warmup steps were already enough for this tiny local run.

Conceptual interpretation:

- First iterations include one-time costs: allocator setup, lazy backend initialization, cache effects, and MPS/PyTorch internal setup.
- Warmup removes those first-iteration costs from the measured sample.
- On CUDA, warmup is also important for library/kernel autotuning and asynchronous execution effects.

Possible writeup sentence:

```text
Without warmup, the measured step time was substantially larger and more variable. After one or two warmup steps, the timing stabilized, suggesting that the initial measurements were dominated by one-time setup and allocation overhead rather than steady-state model execution.
```

## 2.1.5 Mixed Precision

Files:

```text
outputs/mac_2_1_5_accumulation.json
outputs/mac_2_1_5_toy_autocast.json
outputs/mac_2_1_5_fp32.json
outputs/mac_2_1_5_bf16.json
```

Accumulation results:

| Case | Result |
| --- | ---: |
| FP32 accumulator, FP32 addend | 10.00013 |
| FP16 accumulator, FP16 addend | 9.95313 |
| FP32 accumulator, FP16 addend | 10.00214 |
| FP32 accumulator, FP16 addend cast to FP32 | 10.00214 |

Autocast dtype observations:

- Parameters stayed `torch.float32`.
- First linear output was `torch.bfloat16`.
- LayerNorm output was `torch.float32`.
- Logits were `torch.bfloat16`.
- Loss was `torch.float32`.
- Gradients were `torch.float32`.

FP32 vs BF16 timing:

| Dtype | Forward mean | Backward mean | Optimizer mean | Total mean |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 0.002308 s | 0.002939 s | 0.002498 s | 0.007745 s |
| BF16 autocast | 0.003073 s | 0.003878 s | 0.002602 s | 0.009554 s |

Noticeable details:

- FP16 accumulation drifted downward to `9.95313`, demonstrating precision loss.
- BF16 autocast was slower than FP32 on this MPS run.
- LayerNorm stayed FP32 under autocast, matching the idea that normalization/reduction operations are numerically sensitive.

Conceptual interpretation:

- Mixed precision is not automatically faster on every backend. NVIDIA Tensor Cores are designed for low-precision matmuls; Apple MPS behavior differs.
- BF16 is usually more numerically stable than FP16 because it has a wider exponent range, but speed depends on hardware and kernel support.
- Autocast keeps sensitive operations and accumulated quantities in higher precision where needed.

Possible writeup sentence:

```text
The accumulation experiment shows why low-precision accumulation can be inaccurate: repeatedly adding 0.01 in FP16 produced 9.953 rather than 10. On the local MPS benchmark, BF16 autocast was slower than FP32, so these numbers should be treated as a correctness and dtype-behavior check rather than evidence for CUDA Tensor Core performance.
```

## 2.1.6 Memory Profiling

Files:

```text
outputs/mac_2_1_6_forward_memory.json
outputs/mac_2_1_6_train_memory.json
outputs/mac_2_1_6_residual_stream_size.json
```

Forward-only versus training memory:

| Run | Peak RSS | Peak accelerator allocated | Saved tensors |
| --- | ---: | ---: | ---: |
| forward-only, no grad | 328.11 MiB | 11.93 MiB | n/a |
| full training | 349.19 MiB | 74.38 MiB | 43.08 MiB/step |

Noticeable details:

- Training peak accelerator allocation is about `6.2x` forward-only allocation here.
- The saved tensors estimate is `43.08 MiB/step`, which is a large part of the training-only memory.
- RSS also rises, but accelerator allocation shows the training/inference difference more clearly.

Residual-stream size:

```text
xl, batch=4, context=2048, d_model=2560, FP32 = 80 MiB
```

Conceptual interpretation:

- Inference-style forward with `no_grad` does not need to save activations for backward.
- Training must retain tensors needed by autograd, and those tensors scale with batch size, sequence length, hidden size, and number of layers.
- Attention can add even worse memory pressure because attention matrices scale quadratically with sequence length.

Possible writeup sentence:

```text
The local memory run shows the expected training/inference gap: forward-only used about 11.9 MiB of sampled accelerator allocation, while full training used about 74.4 MiB and saved about 43.1 MiB of tensors per step for backward. This reflects the cost of retaining activations and other intermediates for autograd.
```

Mac caveat:

```text
These are sampled MPS/process-memory numbers, not official CUDA memory snapshots from torch.cuda.memory._dump_snapshot.
```

## 3.1 Autograd Residuals

File:

```text
outputs/mac_3_1_residuals.json
```

Measured result:

```text
saved_tensors.average_per_step_mib = 23.45 MiB
```

Noticeable details:

- Even a tiny model at context length 128 saves tens of MiB of tensors.
- Saved-tensor memory is not just parameters; it includes intermediate activations and operation-specific saved values needed by backward.

Conceptual interpretation:

- Autograd residuals explain why training memory is much larger than inference memory.
- Fusion can reduce saved intermediates by making a sequence of operations appear as one larger operation to autograd/compiler machinery.
- Attention residuals are especially costly because naive attention saves sequence-by-sequence matrices.

Possible writeup sentence:

```text
The saved-tensor hook reported about 23.4 MiB of saved tensors per step for the tiny local configuration, illustrating that backward memory is driven by retained activations/intermediates rather than just model parameters.
```

## 3.2 Activation Checkpointing

Files:

```text
outputs/mac_3_2_checkpoint_none.json
outputs/mac_3_2_checkpoint_every1.json
outputs/mac_3_2_checkpoint_every2.json
```

Measured results:

| Checkpointing | Total mean | Backward mean | Peak accelerator allocated | Saved tensors |
| --- | ---: | ---: | ---: | ---: |
| none | 0.009444 s | 0.003975 s | 85.31 MiB | 43.08 MiB/step |
| every 1 layer | 0.011607 s | 0.005919 s | 74.89 MiB | 25.17 MiB/step |
| every 2 layers | 0.011856 s | 0.005839 s | 78.51 MiB | 25.04 MiB/step |

Noticeable details:

- Checkpointing reduced saved tensors from `43.08 MiB` to about `25 MiB`.
- Checkpointing increased total time from `9.44 ms` to about `11.6-11.9 ms`.
- Peak sampled accelerator memory went down from `85.31 MiB` to `74.89 MiB` for checkpoint-every-1.
- Checkpoint-every-2 did not improve peak memory as much in this tiny run, likely because the model only has 2 layers and sampling is coarse.

Conceptual interpretation:

- This is the expected memory/compute tradeoff: save fewer tensors in forward, recompute them during backward.
- Backward gets slower because it includes recomputation.
- The benefit is clearer on larger/deeper models; tiny models can be dominated by fixed overhead and sampling noise.

Possible writeup sentence:

```text
Checkpointing reduced the saved-tensor estimate from about 43.1 MiB/step to about 25 MiB/step, but increased total step time from about 9.4 ms to about 11.6 ms. This matches the expected tradeoff: lower activation memory at the cost of extra recomputation during backward.
```

## 4.1 PyTorch Attention Benchmarking

File:

```text
outputs/mac_4_1_attention_pytorch.json
```

Notable local measurements:

| d_model | seq_len | Forward mean | Backward mean |
| ---: | ---: | ---: | ---: |
| 16 | 128 | 0.000483 s | 0.000667 s |
| 16 | 512 | 0.000530 s | 0.000811 s |
| 64 | 128 | 0.000371 s | 0.000586 s |
| 64 | 512 | 0.000634 s | 0.000840 s |
| 128 | 128 | 0.000337 s | 0.000461 s |
| 128 | 512 | 0.000711 s | 0.001038 s |

Noticeable details:

- The largest local configuration measured here, `d_model=128, seq_len=512`, had noticeably higher forward and backward time than the smaller sequence lengths.
- The trend is not perfectly monotonic at tiny sizes because MPS overhead, caching, and small-kernel effects dominate.
- Conceptually, vanilla attention should become much worse at long contexts because the attention matrix scales as `seq_len ** 2`.

Conceptual interpretation:

- Naive attention materializes score/probability matrices of shape roughly `batch * heads * seq_len * seq_len`.
- Memory and IO costs grow quadratically with sequence length.
- FlashAttention addresses this by tiling, online softmax, recomputation, and fusion so the full attention matrix is not written to global memory.

Possible writeup sentence:

```text
On the small MPS sweep, attention timings generally increased at the largest sequence length, although small-kernel overhead makes the trend noisy. The conceptual scaling issue remains the quadratic attention matrix: vanilla attention needs memory proportional to seq_len^2, which is what FlashAttention avoids.
```

## 4.2 Torch Compile

Files:

```text
outputs/mac_4_2_compile_transformer.json
outputs/mac_4_2_compile_attention.json
```

Compiled Transformer result:

| Run | Forward mean | Backward mean | Optimizer mean | Total mean |
| --- | ---: | ---: | ---: | ---: |
| uncompiled train | 0.002526 s | 0.003177 s | 0.002528 s | 0.008231 s |
| compiled train | 0.002170 s | 0.003276 s | 0.003832 s | 0.009278 s |

Noticeable details:

- Compiled forward was slightly faster in this run.
- Compiled total step was slightly slower because optimizer/backward/overhead dominated.
- Compile behavior on MPS is not necessarily predictive of CUDA behavior.

Conceptual interpretation:

- `torch.compile` can improve some computation by fusing/optimizing graph regions.
- Compilation does not necessarily improve every part of the step.
- Profiling attribution can become harder because operations may be fused.

Possible writeup sentence:

```text
In the local tiny run, compiling the Transformer slightly reduced forward time but did not reduce total step time. This suggests that compile benefits depend on backend, graph shape, and whether the optimized region dominates the total workload.
```

## 5.1 Local Distributed Communication

The local all-reduce output was printed in the batch run. For the latest run:

| Tensor size | Rank 0 mean | Rank 1 mean |
| ---: | ---: | ---: |
| 1 MiB | 0.000789 s | 0.000782 s |
| 10 MiB | 0.004289 s | 0.004285 s |

Noticeable details:

- Increasing tensor size from 1 MiB to 10 MiB increased all-reduce time by about `5.4x`.
- Rank timings were close to each other, which is expected because collective operations synchronize participating ranks.

Conceptual interpretation:

- Communication time grows with bytes transferred.
- Collective operations are limited by bandwidth, latency, backend implementation, and synchronization.
- CPU/Gloo is useful for understanding logic, but final assignment communication results need CUDA/NCCL.

Possible writeup sentence:

```text
The local Gloo all-reduce benchmark shows the expected size dependence: the 10 MiB tensor took substantially longer than the 1 MiB tensor, and both ranks reported similar timings because all-reduce synchronizes participants.
```

## 8 Parallelism Math

File:

```text
outputs/mac_8_parallelism_calculator.json
```

Key formulas:

```text
ring all-gather or reduce-scatter: (N - 1) / N * S / W
ring all-reduce: 2 * (N - 1) / N * S / W
```

For the calculator's default example:

```text
estimated backward compute time = 3.46e-05 s
estimated gradient all-reduce time = 9.47e-03 s
communication_bottlenecked = true
```

Conceptual interpretation:

- This example is communication-bottlenecked because the estimated gradient all-reduce time is much larger than the estimated per-rank backward compute time.
- Data parallelism becomes less efficient as communication dominates the compute it overlaps with.
- FSDP and TP change what is communicated: FSDP communicates weights/gradient shards, while TP communicates activations.

Possible writeup sentence:

```text
Using the simplified ring model, the example is communication-bottlenecked because the gradient all-reduce estimate is much larger than the local backward compute estimate. This illustrates why scaling data parallelism eventually requires sharding or additional parallelism strategies.
```

## Recommended Conclusions For The Writeup

Use these carefully, with the Mac caveat:

```text
All local measurements were collected on Apple MPS rather than CUDA, so I use them to validate trends and the benchmarking harness rather than as final GPU profiler results.
```

```text
The strongest local trend is that training costs substantially more time and memory than inference-style forward-only execution, due to backward computation, optimizer updates, and tensors saved for autograd.
```

```text
Warmup is necessary for stable timing: without warmup, the first measured steps include one-time setup overheads and produce higher mean and variance.
```

```text
Activation checkpointing shows the expected tradeoff: it reduces saved activation memory but increases backward/total runtime because the forward computation is partially repeated during backward.
```

```text
Mixed precision behavior is hardware-dependent. On this Mac run, BF16 autocast was slower than FP32, but the dtype experiment still confirms the intended autocast behavior: parameters and gradients remain FP32 while selected operations run in BF16.
```

## What Not To Overclaim

Do not claim:

- These are CUDA kernel timings.
- These replace Nsight screenshots.
- These prove Tensor Core mixed-precision speedups.
- These are valid B200 leaderboard numbers.
- These are NCCL distributed communication results.

Do claim:

- The harness works.
- The trends match the systems concepts.
- The Mac results give small-scale evidence for timing, memory, checkpointing, mixed precision dtype behavior, attention scaling intuition, and communication-size dependence.
