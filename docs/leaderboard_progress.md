# CS336 Assignment 2 Leaderboard — Progress Notes

Working notes for optimizing the full training-step benchmark (8B config, batch=2, seq=32768, BF16, 2 GPUs).

**Target:** beat the verified naive baseline of **10,000 ms** on 2× B200 (we have 2× B300).

**Official timing harness:** `triton.testing.do_bench(train_step, warmup=10_000, rep=30_000)` over one full step (forward + backward + AdamW).

---

## Environment

| Item | Value |
|------|-------|
| Host | `brev-8mg5uaos0` (Brev) |
| GPUs | 2× NVIDIA B300 SXM6 AC (~275 GiB VRAM each) |
| CPU / RAM | AMD EPYC 9575F (60 cores), 501 GiB |
| OS | Ubuntu 24.04.4 LTS |
| PyTorch | 2.11.0, CUDA 13.0 |
| Repo | [`ai_assisted_sandbox`](https://github.com/CuriousCaliBoi/ai_assisted_sandbox) cloned to `/home/shadeform/ai_assisted_sandbox` |
| Benchmark script | `scripts/naive_leaderboard_benchmark.py` |

---

## Leaderboard config (Section 9)

```python
ctx_len = 32768
vocab_size = 151936
d_model = 4096
d_ff = 11008
num_layers = 34
num_heads = 32
batch_size = 2
dtype = bfloat16
is_causal = True
```

### Reference times (Spring 2026, 2× B200)

| Tier | Name | Time |
|------|------|------|
| Top | Keshav Patel Keval | 3,837 ms |
| ~Median | Daphne Barretto | 6,606 ms |
| Verified baseline | naive baseline | 10,000 ms |
| Slowest listed | Jason Meng | 9,555 ms |

---

## Experiments

### 1. Naive `cs336_basics` harness (no optimizations)

**What we tried:** Faithful port of the official timing code using unmodified `BasicsTransformerLM`, `AdamW`, and `cross_entropy` from `cs336_basics`.

**Script flags:** default (no `--compile`)

**README fixes applied in our script:**
- `labels, targets = torch.randint(...)` → two separate `(batch, ctx_len)` tensors
- `BasicsTransformerLM(Config())` → map `ctx_len` → `context_length`

| ctx_len | Result | Step time | Peak VRAM | Notes |
|---------|--------|-----------|-----------|-------|
| 32768 | **OOM** | — | — | Tries to allocate **128 GiB** attention score tensor `(2, 32, 32768, 32768)` on layer 1 |
| 4096 | OK | **1,622 ms** | 251 GiB | GPU nearly full |
| 2048 | OK | **622 ms** | 114 GiB | `--quick` warmup/rep |

**Output files:**
- `outputs/naive_leaderboard_ctx2048.json`
- `outputs/naive_leaderboard_ctx4096.json`

**Takeaway:** Raw `cs336_basics` attention materializes O(n²) scores. Cannot run the real leaderboard config on a single GPU, even with 275 GiB. The 10 s “naive baseline” on the leaderboard is **not** this code — it must include enough optimization (flash attention, sharding, fused CE, etc.) to fit.

---

### 2. `torch.compile` on the naive model

**What we tried:** `--compile` wraps `BasicsTransformerLM` with `torch.compile(model, mode="default")`.

| ctx_len | No compile | With compile | Speedup | Peak VRAM (compile) |
|---------|------------|--------------|---------|---------------------|
| 32768 | OOM | **OOM** | — | Inductor still allocates `(2, 32, 32768, 32768)` bf16 buffer |
| 4096 | 1,622 ms | **673 ms** | ~2.4× | 225 GiB |
| 2048 | 622 ms | **345 ms** | ~1.8× | 101 GiB |

**Output files:**
- `outputs/naive_leaderboard_compile_ctx2048.json`
- `outputs/naive_leaderboard_compile_ctx4096.json`

**Takeaway:** Compile speeds up the **same algorithm** (~2× at ctx 4096) but does **not** change memory complexity. Still OOM at 32768. Worth keeping as an extra layer after flash attention + fused loss.

**Command:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/naive_leaderboard_benchmark.py --compile --quick --ctx-len 4096
```

---

## Not yet tried

- [ ] Flash attention (PyTorch SDPA or Triton forward + backward)
- [ ] Fused LM head + cross-entropy (avoid full `[2, 32768, 151936]` logits)
- [ ] 2-GPU FSDP / DDP across both B300s
- [ ] Fused AdamW
- [ ] Activation checkpointing (memory ↔ speed tradeoff)
- [ ] `torch.compile` on top of flash + fused CE
- [ ] Triton tile autotuning (watch 10-minute benchmark limit)
- [ ] Full official `warmup=10_000, rep=30_000` timing on working config

---

## Key bottlenecks identified

1. **Attention:** `(batch × heads × seq × seq)` score matrix = 128 GiB/layer at seq 32768 — root cause of OOM.
2. **Logits:** `(2, 32768, 151936)` bf16 ≈ 20 GiB if materialized.
3. **Activations:** 34 layers × long seq × wide hidden dim — fills remaining VRAM even at ctx 4096 (~251 GiB).
4. **Single GPU:** Leaderboard expects 2 GPUs; we have only benchmarked on `cuda:0` so far.

---

## Next steps (planned)

1. Implement flash attention path so ctx_len=32768 fits in memory.
2. Fuse cross-entropy with LM head to drop full logits tensor.
3. Spread model across 2× B300 with FSDP.
4. Re-run timing harness; compare against 10 s baseline.
5. Layer on `torch.compile` and kernel tuning.

---

## Changelog

| Date | Update |
|------|--------|
| 2026-06-02 | Cloned repo; set up `uv` env on 2× B300 machine |
| 2026-06-02 | Added `scripts/naive_leaderboard_benchmark.py` matching official timing harness |
| 2026-06-02 | Ran naive baseline: OOM @ 32768; 622 ms @ 2048; 1,622 ms @ 4096 |
| 2026-06-02 | Added `--compile`: ~2× faster @ 2048/4096; still OOM @ 32768 |
| 2026-06-02 | **FA4 CuTeDSL** (`flash-attn-4==4.0.0b15`): attention-only @ ctx=32768 uses ~5 GiB; full model still OOM without checkpointing |
| 2026-06-02 | **CuTe FA + checkpoint every layer @ ctx=32768:** **5,980 ms** step, 139 GiB peak — beats 10 s baseline on 1× B300 |
| 2026-06-02 | **2-GPU FSDP implemented** (`cs336_systems/fsdp.py`); 3/4 FSDP tests pass |
| 2026-06-02 | **2× B300 FSDP + CuTe FA @ ctx=8192:** **718 ms**, 144 GiB/rank (no checkpoint) |
| 2026-06-02 | **2× B300 FSDP @ ctx=32768:** OOM without fused CE; checkpoint+FSDP incompatible (weight gather changes shapes) |
| 2026-06-03 | **Fused CE + ckpt @ ctx=32768:** **6,086 ms**, **106 GiB** peak (saves ~33 GiB vs full logits) |
| 2026-06-03 | **`torch.compile` on CuTe FA stack:** ~1.56× @ ctx=8192 (797 vs 1,242 ms); **no gain** @ ctx=32768+ckpt (~6,010 ms); compile+hurt with fused CE+ckpt (~9,688 ms) |
| 2026-06-02 | Created this progress log |

---

### 3. FlashAttention-4 CuTeDSL (speed-of-light Blackwell kernels)

**What we tried:** Installed `flash-attn-4==4.0.0b15` (CuTeDSL / CUTLASS). Swapped attention in `BasicsTransformerLM` via `cs336_systems/flash_attention.py` + `FlashBasicsTransformerLM`.

**Script flags:** `--attention cute --checkpoint-every 1`

| Component | Result | Notes |
|-----------|--------|-------|
| FA4 attention only @ ctx=32768 | OK | Forward + backward, ~5 GiB VRAM |
| Full 8B step @ ctx=32768, no checkpoint | OOM | ~267 GiB — FFN activations + full logits |
| Full 8B step @ ctx=32768, checkpoint every layer | OK | **5,980 ms**, 139 GiB peak |

**Output:** `outputs/leaderboard_cute_ckpt1_ctx32768_quick.json`

**vs leaderboard (2× B200):**

| Entry | Time |
|-------|------|
| Top (Keshav) | 3,837 ms |
| Verified baseline | 10,000 ms |
| **Us (CuTe FA + ckpt, 1× B300, quick bench)** | **5,980 ms** |

**Takeaway:** CuTe FA4 is the right attention backend. Still need fused CE, 2-GPU sharding, and likely no checkpointing to compete for top spots. Gluon backward not available in upstream Triton examples yet.

**Command:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/naive_leaderboard_benchmark.py \
  --attention cute --checkpoint-every 1 --quick
```

**Memory profiling (PyTorch profiler):**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/naive_leaderboard_benchmark.py \
  --attention cute --checkpoint-every 1 --quick \
  --profile-memory \
  --memory-snapshot outputs/cute_fa_memory_snapshot.pickle \
  --output-json outputs/leaderboard_cute_profiled.json
```

Profiler output includes per-phase VRAM (`before_step`, `after_forward`, `after_loss`, `after_backward`, …), top ops by `self_device_memory_usage`, and full `torch.cuda.memory_summary()`. Upload the `.pickle` to [pytorch.org/memory_viz](https://pytorch.org/memory_viz) for a timeline view.

**Profiled run (ctx=32768, ckpt=1):** peak **139 GiB** during backward; logits materialization spikes to **121 GiB** at `after_loss` (fused CE target).

---

### 4. Fused LM-head + cross-entropy

**What we tried:** Chunked CE in `cs336_systems/fused_ce.py` — avoids materializing `[2, 32768, 151936]` logits.

**Script flags:** `--attention cute --checkpoint-every 1 --fused-ce`

| Config | Time | Peak VRAM |
|--------|------|-----------|
| CuTe FA + ckpt (full logits) | 5,980 ms | 139 GiB |
| CuTe FA + ckpt + **fused CE** | **6,086 ms** | **106 GiB** |

Fused CE saves ~33 GiB (logits spike gone) with ~100 ms overhead from chunked LM-head matmuls.

---

### 5. `torch.compile` on the CuTe FA stack

**Script flags:** add `--compile` (or `--compile-mode max-autotune`)

| Config | No compile | With compile | Notes |
|--------|------------|--------------|-------|
| ctx=8192, cute FA, no ckpt | 1,242 ms | **797 ms** | **1.56×** speedup; 182 → 133 GiB |
| ctx=32768, cute FA + ckpt=1 | 5,980 ms | 6,011 ms | No benefit — FA4 custom autograd + checkpoint limit Inductor |
| ctx=32768, cute FA + ckpt + fused CE | 6,086 ms | 9,688 ms | Compiling `forward_hidden` **slows** (~1.6×) |

**Takeaway:** `torch.compile` helps the FFN/linear regions when the full activation graph fits (ctx=8192). At ctx=32768 with checkpointing, compile is neutral or harmful. Top leaderboard entries likely avoid checkpointing (fused CE + 2-GPU) before layering compile.

**Command (best compile win so far):**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/naive_leaderboard_benchmark.py \
  --attention cute --ctx-len 8192 --compile --quick
```

---

## Not yet tried

- [ ] Gluon FA backward (upstream Triton only ships forward example)
- [x] Fused LM-head + cross-entropy (`--fused-ce`; chunked implementation)
- [x] 2-GPU FSDP across both B300s
- [x] `torch.compile` on CuTe FA stack (see section 5)
- [ ] Fused AdamW
- [ ] FSDP @ ctx=32768 without checkpointing (activation memory still too high)
- [ ] Full official `warmup=10_000, rep=30_000` timing

---

## Changelog

| Date | Update |
|------|--------|
| 2026-06-02 | Cloned repo; set up `uv` env on 2× B300 machine |
| 2026-06-02 | Added `scripts/naive_leaderboard_benchmark.py` matching official timing harness |
| 2026-06-02 | Ran naive baseline: OOM @ 32768; 622 ms @ 2048; 1,622 ms @ 4096 |
