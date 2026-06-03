# SonicMoE ~7B Training Step — Progress Notes

OLMoE-style MoE model (`cs336_moe/`) with **SonicMoE FFN kernels** + **FA4 CuTeDSL attention**.

**Not the CS336 dense leaderboard config** — different architecture (16 layers, H=2048, 64 experts / top-8).

**Harness:** `cs336_moe/benchmark.py` — same step shape as leaderboard (forward + CE + aux loss + backward + AdamW).

---

## Model preset (`build_olmoe_7b_lm`)

| Field | Value |
|-------|-------|
| Params | ~7.34B |
| Layers | 16 |
| hidden_size | 2048 |
| intermediate_size | 1024 |
| num_experts | 64 |
| top_k | 8 |
| Attention | FA4 CuTe (`FlashCausalMultiHeadSelfAttention`) |
| FFN | SonicMoE (`KernelBackendMoE.sonicmoe`) |
| batch / ctx / vocab | 2 / 32768 / 151936 (same as leaderboard) |

**Install:** `sonic-moe` from `/home/shadeform/sonic-moe` (editable pip install; not in `pyproject.toml` yet).

---

## Results (1× B300, `--quick` bench)

| Config | ctx | Step time | Peak VRAM | Notes |
|--------|-----|-----------|-----------|-------|
| FA4 + SonicMoE | 2048 | **149 ms** | 58 GiB | Smoke test |
| FA4 + SonicMoE + **fused CE** | **32768** | **1,322 ms** | 170 GiB | No checkpoint |
| + **`torch.compile`** (forward_hidden) | 32768 | **1,146 ms** | 145 GiB | Best 1-GPU |
| **2× B300 DDP** + fused CE + compile | 32768 | **639 ms** | 107 GiB/rank | batch split 1/GPU |
| **2× DDP + fused CE + compile + fused AdamW** | 32768 | **587 ms** | 108 GiB/rank | **best so far** |
| Full logits (no fused CE) | 32768 | **1,220 ms** | 205 GiB | Fits on B300 |
| Dense leaderboard (ref) | 32768 | ~5,980 ms | 139 GiB | FA4 + ckpt=1, 34 layers |

MoE @ ctx=32768 is **~4.5–9.4× faster** than dense @ same seq length, **without gradient checkpointing**.
2-GPU DDP gives **~1.8×** over 1-GPU compile (639 vs 1,146 ms).
**Fused AdamW** saves another **~8%** (639 → 587 ms) on top of DDP + compile.

**Best throughput:** 65,536 tok/step ÷ 0.587 s ≈ **112K tok/s** @ ctx=32K, batch=2, 2× B300.

### Batch sweep @ ctx=32K (2× DDP, fused CE + compile + fused AdamW)

| Global batch | Local/GPU | Step time | tok/s | VRAM/rank |
|--------------|-----------|-----------|-------|-----------|
| 2 | 1 | 589 ms | 111K | 121 GiB |
| 4 | 2 | 1,155 ms | 113K | 159 GiB |
| **6** | **3** | **1,714 ms** | **115K** | **210 GiB** |
| 8 | 4 | OOM | — | — |

**Max throughput: batch=6** (~3% over batch=2). Step time scales ~linearly with batch; gains are modest once MoE kernels are warm. VRAM ceiling is ~210 GiB/rank before OOM @ batch=8.

---

## Commands

```bash
cd /home/shadeform/ai_assisted_sandbox

# Quick smoke (ctx=2048)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python cs336_moe/benchmark.py --quick --ctx-len 2048

# Full ctx=32768 (first run slow — SonicMoE autotune)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python cs336_moe/benchmark.py --quick --fused-ce \
  --output-json outputs/sonic_moe_fa4_fused_ce_ctx32768_quick.json

# Optional: compile forward_hidden (with --fused-ce)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python cs336_moe/benchmark.py --quick --fused-ce --compile

# 2× GPU DDP (best so far)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python cs336_moe/benchmark.py --ddp --quick --fused-ce --compile \
  --output-json outputs/sonic_moe_ddp2_fused_compile_ctx32768_quick.json

# + fused AdamW (default on; ~587 ms quick)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python cs336_moe/benchmark.py --ddp --quick --fused-ce --compile \
  --output-json outputs/sonic_moe_ddp2_fused_ce_compile_fusedadamw_ctx32768_quick.json
```

### Flags

| Flag | Purpose |
|------|---------|
| `--ddp` | 2-GPU DDP (splits batch, all-reduces grads) |
| `--fused-ce` | Chunked LM-head + CE (avoids full logits tensor) |
| `--checkpoint-every N` | Layer checkpointing if OOM |
| `--compile` | `torch.compile` on model or `forward_hidden` (with fused CE) |
| `--fused-adamw` / `--no-fused-adamw` | PyTorch fused CUDA AdamW (default: on) |
| `--kernel-backend torch` | Fallback PyTorch MoE (debug) |
| `--kernel-warmup-steps N` | Untimed steps before timing (autotune) |

---

## Changelog

| Date | Update |
|------|--------|
| 2026-06-03 | Initial SonicMoE + FA4 stack; smoke @ ctx=2048 |
| 2026-06-03 | Added `forward_hidden`, fused CE, compile, checkpoint flags |
| 2026-06-03 | **ctx=32768 + fused CE: 1,322 ms**, 170 GiB, no ckpt |
| 2026-06-03 | **ctx=32768 + fused CE + compile: 1,146 ms**, 145 GiB |
| 2026-06-03 | ctx=32768 full logits: 1,220 ms, 205 GiB |
| 2026-06-03 | **2× B300 DDP + fused CE + compile: 639 ms**, 107 GiB/rank |
| 2026-06-03 | **+ fused AdamW: 587 ms**; wired into `train.py` + `benchmark.py` |
