# Training open MoEs with SonicMoE

## Which model to use?

| Goal | Model | HF repo | SonicMoE fit |
|------|--------|---------|----------------|
| **Best match for this stack** | OLMoE-1B-7B | `allenai/OLMoE-1B-7B-0924` | Exact (64×top-8, SwiGLU, interleaved experts) |
| **Chat / instruct** | OLMoE-1B-7B-Instruct | `allenai/OLMoE-1B-7B-0924-Instruct` | Same arch, SFT weights |
| **Best quality ≤8B (2026)** | ZAYA1-8B | `Zyphra/ZAYA1-8B` | Custom MoE++; loader not implemented |
| **On-device / tools** | LFM2.5-8B-A1B | `LiquidAI/LFM2.5-8B-A1B` | Hybrid conv+GQA; not SonicMoE-shaped |
| **Larger + LoRA** | Qwen3-30B-A3B | `Qwen/Qwen3-30B-A3B` | Needs weight adapter + 2× GPU LoRA |

**Recommendation:** Start with **OLMoE** for pretrained weights + SonicMoE kernels. Add ZAYA1 or Qwen later if you need higher benchmark scores and can invest in porting.

## Commands

```bash
cd /home/shadeform/ai_assisted_sandbox

# Install HF deps (once)
uv pip install huggingface_hub safetensors transformers

# Smoke: random init, 2 steps
uv run python cs336_moe/train.py --skip-load --steps 2 --ctx-len 512

# Real: load OLMoE-7B base + train (downloads ~14GB weights first time)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python cs336_moe/train.py --ctx-len 2048 --steps 10 --fused-ce

# Instruct checkpoint
uv run python cs336_moe/train.py \
  --hf-repo allenai/OLMoE-1B-7B-0924-Instruct --ctx-len 2048 --steps 10 --fused-ce

# 2× GPU DDP + fused CE + compile + fused AdamW (default)
# Max throughput @ ctx=32K: use --batch-size 6 (3 samples/GPU)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python cs336_moe/train.py --skip-load --steps 5 --ctx-len 32768 \
  --batch-size 6 --fused-ce --compile --ddp

# Real HF weights on 2 GPUs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python cs336_moe/train.py --ctx-len 8192 --steps 10 --fused-ce --compile --ddp
```

## What runs

- **Attention:** FA4 CuTe + OLMoE Q/K RMSNorm (`cs336_moe/attention.py`)
- **FFN:** `sonicmoe.MoE` with `KernelBackendMoE.sonicmoe`
- **Weights:** `cs336_moe/load_hf.py` maps HF expert `gate_proj`/`up_proj` → interleaved `c_fc`
