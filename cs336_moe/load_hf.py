"""Load Allen AI OLMoE checkpoints into SonicMoETransformerLM."""

from __future__ import annotations

import json
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from cs336_moe.model import SonicMoETransformerLM

OLMOE_7B_REPO = "allenai/OLMoE-1B-7B-0924"
OLMOE_7B_INSTRUCT_REPO = "allenai/OLMoE-1B-7B-0924-Instruct"


def olmoe_hf_config(repo_id: str = OLMOE_7B_REPO) -> dict[str, Any]:
    path = hf_hub_download(repo_id, "config.json")
    with open(path) as f:
        return json.load(f)


def _interleave_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """HF separate gate/up [I, H] -> SonicMoE interleaved [2I, H]."""
    i, h = gate.shape
    out = torch.empty(2 * i, h, dtype=gate.dtype, device=gate.device)
    out[0::2] = gate
    out[1::2] = up
    return out


class _ShardCache:
    """Lazy-open safetensor shards (experts are split across files)."""

    def __init__(self, repo_id: str, weight_map: dict[str, str], device: torch.device):
        self.repo_id = repo_id
        self.weight_map = weight_map
        self.device = device
        self._handles: dict[str, safe_open] = {}

    def get(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        if shard not in self._handles:
            path = hf_hub_download(self.repo_id, shard)
            self._handles[shard] = safe_open(path, framework="pt", device=str(self.device))
        return self._handles[shard].get_tensor(key)

    def close(self) -> None:
        self._handles.clear()


def _copy_layer_experts(
    shards: _ShardCache,
    model: SonicMoETransformerLM,
    layer_idx: int,
    *,
    dtype: torch.dtype,
) -> None:
    prefix = f"model.layers.{layer_idx}.mlp.experts"
    num_experts = model.num_experts
    gate0 = shards.get(f"{prefix}.0.gate_proj.weight").to(dtype=dtype)
    i_dim, h_dim = gate0.shape
    c_fc = torch.empty(num_experts, 2 * i_dim, h_dim, dtype=dtype, device=gate0.device)
    c_proj = torch.empty(num_experts, h_dim, i_dim, dtype=dtype, device=gate0.device)

    for e in range(num_experts):
        gate = shards.get(f"{prefix}.{e}.gate_proj.weight").to(dtype=dtype)
        up = shards.get(f"{prefix}.{e}.up_proj.weight").to(dtype=dtype)
        down = shards.get(f"{prefix}.{e}.down_proj.weight").to(dtype=dtype)
        c_fc[e] = _interleave_gate_up(gate, up)
        c_proj[e] = down

    block = model.layers[layer_idx]
    block.moe.c_fc.weight.data.copy_(c_fc)
    block.moe.c_proj.weight.data.copy_(c_proj)
    router_key = f"model.layers.{layer_idx}.mlp.gate.weight"
    block.moe.router.weight.data.copy_(shards.get(router_key).to(dtype=dtype))


def load_olmoe_weights(
    model: SonicMoETransformerLM,
    repo_id: str = OLMOE_7B_REPO,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> SonicMoETransformerLM:
    """Copy HF OLMoE safetensors into a SonicMoE-backed model (in-place)."""
    cfg = olmoe_hf_config(repo_id)
    if cfg["hidden_size"] != model.hidden_size:
        raise ValueError("hidden_size mismatch")
    if cfg["num_hidden_layers"] != model.num_layers:
        raise ValueError("num_layers mismatch")
    if cfg["num_experts"] != model.num_experts:
        raise ValueError("num_experts mismatch")
    if cfg["num_experts_per_tok"] != model.num_experts_per_tok:
        raise ValueError("num_experts_per_tok mismatch")
    if model.vocab_size != cfg["vocab_size"]:
        raise ValueError(f"model.vocab_size={model.vocab_size} != HF vocab_size={cfg['vocab_size']}")

    dev = device or model.token_embeddings.weight.device
    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]

    shards = _ShardCache(repo_id, weight_map, dev)
    for layer_idx in range(model.num_layers):
        _copy_layer_experts(shards, model, layer_idx, dtype=dtype)

    tensors: dict[str, torch.Tensor] = {}
    for key in weight_map:
        if ".mlp.experts." in key or key.endswith(".mlp.gate.weight"):
            continue
        tensors[key] = shards.get(key).to(device=dev, dtype=dtype)
    shards.close()

    model.token_embeddings.weight.data.copy_(tensors["model.embed_tokens.weight"])
    model.ln_final.weight.data.copy_(tensors["model.norm.weight"])
    if "lm_head.weight" in tensors:
        model.lm_head.weight.data.copy_(tensors["lm_head.weight"])
    else:
        model.lm_head.weight.data.copy_(tensors["model.embed_tokens.weight"])

    for layer_idx, layer in enumerate(model.layers):
        p = f"model.layers.{layer_idx}"
        layer.ln1.weight.data.copy_(tensors[f"{p}.input_layernorm.weight"])
        layer.ln2.weight.data.copy_(tensors[f"{p}.post_attention_layernorm.weight"])
        attn = layer.attn
        attn.q_proj.weight.data.copy_(tensors[f"{p}.self_attn.q_proj.weight"])
        attn.k_proj.weight.data.copy_(tensors[f"{p}.self_attn.k_proj.weight"])
        attn.v_proj.weight.data.copy_(tensors[f"{p}.self_attn.v_proj.weight"])
        attn.output_proj.weight.data.copy_(tensors[f"{p}.self_attn.o_proj.weight"])
        attn.q_norm.weight.data.copy_(tensors[f"{p}.self_attn.q_norm.weight"])
        attn.k_norm.weight.data.copy_(tensors[f"{p}.self_attn.k_norm.weight"])

    return model


def build_olmoe_7b_from_hf(
    repo_id: str = OLMOE_7B_REPO,
    *,
    context_length: int = 4096,
    kernel_backend_moe=None,
    checkpoint_every: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> SonicMoETransformerLM:
    """Build SonicMoE LM with OLMoE HF weights."""
    from sonicmoe import KernelBackendMoE

    from cs336_moe.model import build_olmoe_7b_lm

    cfg = olmoe_hf_config(repo_id)
    if kernel_backend_moe is None:
        kernel_backend_moe = KernelBackendMoE.sonicmoe

    model = build_olmoe_7b_lm(
        vocab_size=cfg["vocab_size"],
        context_length=context_length,
        kernel_backend_moe=kernel_backend_moe,
        checkpoint_every=checkpoint_every,
        olmoe_attention=True,
    )
    if device is not None:
        model = model.to(device=device, dtype=dtype)
    load_olmoe_weights(model, repo_id, device=device, dtype=dtype)
    return model
