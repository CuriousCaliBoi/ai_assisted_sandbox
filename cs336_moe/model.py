"""~8B decoder LM with SonicMoE FFN blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import (
    Embedding,
    Linear,
    RMSNorm,
    RotaryEmbedding,
)
from cs336_moe.attention import OlmoeFlashAttention
from cs336_systems.flash_attention import FlashCausalMultiHeadSelfAttention
from sonicmoe import KernelBackendMoE, MoE
from sonicmoe.enums import ActivationType


def moe_ffn_params(num_experts: int, hidden_size: int, intermediate_size: int) -> int:
    """Router + interleaved gate/up + down projections (no bias)."""
    return num_experts * hidden_size * (1 + 3 * intermediate_size)


def estimate_total_params(
    *,
    vocab_size: int,
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_heads: int,
) -> int:
    d_head = hidden_size // num_heads
    attn_params = 4 * hidden_size * hidden_size
    block_params = attn_params + moe_ffn_params(num_experts, hidden_size, intermediate_size)
    ln_params = 2 * hidden_size
    return (
        vocab_size * hidden_size
        + num_layers * (block_params + ln_params)
        + hidden_size
        + hidden_size * vocab_size
    )


class SonicMoETransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_experts: int,
        num_experts_per_tok: int,
        intermediate_size: int,
        positional_encoder: RotaryEmbedding | None,
        kernel_backend_moe: KernelBackendMoE,
        olmoe_attention: bool = False,
    ):
        super().__init__()
        attn_cls = OlmoeFlashAttention if olmoe_attention else FlashCausalMultiHeadSelfAttention
        self.attn = attn_cls(
            d_model=hidden_size,
            num_heads=num_heads,
            positional_encoder=positional_encoder,
        )
        self.moe = MoE(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        )
        self.ln1 = RMSNorm(hidden_size)
        self.ln2 = RMSNorm(hidden_size)
        self.kernel_backend_moe = kernel_backend_moe
        self.last_aux_loss: torch.Tensor | None = None

    def forward(self, x: Float[Tensor, " batch seq hidden"]) -> Float[Tensor, " batch seq hidden"]:
        x = x + self.attn(self.ln1(x))
        moe_out, aux_loss = self.moe(
            self.ln2(x),
            kernel_backend_moe=self.kernel_backend_moe,
            is_inference_mode=not self.training,
        )
        self.last_aux_loss = aux_loss
        return x + moe_out


class SonicMoETransformerLM(nn.Module):
    """
    OLMoE-scale (~7-8B) decoder LM using SonicMoE FFN + FA4 CuTeDSL attention.

    Default preset matches SonicMoE's 7B OLMoE benchmark:
    16 layers, H=2048, I=1024, E=64, top_k=8.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        context_length: int,
        hidden_size: int = 2048,
        num_layers: int = 16,
        num_heads: int = 16,
        intermediate_size: int = 1024,
        num_experts: int = 64,
        num_experts_per_tok: int = 8,
        rope_theta: float | None = 10_000.0,
        kernel_backend_moe: KernelBackendMoE = KernelBackendMoE.sonicmoe,
        checkpoint_every: int = 0,
        olmoe_attention: bool = False,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_experts % 8 != 0:
            raise ValueError("SonicMoE top-k kernel requires num_experts % 8 == 0")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.kernel_backend_moe = kernel_backend_moe
        self.checkpoint_every = checkpoint_every

        d_head = hidden_size // num_heads
        self.token_embeddings = Embedding(vocab_size, hidden_size)
        self.positional_encoder = (
            RotaryEmbedding(context_length, d_head, rope_theta) if rope_theta is not None else None
        )
        self.layers = nn.ModuleList(
            [
                SonicMoETransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    intermediate_size=intermediate_size,
                    positional_encoder=self.positional_encoder,
                    kernel_backend_moe=kernel_backend_moe,
                    olmoe_attention=olmoe_attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(hidden_size)
        self.lm_head = Linear(hidden_size, vocab_size)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def estimated_params(self) -> int:
        return estimate_total_params(
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.layers[0].moe.intermediate_size,
            num_experts=self.num_experts,
            num_heads=self.layers[0].attn.num_heads,
        )

    def load_balance_loss(self) -> torch.Tensor:
        losses = [layer.last_aux_loss for layer in self.layers if layer.last_aux_loss is not None]
        if not losses:
            device = self.token_embeddings.weight.device
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    def forward_hidden(self, x: Int[Tensor, " batch seq"]) -> Float[Tensor, " batch seq hidden"]:
        x = self.token_embeddings(x)
        if self.checkpoint_every <= 0:
            for layer in self.layers:
                x = layer(x)
        else:
            for start in range(0, len(self.layers), self.checkpoint_every):
                group = self.layers[start : start + self.checkpoint_every]

                def run_group(hidden: torch.Tensor, group=group) -> torch.Tensor:
                    for layer in group:
                        hidden = layer(hidden)
                    return hidden

                x = checkpoint(run_group, x, use_reentrant=False)
        return self.ln_final(x)

    def forward(
        self,
        x: Int[Tensor, " batch seq"],
        return_aux_loss: bool = False,
        return_hidden: bool = False,
    ) -> (
        Float[Tensor, " batch seq vocab"]
        | tuple[Float[Tensor, " batch seq vocab"], torch.Tensor]
        | Float[Tensor, " batch seq hidden"]
        | tuple[Float[Tensor, " batch seq hidden"], torch.Tensor]
    ):
        hidden = self.forward_hidden(x)
        if return_hidden:
            if return_aux_loss:
                return hidden, self.load_balance_loss()
            return hidden
        logits = self.lm_head(hidden)
        if return_aux_loss:
            return logits, self.load_balance_loss()
        return logits


def build_olmoe_7b_lm(
    *,
    vocab_size: int,
    context_length: int,
    kernel_backend_moe: KernelBackendMoE = KernelBackendMoE.sonicmoe,
    checkpoint_every: int = 0,
    olmoe_attention: bool = False,
) -> SonicMoETransformerLM:
    """OLMoE-style ~7B config used in SonicMoE blog benchmarks."""
    return SonicMoETransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        hidden_size=2048,
        num_layers=16,
        num_heads=16,
        intermediate_size=1024,
        num_experts=64,
        num_experts_per_tok=8,
        kernel_backend_moe=kernel_backend_moe,
        checkpoint_every=checkpoint_every,
        olmoe_attention=olmoe_attention,
    )
