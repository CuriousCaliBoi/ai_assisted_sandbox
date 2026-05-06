from __future__ import annotations

import logging

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor

from cs336_basics.model import (
    CausalMultiHeadSelfAttention,
    Embedding,
    Linear,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
)


logger = logging.getLogger(__name__)


class TopKRouter(nn.Module):
    """
    Token router for a sparse Mixture-of-Experts FFN.

    For every token representation, it produces one score per expert, turns those
    scores into probabilities, and keeps only the top-k experts for that token.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = Linear(d_model, num_experts)

    def forward(
        self, x: Float[Tensor, " batch seq d_model"]
    ) -> tuple[
        Float[Tensor, " batch seq num_experts"],
        Float[Tensor, " batch seq top_k"],
        Int[Tensor, " batch seq top_k"],
    ]:
        router_logits = self.router(x)
        router_probs = torch.softmax(router_logits, dim=-1)

        topk_probs, topk_experts = torch.topk(router_probs, k=self.top_k, dim=-1)

        # Renormalize among selected experts so each token's expert weights sum to 1.
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return router_probs, topk_probs, topk_experts


class SparseMoEFeedForward(nn.Module):
    """
    Sparse MoE replacement for the dense SwiGLU FFN.

    Dense Transformer block:
        x -> one SwiGLU FFN

    MoE Transformer block:
        x -> router -> top-k SwiGLU experts per token -> weighted expert sum

    This implementation is intentionally followable rather than kernel-efficient:
    it loops over selected experts so you can inspect the routing behavior.
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(d_model=d_model, num_experts=num_experts, top_k=top_k)
        self.experts = nn.ModuleList([SwiGLU(d_model=d_model, d_ff=d_ff) for _ in range(num_experts)])

        self.last_router_probs: torch.Tensor | None = None
        self.last_topk_probs: torch.Tensor | None = None
        self.last_topk_experts: torch.Tensor | None = None
        self.last_load_balance_loss: torch.Tensor | None = None

    def _load_balance_loss(
        self,
        router_probs: Float[Tensor, " batch seq num_experts"],
        topk_experts: Int[Tensor, " batch seq top_k"],
    ) -> Float[Tensor, ""]:
        """
        Small auxiliary loss used by many MoE systems.

        It encourages the router to spread tokens across experts instead of
        sending everything to one expert.
        """
        importance = router_probs.mean(dim=(0, 1))
        selected = torch.nn.functional.one_hot(topk_experts, num_classes=self.num_experts).to(router_probs.dtype)
        load = selected.sum(dim=-2).mean(dim=(0, 1)) / self.top_k
        return self.num_experts * torch.sum(importance * load)

    def forward(self, x: Float[Tensor, " batch seq d_model"]) -> Float[Tensor, " batch seq d_model"]:
        router_probs, topk_probs, topk_experts = self.router(x)

        output = torch.zeros_like(x)
        for rank in range(self.top_k):
            expert_for_token = topk_experts[..., rank]
            expert_weight = topk_probs[..., rank]

            for expert_idx, expert in enumerate(self.experts):
                token_mask = expert_for_token == expert_idx
                if not token_mask.any():
                    continue

                expert_input = x[token_mask]
                expert_output = expert(expert_input)
                output[token_mask] += expert_weight[token_mask].unsqueeze(-1) * expert_output

        self.last_router_probs = router_probs
        self.last_topk_probs = topk_probs
        self.last_topk_experts = topk_experts
        self.last_load_balance_loss = self._load_balance_loss(router_probs, topk_experts)
        return output


class MoETransformerBlock(nn.Module):
    """Transformer block with ordinary causal self-attention and an MoE FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        positional_encoder: RotaryEmbedding | None,
    ):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            positional_encoder=positional_encoder,
        )
        self.moe = SparseMoEFeedForward(d_model=d_model, d_ff=d_ff, num_experts=num_experts, top_k=top_k)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: Float[Tensor, " batch seq d_model"]) -> Float[Tensor, " batch seq d_model"]:
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x


class MoETransformerLM(nn.Module):
    """
    Small decoder-only MoE language model.

    Compared with BasicsTransformerLM, the main architectural change is:

        TransformerBlock.ffn: SwiGLU
        MoETransformerBlock.moe: router + many SwiGLU experts

    Only top_k experts are active per token, so the model can have more total FFN
    parameters than a dense model without using all of them for every token.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        num_experts: int = 4,
        top_k: int = 2,
        rope_theta: float | None = 10_000.0,
    ):
        super().__init__()
        self.config = {
            k: v for k, v in locals().items() if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        self.context_length = context_length
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        d_head = d_model // num_heads
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.positional_encoder = (
            RotaryEmbedding(context_length, d_head, rope_theta) if rope_theta is not None else None
        )
        self.layers = nn.ModuleList(
            [
                MoETransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    num_experts=num_experts,
                    top_k=top_k,
                    positional_encoder=self.positional_encoder,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        logger.info(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def load_balance_loss(self) -> torch.Tensor:
        losses = [layer.moe.last_load_balance_loss for layer in self.layers if layer.moe.last_load_balance_loss is not None]
        if not losses:
            return torch.tensor(0.0, device=self.token_embeddings.weight.device)
        return torch.stack(losses).mean()

    def expert_assignments(self) -> list[torch.Tensor | None]:
        """Return the most recent top-k expert IDs for each layer."""
        return [layer.moe.last_topk_experts for layer in self.layers]

    def forward(
        self,
        x: Int[Tensor, " batch seq"],
        return_aux_loss: bool = False,
    ) -> Float[Tensor, " batch seq vocab_size"] | tuple[Float[Tensor, " batch seq vocab_size"], torch.Tensor]:
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        if return_aux_loss:
            return logits, self.load_balance_loss()
        return logits


def describe_moe_routing(model: MoETransformerLM) -> None:
    """
    Convenience helper for tinkering in a notebook/script after a forward pass.
    """
    for layer_idx, assignments in enumerate(model.expert_assignments()):
        if assignments is None:
            print(f"layer {layer_idx}: no routing data yet")
            continue

        flat = rearrange(assignments, "batch seq top_k -> (batch seq top_k)")
        counts = torch.bincount(flat, minlength=model.num_experts)
        print(f"layer {layer_idx} expert assignment counts: {counts.tolist()}")
