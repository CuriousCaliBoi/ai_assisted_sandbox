"""Fused CUDA AdamW via PyTorch, with cs336_basics fallback."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from cs336_basics.optimizer import AdamW as BasicsAdamW


def build_adamw(
    params: Iterable[torch.nn.Parameter],
    *,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    fused: bool = True,
) -> torch.optim.Optimizer:
    """Return AdamW; use PyTorch fused CUDA kernel when available."""
    if fused and torch.cuda.is_available():
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=True,
        )
    return BasicsAdamW(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
