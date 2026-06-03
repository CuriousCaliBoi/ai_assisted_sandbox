"""Chunked LM-head + cross-entropy without materializing full (batch, seq, vocab) logits."""

from __future__ import annotations

import torch
import torch.nn as nn

from cs336_basics.nn_utils import cross_entropy


def fused_lm_head_cross_entropy(
    hidden: torch.Tensor,
    lm_head: nn.Module,
    targets: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Compute mean token cross-entropy matching cs336_basics cross_entropy on full logits.

    Processes tokens in chunks so peak logits memory is (chunk_size, vocab_size) instead of
    (batch * seq, vocab_size).
    """
    if hidden.shape[:-1] != targets.shape:
        raise ValueError(
            f"hidden batch dims {hidden.shape[:-1]} must match targets shape {targets.shape}"
        )

    d_model = hidden.shape[-1]
    flat_hidden = hidden.reshape(-1, d_model)
    flat_targets = targets.reshape(-1)
    num_tokens = flat_hidden.shape[0]
    if num_tokens == 0:
        return hidden.new_zeros(())

    total_nll = hidden.new_zeros(())
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        chunk_logits = lm_head(flat_hidden[start:end])
        total_nll = total_nll + cross_entropy(chunk_logits, flat_targets[start:end]) * (end - start)

    return total_nll / num_tokens
