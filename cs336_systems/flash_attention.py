"""FlashAttention wrappers for CS336 systems work.

Primary backend: FlashAttention-4 CuTeDSL (Blackwell/Hopper speed-of-light kernels).
Gluon: Triton upstream ships forward-only Blackwell example; backward not wired here yet.
"""

from __future__ import annotations

import torch
from einops import rearrange
from flash_attn.cute import flash_attn_func
from flash_attn.cute.interface import _flash_attn_bwd

from cs336_basics.model import CausalMultiHeadSelfAttention


def _to_fa_dtype(t: torch.Tensor) -> torch.Tensor:
    if t.dtype in (torch.float16, torch.bfloat16):
        return t
    return t.to(torch.bfloat16)


class FlashAttention2PyTorchAutograd(torch.autograd.Function):
    """Pure-PyTorch FlashAttention-2 forward (tiled online softmax).

    Input layout: (batch, seq, head_dim) — single-head API used by tests/adapters.py.
    Tile sizes are at least 16×16 as required by the assignment.
    """

    _BR = 16
    _BC = 16

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        del is_causal  # not required for this task

        batch, n_queries, head_dim = q.shape
        n_keys = k.shape[-2]
        scale = head_dim**-0.5

        o = torch.zeros_like(q)
        l = torch.empty(batch, n_queries, device=q.device, dtype=q.dtype)

        br, bc = FlashAttention2PyTorchAutograd._BR, FlashAttention2PyTorchAutograd._BC
        for q_start in range(0, n_queries, br):
            q_block = q[:, q_start : q_start + br, :]
            o_block = torch.zeros(batch, br, head_dim, device=q.device, dtype=q.dtype)
            m_block = torch.full((batch, br), float("-inf"), device=q.device, dtype=q.dtype)
            l_block = torch.zeros(batch, br, device=q.device, dtype=q.dtype)

            for k_start in range(0, n_keys, bc):
                k_block = k[:, k_start : k_start + bc, :]
                v_block = v[:, k_start : k_start + bc, :]

                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale
                m_new = torch.maximum(m_block, scores.amax(dim=-1))
                p = torch.exp(scores - m_new.unsqueeze(-1))
                l_new = torch.exp(m_block - m_new) * l_block + p.sum(dim=-1)
                o_block = torch.exp(m_block.unsqueeze(-1) - m_new.unsqueeze(-1)) * o_block + torch.matmul(
                    p, v_block
                )
                m_block = m_new
                l_block = l_new

            o[:, q_start : q_start + br, :] = o_block / l_block.unsqueeze(-1)
            l[:, q_start : q_start + br] = m_block + torch.log(l_block)

        ctx.save_for_backward(l, q, k, v, o)
        return o

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        raise NotImplementedError


class FlashAttentionCuteAutograd(torch.autograd.Function):
    """Assignment/test wrapper around FA4 CuTe kernels (forward + backward).

    Input layout: (batch, seq, head_dim) — single-head API used by tests/adapters.py.
    Saves log-sum-exp L with shape (batch, seq) as required by the test suite.
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool):
        in_dtype = q.dtype
        q_h = _to_fa_dtype(q).unsqueeze(2)
        k_h = _to_fa_dtype(k).unsqueeze(2)
        v_h = _to_fa_dtype(v).unsqueeze(2)
        out_h, lse = flash_attn_func(q_h, k_h, v_h, causal=is_causal, return_lse=True)
        l = lse.squeeze(1)
        ctx.save_for_backward(q_h, k_h, v_h, out_h, lse, l)
        ctx.is_causal = is_causal
        ctx.in_dtype = in_dtype
        return out_h.squeeze(2).to(in_dtype)

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q_h, k_h, v_h, out_h, lse, _l = ctx.saved_tensors
        dout_h = _to_fa_dtype(dout).unsqueeze(2)
        dq, dk, dv = _flash_attn_bwd(
            q_h,
            k_h,
            v_h,
            out_h,
            dout_h,
            lse,
            softmax_scale=None,
            causal=ctx.is_causal,
        )
        return dq.squeeze(2).to(ctx.in_dtype), dk.squeeze(2).to(ctx.in_dtype), dv.squeeze(2).to(ctx.in_dtype), None


class FlashCausalMultiHeadSelfAttention(CausalMultiHeadSelfAttention):
    """Same API as cs336_basics attention, but uses FA4 CuTe forward + backward."""

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        *batch_dims, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q, k, v = (
            rearrange(t, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
            for t in (q, k, v)
        )

        if self.positional_encoder is not None:
            if token_positions is not None:
                token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
            q = self.positional_encoder(q, token_positions)
            k = self.positional_encoder(k, token_positions)

        q = _to_fa_dtype(q).transpose(-2, -3)
        k = _to_fa_dtype(k).transpose(-2, -3)
        v = _to_fa_dtype(v).transpose(-2, -3)
        attn_output, _ = flash_attn_func(q, k, v, causal=True, return_lse=True)
        attn_output = rearrange(attn_output, "batch seq heads d_v -> batch seq (heads d_v)").contiguous()
        return self.output_proj(attn_output)


def patch_model_flash_attention(model: torch.nn.Module) -> torch.nn.Module:
    """Swap each TransformerBlock attention module for the CuTe FA variant."""
    for layer in model.layers:
        old_attn = layer.attn
        new_attn = FlashCausalMultiHeadSelfAttention(
            d_model=old_attn.d_model,
            num_heads=old_attn.num_heads,
            positional_encoder=old_attn.positional_encoder,
        )
        new_attn.load_state_dict(old_attn.state_dict())
        layer.attn = new_attn
    return model
