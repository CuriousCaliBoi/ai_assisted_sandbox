"""Minimal FSDP for cs336_basics Linear/Embedding modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from cs336_basics.model import Embedding, Linear


@dataclass
class _ShardMeta:
    dim: int
    full_shape: torch.Size
    rank_offset: int
    rank_numel: int


class FSDP(nn.Module):
    """Fully-sharded data parallel wrapper for cs336_basics modules."""

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("FSDP requires an initialized process group.")

        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._shard_meta: dict[nn.Parameter, _ShardMeta] = {}
        self._hook_handles: list = []

        self._shard_module_parameters()
        self._register_fsdp_hooks()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def _shard_module_parameters(self) -> None:
        for mod in self.module.modules():
            if not isinstance(mod, (Linear, Embedding)):
                continue
            weight = mod.weight
            full = weight.data.to(torch.float32)
            shard_dim = 0
            full_rows = full.shape[shard_dim]
            if full_rows % self.world_size != 0:
                raise ValueError(
                    f"Cannot shard {type(mod).__name__} weight with shape {tuple(full.shape)} "
                    f"evenly across {self.world_size} ranks."
                )
            shard_rows = full_rows // self.world_size
            start = self.rank * shard_rows
            stop = start + shard_rows
            sl = [slice(None)] * full.ndim
            sl[shard_dim] = slice(start, stop)
            local_shard = full[tuple(sl)].contiguous()

            weight.data = local_shard
            self._shard_meta[weight] = _ShardMeta(
                dim=shard_dim,
                full_shape=full.shape,
                rank_offset=start,
                rank_numel=local_shard.numel(),
            )

    def _register_fsdp_hooks(self) -> None:
        for mod in self.module.modules():
            if isinstance(mod, Linear):
                self._hook_handles.append(mod.register_forward_pre_hook(self._make_linear_forward_pre(mod)))
                self._hook_handles.append(mod.register_forward_hook(self._make_forward_post(mod)))
                self._hook_handles.append(mod.register_full_backward_pre_hook(self._make_linear_backward_pre(mod)))
                mod.weight.register_post_accumulate_grad_hook(self._make_grad_hook(mod.weight))
            elif isinstance(mod, Embedding):
                self._hook_handles.append(mod.register_forward_pre_hook(self._make_embedding_forward_pre(mod)))
                self._hook_handles.append(mod.register_forward_hook(self._make_forward_post(mod)))
                mod.weight.register_post_accumulate_grad_hook(self._make_grad_hook(mod.weight))

        for mod in self.module.modules():
            if isinstance(mod, (Linear, Embedding)):
                continue
            for param in mod.parameters(recurse=False):
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self._make_replicated_grad_hook(param))

    def _gather_weight(self, param: nn.Parameter, dtype: torch.dtype) -> torch.Tensor:
        meta = self._shard_meta[param]
        local = param.data
        local_cast = local.to(dtype)
        gathered = [torch.empty_like(local_cast) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_cast)
        return torch.cat(gathered, dim=meta.dim)

    def _make_linear_forward_pre(self, mod: Linear):
        def hook(_mod: Linear, _inputs: tuple[torch.Tensor, ...]) -> None:
            full_weight = self._gather_weight(mod.weight, self.compute_dtype or torch.float32)
            mod._fsdp_saved_shard = mod.weight.data
            mod.weight.data = full_weight

        return hook

    def _make_embedding_forward_pre(self, mod: Embedding):
        def hook(_mod: Embedding, _inputs: tuple[torch.Tensor, ...]) -> None:
            full_weight = self._gather_weight(mod.weight, self.compute_dtype or torch.float32)
            mod._fsdp_saved_shard = mod.weight.data
            mod.weight.data = full_weight

        return hook

    def _make_forward_post(self, mod: nn.Module):
        def hook(_mod: nn.Module, _inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            if hasattr(mod, "_fsdp_saved_shard"):
                mod.weight.data = mod._fsdp_saved_shard
                del mod._fsdp_saved_shard
            mod.weight.grad = None

        return hook

    def _make_linear_backward_pre(self, mod: Linear):
        def hook(_mod: Linear, _grad_output: torch.Tensor) -> None:
            full_weight = self._gather_weight(mod.weight, self.compute_dtype or torch.float32)
            mod._fsdp_saved_shard_bwd = mod.weight.data
            mod.weight.data = full_weight
            mod.weight.grad = None

        return hook

    def _make_grad_hook(self, param: nn.Parameter):
        meta = self._shard_meta[param]

        def hook(full_param: nn.Parameter) -> None:
            if full_param.grad is None:
                return
            mod = None
            for m in self.module.modules():
                if isinstance(m, (Linear, Embedding)) and m.weight is full_param:
                    mod = m
                    break
            if mod is not None and hasattr(mod, "_fsdp_saved_shard_bwd"):
                mod.weight.data = mod._fsdp_saved_shard_bwd
                del mod._fsdp_saved_shard_bwd

            full_grad = full_param.grad.to(torch.float32)
            chunks = list(full_grad.chunk(self.world_size, dim=meta.dim))
            local_grad = torch.zeros_like(param.data, dtype=torch.float32)
            dist.reduce_scatter(local_grad, chunks, op=dist.ReduceOp.SUM)
            local_grad.div_(self.world_size)
            param.grad = local_grad

        return hook

    def _make_replicated_grad_hook(self, param: nn.Parameter):
        def hook(p: nn.Parameter) -> None:
            if p.grad is None:
                return
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(self.world_size)

        return hook

    def finish_gradient_synchronization(self) -> None:
        """No-op: sharded/replicated sync happens in grad hooks."""
        return


def fsdp_on_after_backward(fsdp_model: FSDP, optimizer: torch.optim.Optimizer) -> None:
    del optimizer
    fsdp_model.finish_gradient_synchronization()


def fsdp_gather_full_params(fsdp_model: FSDP) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, param in fsdp_model.module.named_parameters():
        if param in fsdp_model._shard_meta:
            meta = fsdp_model._shard_meta[param]
            local = param.data
            gathered = [torch.empty_like(local) for _ in range(fsdp_model.world_size)]
            dist.all_gather(gathered, local)
            state[name] = torch.cat(gathered, dim=meta.dim)
        else:
            state[name] = param.data.detach().clone()
    return state
