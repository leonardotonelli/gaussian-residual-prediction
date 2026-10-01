"""Preprint implementation: selected components from the research codebase."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
import torch.distributed as dist
from torch import Tensor


class _GatherGlobalBatch(torch.autograd.Function):
    """Gather samples while producing the correct gradient under DDP averaging.

    Every rank evaluates the same loss on the gathered global batch, but its model
    graph only produced that rank's local samples. DDP subsequently averages model
    gradients, so the local slice gradient is scaled by the world size here. This
    gives exactly the gradient of one global-batch loss and avoids backend-specific
    autograd behavior in the functional distributed gather.
    """

    @staticmethod
    def forward(ctx: Any, local_tokens: Tensor) -> tuple[Tensor, ...]:
        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()
        gathered = [torch.empty_like(local_tokens) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, local_tokens.contiguous())
        return tuple(gathered)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tensor:
        return grad_outputs[ctx.rank].contiguous() * ctx.world_size


def _global_batch(local_tokens: Tensor) -> Tensor:
    """Differentiably gather the local image batch in rank order for DDP."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return local_tokens
    return torch.cat(_GatherGlobalBatch.apply(local_tokens), dim=0)


def gather_global_batch(local_tokens: Tensor) -> Tensor:
    """Public differentiable global-batch gather for new regularizers."""
    return _global_batch(local_tokens)


@dataclass(frozen=True)
class AntiCollapseLosses:
    """Differentiable loss components and detached health diagnostics."""

    auxiliary: Tensor
    variance: Tensor
    covariance: Tensor
    sigreg: Tensor
    mean_std: Tensor
    min_std: Tensor
