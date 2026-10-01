"""Small, explicit helpers for single-node distributed IWM training."""

from dataclasses import dataclass
from datetime import timedelta
import os

import torch
import torch.distributed as dist

from .utils import resolve_device


# Rank zero runs complete validation while the other DDP ranks wait at the
# epoch barrier.  D's retrieval validation and N's three-future validation can
# legitimately exceed NCCL's default 10-minute collective timeout on V100s.
# This is a communication watchdog allowance, not a training-budget change.
_MPI3D_VALIDATION_PROCESS_GROUP_TIMEOUT = timedelta(minutes=30)


@dataclass(frozen=True)
class DistributedContext:
    """The process identity and device selected by a torchrun launch."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def initialize_distributed(device_name: str) -> DistributedContext:
    """Initialize NCCL only when torchrun requested more than one process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return DistributedContext(rank=0, world_size=1, local_rank=0, device=resolve_device(device_name))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed MPI3D training requires CUDA for NCCL")
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError as error:
        raise RuntimeError("Distributed MPI3D training must be launched with torchrun") from error
    if not 0 <= rank < world_size:
        raise RuntimeError("RANK must be in [0, WORLD_SIZE)")
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError("LOCAL_RANK does not identify an available CUDA device")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=_MPI3D_VALIDATION_PROCESS_GROUP_TIMEOUT,
    )
    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=torch.device(f"cuda:{local_rank}"),
    )


def barrier(context: DistributedContext) -> None:
    """Synchronize ranks only for a multi-process launch."""
    if context.is_distributed:
        dist.barrier()


def destroy_distributed(context: DistributedContext) -> None:
    """Release the process group after every successful or failed launch."""
    if context.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def mean_across_ranks(value: float, context: DistributedContext) -> float:
    """Return the arithmetic mean of one scalar reported by every DDP rank."""
    if not context.is_distributed:
        return float(value)
    tensor = torch.tensor(float(value), device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float((tensor / context.world_size).item())
