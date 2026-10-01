"""Preprint implementation: selected components from the research codebase."""
from __future__ import annotations
import torch
from torch import Tensor


def deranged_source_permutation(source_factors: Tensor) -> Tensor:
    """Return a deterministic source shuffle with no unchanged factor tuple."""
    if source_factors.ndim != 2 or source_factors.shape[0] < 2:
        raise ValueError("source_factors must have shape [B, F] with B >= 2")
    indices = torch.arange(source_factors.shape[0], device=source_factors.device)
    preferred_shift = max(1, source_factors.shape[0] // 2)
    shifts = list(range(preferred_shift, source_factors.shape[0])) + list(
        range(1, preferred_shift)
    )
    for shift in shifts:
        candidate = torch.roll(indices, shifts=shift)
        unchanged = (source_factors[candidate] == source_factors).all(dim=1)
        if not torch.any(unchanged):
            return candidate
    raise ValueError("batch does not admit a source-factor derangement")
