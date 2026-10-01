"""Preprint implementation: selected components from the research codebase."""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader


def clip_loader(dataset, *, batch_size: int, start: int = 0, end: int | None = None,
                workers: int = 0, seed: int = 0, pin_memory: bool = False):
    """Separate loader RNG avoids advancing model RNG on iterator creation."""
    return DataLoader(dataset, batch_size=batch_size, sampler=range(start, len(dataset) if end is None else end),
                      num_workers=workers, pin_memory=pin_memory,
                      generator=torch.Generator().manual_seed(seed),
                      persistent_workers=workers > 0)
