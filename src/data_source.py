"""Shared contracts and small utilities for dataset-family adapters."""

from typing import Dict, Protocol

from torch import Tensor
from torch.utils.data import Dataset


class IWMDataSource(Protocol):
    """Dataset-family adapter used by the shared training and evaluation loaders."""

    def build_transition_dataset(
        self,
        cfg: Dict[str, object],
        split: str,
        *,
        return_label: bool,
        return_original: bool,
    ) -> Dataset: ...

    def build_k_view_dataset(self, cfg: Dict[str, object], split: str) -> Dataset: ...

    def build_repeated_future_dataset(self, cfg: Dict[str, object], split: str) -> Dataset: ...

    def build_clean_labeled_dataset(self, cfg: Dict[str, object], split: str) -> Dataset: ...


def clone_sample(value):
    """Recursively clone tensors inside a transition sample."""
    if isinstance(value, Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_sample(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_sample(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_sample(item) for item in value)
    return value
