"""Versioned campaign RNG addresses; legacy callers opt in explicitly.

NumPy uses all 256 digest bits. Torch's public manual_seed accepts 63 bits,
but the supported CPU MT19937 generator only uses the low 32 effectively.
We therefore reject effective seed collisions and use a small number of
checkpointed Torch streams, never millions of per-update Torch reseeds.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import random
from typing import Any, Iterator

DERIVATION_VERSION = "structured-sha256-v1"
CAMPAIGN = "jepa-five-seed-20260924-v1"
DATASETS = ("moving_mnist", "mpi3d")
PURPOSES = ("main-training", "development-training", "software-smoke", "evaluation")
_INDEX_FIELDS = ("epoch", "sample_index", "step", "restart", "rank")
_NAME_FIELDS = ("module", "partition", "view")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SeedKey:
    campaign: str
    dataset: str
    purpose: str
    replication: int | None
    component: str
    module: str | None = None
    partition: str | None = None
    epoch: int | None = None
    sample_index: int | None = None
    step: int | None = None
    view: str | None = None
    restart: int | None = None
    rank: int | None = None
    derivation_version: str = DERIVATION_VERSION

    def __post_init__(self):
        if self.derivation_version != DERIVATION_VERSION:
            raise ValueError("Unsupported seed derivation version")
        if not isinstance(self.campaign, str) or not self.campaign:
            raise ValueError("A nonempty campaign namespace is required")
        if self.dataset not in DATASETS or self.purpose not in PURPOSES:
            raise ValueError("Unsupported seed dataset/purpose")
        if self.purpose == "main-training":
            if type(self.replication) is not int or self.replication not in range(1, 6):
                raise ValueError("Main replications are exactly 1..5")
        elif self.purpose == "evaluation":
            if self.replication is not None:
                raise ValueError("Evaluation banks/heads are fixed across training replications")
        elif type(self.replication) is not int or self.replication != 0:
            raise ValueError("Development and software-smoke use ID 0 in separate namespaces")
        if not isinstance(self.component, str) or not self.component:
            raise ValueError("A nonempty component name is required")
        for name in _NAME_FIELDS:
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty string or null")
        for name in _INDEX_FIELDS:
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or null")
        # Data addresses are always global. Rank affects runtime/model-noise only.
        if self.rank is not None and self.component.startswith(("data-", "evaluation-")):
            raise ValueError("Data/evaluation-bank addresses must be independent of rank")

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def canonical(self) -> str:
        return canonical_json(self.as_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()

    @property
    def torch_seed(self) -> int:
        return int.from_bytes(bytes.fromhex(self.digest)[:8], "big") & ((1 << 63) - 1)

    @property
    def numpy_entropy(self) -> tuple[int, ...]:
        raw = bytes.fromhex(self.digest)
        return tuple(int.from_bytes(raw[i:i + 4], "big") for i in range(0, 32, 4))

    def record(self) -> dict:
        return {"key": self.as_dict(), "sha256": self.digest,
                "torch_seed": self.torch_seed, "torch_cpu_effective_seed": self.torch_seed & 0xffffffff,
                "numpy_entropy_u32_be": list(self.numpy_entropy)}


class SeedRegistry:
    """Detect actual seed reuse; equal keys are deliberately reusable across roles."""
    def __init__(self):
        self._torch: dict[int, str] = {}
        self._cpu: dict[int, str] = {}

    def register(self, key: SeedKey) -> int:
        seed, digest = key.torch_seed, key.digest
        for mapping, value in ((self._torch, seed), (self._cpu, seed & 0xffffffff)):
            previous = mapping.get(value)
            if previous is not None and previous != digest:
                raise ValueError("Distinct keys collide in an effective Torch seed; revise/freeze a new derivation, do not reuse")
            mapping[value] = digest
        return seed


# Process-local guard, supplemented by the full fixed-manifest collision audit.
# This never receives per-example NumPy keys, so it cannot grow with clip count.
_TORCH_KEYS = SeedRegistry()


@dataclass(frozen=True)
class SeedContext:
    dataset: str
    purpose: str
    replication: int | None
    campaign: str = CAMPAIGN
    derivation_version: str = DERIVATION_VERSION

    def __post_init__(self):
        self.key("context-validation")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> SeedContext:
        if not isinstance(value, dict) or set(value) != {
                "dataset", "purpose", "replication", "campaign", "derivation_version"}:
            raise ValueError("An explicit complete versioned seed context is required")
        return cls(**value)

    def key(self, component: str, **indices) -> SeedKey:
        unknown = set(indices) - set(_INDEX_FIELDS) - set(_NAME_FIELDS)
        if unknown:
            raise ValueError(f"Unknown seed address fields: {sorted(unknown)}")
        return SeedKey(**self.as_dict(), component=component, **indices)

    def seed(self, component: str, **indices) -> int:
        return _TORCH_KEYS.register(self.key(component, **indices))

    def numpy_rng(self, component: str, **indices):
        import numpy as np
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            self.key(component, **indices).numpy_entropy)))

    def torch_generator(self, component: str, *, device="cpu", **indices):
        import torch
        return torch.Generator(device=device).manual_seed(self.seed(component, **indices))

    @property
    def sha256(self) -> str:
        return content_sha256(self.as_dict())


def context_from_config(config: dict) -> SeedContext | None:
    """Missing field means legacy; present malformed/null metadata is rejected."""
    return SeedContext.from_dict(config["seed_streams"]) if "seed_streams" in config else None


@contextmanager
def preserve_rng() -> Iterator[None]:
    """Restore Python/NumPy/Torch and already-initialized CUDA RNG, even on error."""
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_initialized = torch.cuda.is_initialized()
    cuda_state = torch.cuda.get_rng_state_all() if cuda_initialized else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@contextmanager
def seeded_rng(seed: int) -> Iterator[None]:
    """Isolated module initialization, including CUDA if already initialized."""
    import numpy as np
    import torch
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Expected nonnegative 63-bit seed")
    with preserve_rng():
        random.seed(seed)
        np.random.seed(seed & 0xffffffff)
        # Avoid manual_seed's lazy CUDA callbacks when CUDA is uninitialized.
        torch.random.default_generator.manual_seed(seed)
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed_all(seed)
        yield
