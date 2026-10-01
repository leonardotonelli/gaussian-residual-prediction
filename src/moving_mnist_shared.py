"""Concept2 S0: shared full-gradient video encoder and real-clip SIGReg."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .lewm_adassl import PatchSIGReg
from .moving_mnist_models import VideoEncoder, _mlp


SHARED_MODEL_VERSION = "concept2-shared-s0-v1"


@dataclass(frozen=True)
class S0Config:
    role: str = "S0"
    batch_size: int = 128
    spatial_strides: tuple[int, int, int] = (2, 2, 2)
    sigreg_weight: float = 0.1
    sigreg_projections: int = 128
    sigreg_knots: int = 17
    sigreg_interval: tuple[float, float] = (0.2, 4.0)
    sigreg_seed: int = 4000
    sigreg_placement: str = "projector-output-after-bn-before-cosine"
    alignment: str = "forward-negative-cosine"

    def __post_init__(self):
        object.__setattr__(self, "spatial_strides", tuple(self.spatial_strides))
        object.__setattr__(self, "sigreg_interval", tuple(self.sigreg_interval))
        if self.role != "S0":
            raise ValueError("This implementation is S0 only")
        if self.batch_size < 2:
            raise ValueError("S0 training batch size must be >=2")
        if len(self.spatial_strides) != 3 or any(s not in (1, 2) for s in self.spatial_strides):
            raise ValueError("Three spatial strides, each 1 or 2, are required")
        if not math.isfinite(self.sigreg_weight) or self.sigreg_weight <= 0:
            raise ValueError("S0 requires a finite positive SIGReg weight")
        if self.sigreg_projections < 1 or self.sigreg_knots < 2:
            raise ValueError("Invalid SIGReg projection or knot count")
        if (len(self.sigreg_interval) != 2 or not all(math.isfinite(x) for x in self.sigreg_interval)
                or not 0 <= self.sigreg_interval[0] < self.sigreg_interval[1]):
            raise ValueError("Invalid SIGReg interval")
        if self.alignment != "forward-negative-cosine" or self.sigreg_placement != "projector-output-after-bn-before-cosine":
            raise ValueError("Unsupported S0 objective/placement")


class MovingMNISTS0(nn.Module):
    """One encoder for both real clips; no target copy, stop-gradient or EMA.

    L = -mean cosine(predictor(z), t) + lambda/2 * (SIGReg(z)+SIGReg(t)).
    z and t are real-clip projector outputs, before cosine's unit normalization.
    Separate source then target forwards update shared BN twice per step.
    """

    def __init__(self, config: S0Config):
        super().__init__()
        self.config = config
        self.encoder = VideoEncoder(config.spatial_strides)
        self.predictor = _mlp((128, 1024, 1024, 128))
        self.regularizer = PatchSIGReg(
            global_batch_size=config.batch_size, num_projections=config.sigreg_projections,
            knots=config.sigreg_knots, interval=config.sigreg_interval, rng_seed=config.sigreg_seed)

    def get_extra_state(self):
        # PatchSIGReg owns a CPU generator; include it in ordinary model snapshots.
        return {"version": SHARED_MODEL_VERSION, "sigreg_rng": self.regularizer.rng_state()}

    def set_extra_state(self, state):
        if state["version"] != SHARED_MODEL_VERSION:
            raise ValueError("S0 RNG snapshot version mismatch")
        self.regularizer.set_rng_state(state["sigreg_rng"])

    def _real_features(self, clip):
        captured = []
        hook = self.encoder.projector[-1].register_forward_pre_hook(
            lambda module, args: captured.append(args[0]))
        try:
            features = self.encoder(clip)
        finally:
            hook.remove()
        return {**features, "pre_output_bn": captured[0]}

    def forward(self, source, target, *, sigreg_generators=None):
        if source.shape != target.shape or len(source) != self.config.batch_size:
            raise ValueError("S0 requires matched source/target clips at its configured training batch size")
        source_features = self._real_features(source)
        target_features = self._real_features(target)
        z, t = source_features["projector"], target_features["projector"]
        prediction = self.predictor(z)
        alignment = -F.cosine_similarity(prediction, t, dim=-1, eps=1e-8).mean()
        # One clip embedding = one patch. Each real view gets independent fresh
        # directions from the same checkpointed generator, in source/target order.
        source_rng, target_rng = (None, None) if sigreg_generators is None else sigreg_generators
        source_reg = self.regularizer(z[:, None], generator=source_rng).sigreg
        target_reg = self.regularizer(t[:, None], generator=target_rng).sigreg
        sigreg = 0.5 * (source_reg + target_reg)
        return {"loss": alignment + self.config.sigreg_weight * sigreg,
                "alignment": alignment, "sigreg": sigreg, "source_sigreg": source_reg,
                "target_sigreg": target_reg, "prediction": prediction,
                "source_embedding": z, "target_embedding": t,
                "source_pre_output_bn": source_features["pre_output_bn"],
                "target_pre_output_bn": target_features["pre_output_bn"]}

    def _require_eval(self):
        if any(module.training for module in self.modules()):
            raise RuntimeError("Call model.eval() before frozen encoding or forecasting")

    @torch.inference_mode()
    def encode(self, clip, *, branch="online"):
        self._require_eval()
        if branch not in ("online", "target"):
            raise ValueError("Branch must be online or target; both use the shared encoder")
        return self._real_features(clip)

    @torch.inference_mode()
    def forecast(self, source, *, num_samples=1):
        self._require_eval()
        if num_samples != 1:
            raise ValueError("S0 is a singleton point predictor")
        return {"prediction": self.predictor(self.encoder(source)["projector"])[:, None]}

    def parameter_counts(self):
        return {"trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "target": 0, "total": sum(p.numel() for p in self.parameters())}
