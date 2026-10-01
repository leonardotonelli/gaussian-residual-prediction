"""S1: shared full-gradient video model with Gaussian residual and SIGReg."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch
from torch import nn
import torch.nn.functional as F

from .lewm_adassl import PatchSIGReg
from .moving_mnist_models import VideoEncoder, GaussianHead, _mlp, _sample, diagonal_gaussian_kl
from .moving_mnist_shared import S0Config, MovingMNISTS0


S1_MODEL_VERSION = "concept2-shared-s1-v1"


@dataclass(frozen=True)
class S1Config(S0Config):
    role: str = "S1"
    beta: float = 0.001
    scale_floor: float = 1e-6
    posterior_gradients: str = "shared-source-and-surrogate"

    def __post_init__(self):
        if self.role != "S1":
            raise ValueError("This implementation is S1 only")
        common = {field.name: getattr(self, field.name) for field in fields(S0Config)}
        common["role"] = "S0"
        validated = S0Config(**common)
        object.__setattr__(self, "spatial_strides", validated.spatial_strides)
        object.__setattr__(self, "sigreg_interval", validated.sigreg_interval)
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("Beta must be finite and nonnegative")
        if not math.isfinite(self.scale_floor) or self.scale_floor <= 0:
            raise ValueError("Gaussian scale floor must be finite and positive")
        if self.posterior_gradients != "shared-source-and-surrogate":
            raise ValueError("Unsupported posterior gradient policy")


class MovingMNISTS1(MovingMNISTS0):
    """S0's shared encoder/SIGReg with R1's residual family and information flow.

    One shared encoder processes source, target, surrogate separately in order.
    Only source and target receive SIGReg. Alignment reaches all three clips;
    KL reaches source/surrogate and both heads, but never target pixels.
    """

    def __init__(self, config: S1Config):
        # Build directly, preserving R1's seeded encoder/predictor/head init order.
        nn.Module.__init__(self)
        self.config = config
        self.encoder = VideoEncoder(config.spatial_strides)
        self.predictor = _mlp((130, 1024, 1024, 128))
        self.prior = GaussianHead(128, config.scale_floor)
        self.posterior = GaussianHead(256, config.scale_floor)
        self.regularizer = PatchSIGReg(
            global_batch_size=config.batch_size, num_projections=config.sigreg_projections,
            knots=config.sigreg_knots, interval=config.sigreg_interval, rng_seed=config.sigreg_seed)

    def get_extra_state(self):
        return {"version": S1_MODEL_VERSION, "sigreg_rng": self.regularizer.rng_state()}

    def set_extra_state(self, state):
        if state["version"] != S1_MODEL_VERSION:
            raise ValueError("S1 RNG snapshot version mismatch")
        self.regularizer.set_rng_state(state["sigreg_rng"])

    def forward(self, source, target, surrogate, *, generator=None, sigreg_generators=None):
        if (source.shape != target.shape or source.shape != surrogate.shape
                or len(source) != self.config.batch_size):
            raise ValueError("S1 requires matched source/target/surrogate clips at its training batch size")
        sf, tf = self._real_features(source), self._real_features(target)
        z, t = sf["projector"], tf["projector"]
        u = self.encoder(surrogate)["projector"]
        p_mean, p_std = self.prior(z)
        q_mean, q_std = self.posterior(torch.cat((z, u), dim=-1))
        residual = _sample(q_mean, q_std, generator)
        prediction = self.predictor(torch.cat((z, residual), dim=-1))
        alignment = -F.cosine_similarity(prediction, t, dim=-1, eps=1e-8).mean()
        kl = diagonal_gaussian_kl(q_mean, q_std, p_mean, p_std)
        source_rng, target_rng = (None, None) if sigreg_generators is None else sigreg_generators
        source_reg = self.regularizer(z[:, None], generator=source_rng).sigreg
        target_reg = self.regularizer(t[:, None], generator=target_rng).sigreg
        sigreg = 0.5 * (source_reg + target_reg)
        return {"loss": alignment + self.config.beta * kl + self.config.sigreg_weight * sigreg,
                "alignment": alignment, "kl": kl, "sigreg": sigreg,
                "source_sigreg": source_reg, "target_sigreg": target_reg,
                "prediction": prediction, "source_embedding": z, "target_embedding": t,
                "surrogate_embedding": u, "source_pre_output_bn": sf["pre_output_bn"],
                "target_pre_output_bn": tf["pre_output_bn"],
                "prior_mean": p_mean, "prior_std": p_std,
                "posterior_mean": q_mean, "posterior_std": q_std, "residual": residual}

    @torch.inference_mode()
    def forecast(self, source, *, num_samples=1, fixed_residual=False, generator=None):
        self._require_eval()
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        z = self.encoder(source)["projector"]
        mean, std = self.prior(z)
        expanded = mean[:, None].expand(-1, num_samples, -1)
        residual = expanded if fixed_residual else _sample(expanded, std[:, None].expand_as(expanded), generator)
        inputs = torch.cat((z[:, None].expand(-1, num_samples, -1), residual), dim=-1)
        prediction = self.predictor(inputs.flatten(0, 1)).reshape(len(z), num_samples, 128)
        return {"prediction": prediction, "residual": residual, "prior_mean": mean, "prior_std": std}
