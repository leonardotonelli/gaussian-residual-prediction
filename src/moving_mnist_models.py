"""Concept2 R0/R1 video models, provisional reference implementation v1.

The missing upstream video code makes strides, temporal loss direction, and
target BatchNorm conventions local choices. See P01_MODEL_PROTOCOL.md before
using these classes for scientific runs. No action or simulator label inputs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


MODEL_VERSION = "concept2-reference-model-smoke-v1"


@dataclass(frozen=True)
class ModelConfig:
    role: str = "R0"
    spatial_strides: tuple[int, int, int] = (2, 2, 2)
    ema_decay: float = 0.996
    beta: float = 0.001
    scale_floor: float = 1e-6
    alignment: str = "forward-negative-cosine"
    target_batchnorm: str = "independent-batch-stats"
    posterior_gradients: str = "online-source-and-surrogate"

    def __post_init__(self):
        object.__setattr__(self, "spatial_strides", tuple(self.spatial_strides))
        if self.role not in ("R0", "R1"):
            raise ValueError("This slice implements only R0 and R1")
        if len(self.spatial_strides) != 3 or any(s not in (1, 2) for s in self.spatial_strides):
            raise ValueError("Three spatial strides, each 1 or 2, are required")
        if not 0 <= self.ema_decay <= 1:
            raise ValueError("EMA decay must be in [0,1]")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("Beta must be finite and nonnegative")
        if not math.isfinite(self.scale_floor) or self.scale_floor <= 0:
            raise ValueError("Gaussian scale floor must be finite and positive")
        if (self.alignment, self.target_batchnorm, self.posterior_gradients) != (
            "forward-negative-cosine", "independent-batch-stats", "online-source-and-surrogate"
        ):
            raise ValueError("Unsupported objective, target BN, or posterior gradient policy")


def _mlp(sizes: tuple[int, ...], *, output_bias: bool = False,
         output_batchnorm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (input_size, output_size) in enumerate(zip(sizes[:-1], sizes[1:])):
        final = index == len(sizes) - 2
        layers.append(nn.Linear(input_size, output_size, bias=output_bias if final else False))
        if not final or output_batchnorm:
            layers.append(nn.BatchNorm1d(output_size))
        if not final:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class VideoEncoder(nn.Module):
    """(B,1,3,64,64) -> backbone (B,768), projector (B,128)."""

    def __init__(self, spatial_strides: tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        channels = (1, 32, 64, 128, 128, 256)
        kernels = ((1, 3, 3), (1, 3, 3), (3, 1, 1), (3, 1, 1), (1, 3, 3))
        strides = (spatial_strides[0], spatial_strides[1], 1, 1, spatial_strides[2])
        layers: list[nn.Module] = []
        for i, (kernel, stride) in enumerate(zip(kernels, strides)):
            layers.extend([
                nn.Conv3d(channels[i], channels[i + 1], kernel,
                          stride=(1, stride, stride), padding=tuple(k // 2 for k in kernel), bias=False),
                nn.BatchNorm3d(channels[i + 1]), nn.ReLU(),
            ])
        self.backbone = nn.Sequential(*layers)
        self.projector = _mlp((768, 1024, 1024, 128), output_batchnorm=True)

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        if clip.ndim != 5 or tuple(clip.shape[1:]) != (1, 3, 64, 64) or clip.shape[0] < 1:
            raise ValueError("Expected nonempty video batch (B,1,3,64,64)")
        if not clip.is_floating_point():
            raise ValueError("Video pixels must be floating point in [0,1]")
        if self.training and clip.shape[0] < 2:
            raise ValueError("Training BatchNorm requires at least two clips")
        # Keep all three temporal outputs: (B,256,3,H,W) -> (B,256,3) -> (B,768).
        representation = self.backbone(clip).mean(dim=(-2, -1)).flatten(1)
        return {"backbone": representation, "projector": self.projector(representation)}


class GaussianHead(nn.Module):
    """Factorized 2-D Gaussian; softplus standard deviations, not variances."""

    def __init__(self, input_dim: int, scale_floor: float):
        super().__init__()
        self.net = _mlp((input_dim, 1024, 4), output_bias=True)
        self.scale_floor = scale_floor
        # Match the released sampler's MLP initializer, including the last layer.
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, preactivation = self.net(features).chunk(2, dim=-1)
        return mean, F.softplus(preactivation) + self.scale_floor


def diagonal_gaussian_kl(q_mean: torch.Tensor, q_std: torch.Tensor,
                         p_mean: torch.Tensor, p_std: torch.Tensor) -> torch.Tensor:
    """KL(q||p), summed over residual dimensions and averaged over examples."""
    terms = (p_std.log() - q_std.log()
             + 0.5 * ((q_std / p_std).square() + ((q_mean - p_mean) / p_std).square() - 1))
    return terms.sum(dim=-1).mean()


def _sample(mean: torch.Tensor, std: torch.Tensor,
            generator: torch.Generator | None) -> torch.Tensor:
    epsilon = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
    return mean + std * epsilon


class MovingMNISTReference(nn.Module):
    """Matched deterministic R0 and variational R1, with one frozen EMA target.

    During training the target uses its own batch/running statistics (no parameter
    gradients). update_target() changes parameters only, after an optimizer step.
    eval() switches both encoders to their own frozen running statistics.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.online_encoder = VideoEncoder(config.spatial_strides)
        self.target_encoder = copy.deepcopy(self.online_encoder).requires_grad_(False)
        self.predictor = _mlp((130 if config.role == "R1" else 128, 1024, 1024, 128))
        if config.role == "R1":
            self.prior = GaussianHead(128, config.scale_floor)
            self.posterior = GaussianHead(256, config.scale_floor)
        else:
            self.prior = None
            self.posterior = None
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))

    def forward(self, source: torch.Tensor, target: torch.Tensor,
                surrogate: torch.Tensor | None = None, *,
                generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """Training/diagnostic loss. Target-informed posterior is never a forecast.

        R1: -mean cosine(g(z,r), stopgrad(z_target)) + beta KL(q||p),
        r~q(r|online(source),online(surrogate)). Both posterior inputs retain
        gradients; only the EMA target path is stopped. No temporal reverse term.
        """
        if source.shape != target.shape or (surrogate is not None and surrogate.shape != source.shape):
            raise ValueError("Source, target and optional surrogate shapes must match")
        if self.config.role == "R1" and surrogate is None:
            raise ValueError("R1 training requires the later surrogate clip")
        if self.config.role == "R0" and surrogate is not None:
            raise ValueError("R0 does not use a surrogate clip")
        source_features = self.online_encoder(source)
        z = source_features["projector"]
        with torch.no_grad():
            z_target = self.target_encoder(target)["projector"]
        diagnostics: dict[str, torch.Tensor] = {}
        kl = z.new_zeros(())
        if self.config.role == "R1":
            u = self.online_encoder(surrogate)["projector"]
            p_mean, p_std = self.prior(z)
            q_mean, q_std = self.posterior(torch.cat((z, u), dim=-1))
            residual = _sample(q_mean, q_std, generator)
            prediction = self.predictor(torch.cat((z, residual), dim=-1))
            kl = diagonal_gaussian_kl(q_mean, q_std, p_mean, p_std)
            diagnostics = {"prior_mean": p_mean, "prior_std": p_std,
                           "posterior_mean": q_mean, "posterior_std": q_std,
                           "residual": residual}
        else:
            prediction = self.predictor(z)
        alignment = -F.cosine_similarity(prediction, z_target, dim=-1, eps=1e-8).mean()
        return {"loss": alignment + self.config.beta * kl, "alignment": alignment,
                "kl": kl, "prediction": prediction, "target_embedding": z_target,
                "source_backbone": source_features["backbone"], "source_embedding": z,
                **diagnostics}

    @torch.no_grad()
    def update_target(self) -> None:
        """Exactly one constant-decay EMA update after each optimizer update.

        Do not reuse the image project's update_ema: that copies online BN
        buffers, which would change the declared independent-target BN policy.
        """
        for name, target in self.target_encoder.named_parameters():
            online = self.online_encoder.get_parameter(name)
            target.mul_(self.config.ema_decay).add_(online, alpha=1 - self.config.ema_decay)
        self.ema_updates.add_(1)

    def _require_eval(self) -> None:
        if any(module.training for module in self.modules()):
            raise RuntimeError("Call model.eval() before frozen encoding or forecasting")

    @torch.inference_mode()
    def encode(self, clip: torch.Tensor, *, branch: str) -> dict[str, torch.Tensor]:
        """Explicit online probe vs actual target-space readout/persistence view."""
        self._require_eval()
        if branch not in ("online", "target"):
            raise ValueError("Encoder branch must be online or target")
        encoder = self.online_encoder if branch == "online" else self.target_encoder
        return encoder(clip)

    @torch.inference_mode()
    def forecast(self, source: torch.Tensor, *, num_samples: int = 1,
                 fixed_residual: bool = False,
                 generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """Source-only predictions (B,M,128), in the forecast target space.

        No future, posterior, label, or supplied residual argument is accepted.
        R0 returns a singleton and requires num_samples=1. R1 uses independent
        conditional prior draws; fixed_residual uses the prior mean. Frozen BN
        keeps samples independent of batch composition and sample count.
        """
        self._require_eval()
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if self.config.role == "R0" and (num_samples != 1 or fixed_residual):
            raise ValueError("R0 is a singleton point predictor with no residual")
        z = self.online_encoder(source)["projector"]
        if self.config.role == "R0":
            return {"prediction": self.predictor(z)[:, None]}
        mean, std = self.prior(z)
        expanded_mean = mean[:, None].expand(-1, num_samples, -1)
        residual = (expanded_mean if fixed_residual else
                    _sample(expanded_mean, std[:, None].expand_as(expanded_mean), generator))
        inputs = torch.cat((z[:, None].expand(-1, num_samples, -1), residual), dim=-1)
        prediction = self.predictor(inputs.flatten(0, 1)).reshape(len(z), num_samples, 128)
        return {"prediction": prediction, "residual": residual,
                "prior_mean": mean, "prior_std": std}

    def parameter_counts(self) -> dict[str, int]:
        return {"trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "target": sum(p.numel() for p in self.target_encoder.parameters()),
                "total": sum(p.numel() for p in self.parameters())}
