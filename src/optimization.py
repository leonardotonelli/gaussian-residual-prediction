"""Optimizer construction and per-step schedules for patch-token IWM training."""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Sequence

import torch
from torch import nn


_NORMALIZATION_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
)


def build_weight_decay_parameter_groups(
    modules: Iterable[nn.Module],
    weight_decay: float,
) -> list[Dict[str, object]]:
    """Split trainable weights from biases and normalization parameters."""
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    decay_parameters = []
    no_decay_parameters = []
    seen_parameter_ids = set()
    for root_module in modules:
        for module in root_module.modules():
            is_normalization = isinstance(module, _NORMALIZATION_TYPES)
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if not parameter.requires_grad:
                    continue
                parameter_id = id(parameter)
                if parameter_id in seen_parameter_ids:
                    raise ValueError("A parameter was included in multiple optimizer modules")
                seen_parameter_ids.add(parameter_id)
                if parameter_name == "bias" or is_normalization:
                    no_decay_parameters.append(parameter)
                else:
                    decay_parameters.append(parameter)

    if not decay_parameters or not no_decay_parameters:
        raise ValueError("Expected both decay and no-decay trainable parameters")

    return [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
            "use_weight_decay": True,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
            "use_weight_decay": False,
        },
    ]


def build_adamw_optimizer(
    modules: Iterable[nn.Module],
    learning_rate: float,
    weight_decay: float,
    betas: Sequence[float],
) -> torch.optim.AdamW:
    """Build AdamW with no decay on biases or normalization parameters."""
    if len(betas) != 2:
        raise ValueError("betas must contain exactly two values")
    return torch.optim.AdamW(
        build_weight_decay_parameter_groups(modules, weight_decay=weight_decay),
        lr=learning_rate,
        betas=(float(betas[0]), float(betas[1])),
    )


@dataclass(frozen=True)
class OptimizationValues:
    """The optimizer and EMA settings used for one training step."""

    learning_rate: float
    weight_decay: float
    ema_decay: float


@dataclass(frozen=True)
class IWMOptimizationSchedule:
    """Warm up LR, then schedule LR, decay, and EMA by global step."""

    base_learning_rate: float
    weight_decay_start: float
    weight_decay_end: float
    ema_decay_start: float
    ema_decay_end: float
    warmup_steps: int
    total_steps: int
    learning_rate_schedule_steps: int

    def __post_init__(self) -> None:
        if self.base_learning_rate <= 0:
            raise ValueError("base_learning_rate must be positive")
        if self.weight_decay_start < 0 or self.weight_decay_end < 0:
            raise ValueError("weight decay values must be non-negative")
        if not 0.0 <= self.ema_decay_start <= self.ema_decay_end <= 1.0:
            raise ValueError("EMA decays must satisfy 0 <= start <= end <= 1")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= self.warmup_steps < self.learning_rate_schedule_steps:
            raise ValueError("warmup_steps must be smaller than learning_rate_schedule_steps")
        if self.learning_rate_schedule_steps < self.total_steps:
            raise ValueError("learning_rate_schedule_steps must cover all training steps")

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, object],
        steps_per_epoch: int,
    ) -> "IWMOptimizationSchedule":
        """Build a schedule from YAML values and the actual dataloader length."""
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")

        optim_cfg = cfg["optim"]
        train_cfg = cfg["train"]
        total_steps = int(train_cfg["epochs"]) * steps_per_epoch
        schedule_scale = float(optim_cfg.get("learning_rate_schedule_scale", 1.0))
        if schedule_scale < 1.0:
            raise ValueError("learning_rate_schedule_scale must be at least 1")

        return cls(
            base_learning_rate=float(optim_cfg["lr"]),
            weight_decay_start=float(optim_cfg["weight_decay_start"]),
            weight_decay_end=float(optim_cfg["weight_decay_end"]),
            ema_decay_start=float(optim_cfg["ema_decay_start"]),
            ema_decay_end=float(optim_cfg["ema_decay_end"]),
            warmup_steps=int(train_cfg["warmup_epochs"]) * steps_per_epoch,
            total_steps=total_steps,
            learning_rate_schedule_steps=math.ceil(total_steps * schedule_scale),
        )

    def values_at(self, global_step: int) -> OptimizationValues:
        """Return LR, weight decay, and EMA momentum for one zero-based step."""
        if not 0 <= global_step < self.total_steps:
            raise ValueError("global_step is outside the configured training range")

        if global_step < self.warmup_steps:
            learning_rate = self.base_learning_rate * (global_step + 1) / self.warmup_steps
        else:
            progress = (global_step - self.warmup_steps) / (
                self.learning_rate_schedule_steps - self.warmup_steps
            )
            learning_rate = self.base_learning_rate * _cosine_interpolate(1.0, 0.0, progress)

        training_progress = global_step / max(self.total_steps - 1, 1)
        return OptimizationValues(
            learning_rate=learning_rate,
            weight_decay=_cosine_interpolate(
                self.weight_decay_start,
                self.weight_decay_end,
                training_progress,
            ),
            ema_decay=_cosine_interpolate(
                self.ema_decay_start,
                self.ema_decay_end,
                training_progress,
            ),
        )


def apply_optimization_values(
    optimizer: torch.optim.Optimizer,
    values: OptimizationValues,
) -> None:
    """Apply one step's scheduled values to the optimizer parameter groups."""
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = values.learning_rate
        parameter_group["weight_decay"] = values.weight_decay if parameter_group.get(
            "use_weight_decay", True
        ) else 0.0


def _cosine_interpolate(start: float, end: float, progress: float) -> float:
    """Interpolate smoothly between two values for progress in the range [0, 1]."""
    clipped_progress = min(max(progress, 0.0), 1.0)
    cosine_fraction = 0.5 * (1.0 - math.cos(math.pi * clipped_progress))
    return start + (end - start) * cosine_fraction
