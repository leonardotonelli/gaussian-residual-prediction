"""Preprint implementation: selected components from the research codebase."""
from __future__ import annotations
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any, Literal
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .seed_streams import SeedContext, seeded_rng
from .mpi3d_data import MPI3D_COMMON_SOURCE_MAX, MPI3D_COMMON_SOURCE_MIN


MPI3D_PROBE_LABEL_KEYS = (
    "position",
    "camera_height",
    "shape_id",
    "color_id",
    "size_id",
)


MPI3D_PROBE_CLASS_COUNTS = {
    "camera_height": 3,
    "shape_id": 6,
    "color_id": 6,
    "size_id": 2,
    "shape_parity_id": 2,
}


def _validate_feature_matrix(features: Tensor, *, name: str = "features") -> None:
    if not isinstance(features, Tensor) or features.ndim != 2:
        raise ValueError(f"{name} must have shape [num_samples, feature_dim]")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty")
    if not features.is_floating_point():
        raise ValueError(f"{name} must have a floating dtype")


def fit_feature_standardizer(features: Tensor) -> dict[str, Tensor]:
    """Fit the locked training-only per-coordinate standardization transform."""
    _validate_feature_matrix(features)
    mean = features.mean(dim=0)
    standard_deviation = features.std(dim=0, unbiased=False)
    divisor = torch.where(standard_deviation < 1e-6, torch.ones_like(standard_deviation), standard_deviation)
    return {"mean": mean, "divisor": divisor, "negligible_count": (standard_deviation < 1e-6).sum()}


def apply_feature_standardizer(features: Tensor, standardizer: Mapping[str, Tensor]) -> Tensor:
    """Apply a previously fitted train-only feature transform unchanged."""
    _validate_feature_matrix(features)
    if set(standardizer) != {"mean", "divisor", "negligible_count"}:
        raise ValueError("MPI3D feature standardizer has unexpected fields")
    mean = standardizer["mean"]
    divisor = standardizer["divisor"]
    if mean.ndim != 1 or divisor.ndim != 1 or mean.shape != divisor.shape:
        raise ValueError("MPI3D feature standardizer mean/divisor must be matching vectors")
    if mean.shape[0] != features.shape[1]:
        raise ValueError("MPI3D feature standardizer dimension does not match features")
    return (features - mean) / divisor


def normalize_mpi3d_positions(positions: Tensor) -> Tensor:
    """Map legal MPI3D coordinate IDs linearly from ``[4, 35]`` to ``[-1, 1]``."""
    if positions.dtype != torch.long or positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must use torch.long and have shape [num_samples, 2]")
    if positions.numel() == 0:
        raise ValueError("positions cannot be empty")
    if positions.min() < MPI3D_COMMON_SOURCE_MIN or positions.max() > MPI3D_COMMON_SOURCE_MAX:
        raise ValueError("positions must lie in the locked legal MPI3D source region [4, 35]")
    scale = MPI3D_COMMON_SOURCE_MAX - MPI3D_COMMON_SOURCE_MIN
    return 2.0 * (positions.to(torch.float32) - MPI3D_COMMON_SOURCE_MIN) / scale - 1.0


def _regression_metrics(predictions: Tensor, targets: Tensor) -> dict[str, float]:
    if predictions.shape != targets.shape or tuple(predictions.shape[1:]) != (2,):
        raise ValueError("position predictions and targets must both have shape [num_samples, 2]")
    residuals = predictions - targets
    squared_error = residuals.square().sum(dim=0)
    centered_total = (targets - targets.mean(dim=0)).square().sum(dim=0)
    r2 = torch.where(
        centered_total > torch.finfo(targets.dtype).eps,
        1.0 - squared_error / centered_total,
        torch.zeros_like(centered_total),
    )
    grid_index_scale = (MPI3D_COMMON_SOURCE_MAX - MPI3D_COMMON_SOURCE_MIN) / 2.0
    mae = residuals.abs().mean(dim=0) * grid_index_scale
    return {
        "horizontal_r2": float(r2[0].item()),
        "vertical_r2": float(r2[1].item()),
        "mean_position_r2": float(r2.mean().item()),
        "horizontal_mae_grid_indices": float(mae[0].item()),
        "vertical_mae_grid_indices": float(mae[1].item()),
        "mean_mae_grid_indices": float(mae.mean().item()),
    }


def _balanced_accuracy(logits: Tensor, labels: Tensor, *, num_classes: int) -> float:
    if logits.ndim != 2 or logits.shape != (labels.shape[0], num_classes):
        raise ValueError("classification logits have an unexpected shape")
    if labels.dtype != torch.long or labels.ndim != 1:
        raise ValueError("classification labels must use torch.long and have shape [num_samples]")
    recalls = []
    predictions = logits.argmax(dim=1)
    for class_id in range(num_classes):
        members = labels == class_id
        if not bool(members.any()):
            raise ValueError("balanced accuracy requires every class to be present")
        recalls.append((predictions[members] == class_id).to(torch.float32).mean())
    return float(torch.stack(recalls).mean().item())


ProbeKind = Literal[
    "position", "camera_height", "shape_id", "color_id", "size_id", "shape_parity_id"
]


def _probe_output_dimension(kind: ProbeKind) -> int:
    if kind == "position":
        return 2
    try:
        return MPI3D_PROBE_CLASS_COUNTS[kind]
    except KeyError as error:
        raise ValueError(f"Unsupported MPI3D probe kind: {kind}") from error


@torch.no_grad()
def evaluate_mpi3d_affine_probe(
    features: Tensor,
    labels: Tensor,
    head: nn.Linear,
    kind: ProbeKind,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, float]:
    """Evaluate one fitted affine head without changing it or the encoder."""
    _validate_feature_matrix(features)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    head.eval()
    outputs = []
    for start in range(0, features.shape[0], batch_size):
        outputs.append(head(features[start : start + batch_size].to(device)).cpu())
    output = torch.cat(outputs, dim=0)
    if kind == "position":
        return _regression_metrics(output, labels.cpu())
    return {"balanced_accuracy": _balanced_accuracy(output, labels.cpu(), num_classes=_probe_output_dimension(kind))}


def _validate_probe_context(seed_context: SeedContext) -> None:
    if seed_context.dataset != "mpi3d" or seed_context.purpose != "evaluation":
        raise ValueError("MPI3D probes require a fixed evaluation SeedContext")


def _epoch_permutation(
    num_samples: int, *, probe_seed: int, epoch: int,
    seed_context: SeedContext | None = None, module: str = "affine", restart: int = 0,
) -> Tensor:
    if epoch < 1:
        raise ValueError("probe epoch must start at one")
    if seed_context is not None:
        _validate_probe_context(seed_context)
        rng = seed_context.numpy_rng(
            "probe-order", module=module, partition="fit", epoch=epoch - 1, restart=restart
        )
        return torch.from_numpy(rng.permutation(num_samples))
    else:
        generator = torch.Generator().manual_seed(probe_seed + epoch - 1)
    return torch.randperm(num_samples, generator=generator)


def _new_affine_head(
    input_dim: int, output_dim: int, *, probe_seed: int, device: torch.device,
    seed_context: SeedContext | None = None, module: str = "affine", restart: int = 0,
) -> nn.Linear:
    if seed_context is not None:
        _validate_probe_context(seed_context)
        with seeded_rng(seed_context.seed("probe-init", module=module, partition="fit", restart=restart)):
            return nn.Linear(input_dim, output_dim).to(device)
    # Keep the recorded legacy initialization and its seed arithmetic unchanged.
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(probe_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(probe_seed)
        return nn.Linear(input_dim, output_dim).to(device)


def fit_mpi3d_affine_probe(
    train_features: Tensor,
    train_labels: Tensor,
    validation_features: Tensor,
    validation_labels: Tensor,
    *,
    kind: ProbeKind,
    probe_seed: int,
    lambda_grid: Iterable[float],
    max_epochs: int,
    device: torch.device,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    seed_context: SeedContext | None = None,
    representation: str = "features",
    restart: int = 0,
) -> tuple[nn.Linear, dict[str, Any]]:
    """Fit/select a locked affine head using training and validation only.

    Selection maximizes mean position R² or balanced accuracy.  Ties retain the
    first lambda in the declared grid and the earliest epoch, a deterministic
    rule recorded in the returned metadata.  The untouched test split is not
    accepted by this function.
    """
    _validate_feature_matrix(train_features, name="train_features")
    _validate_feature_matrix(validation_features, name="validation_features")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("train and validation feature dimensions must match")
    if train_features.shape[0] != train_labels.shape[0] or validation_features.shape[0] != validation_labels.shape[0]:
        raise ValueError("feature and label sample counts must match")
    if max_epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("max_epochs, batch_size, and learning_rate must be positive")
    penalties = tuple(float(value) for value in lambda_grid)
    if not penalties or any(value < 0 for value in penalties):
        raise ValueError("lambda_grid must contain one or more non-negative values")

    if kind == "position":
        if train_labels.dtype != torch.float32 or validation_labels.dtype != torch.float32:
            raise ValueError("position labels must be normalized float32 coordinates")
        selection_metric = "mean_position_r2"
    else:
        if train_labels.dtype != torch.long or validation_labels.dtype != torch.long:
            raise ValueError("classification labels must use torch.long")
        selection_metric = "balanced_accuracy"

    stream_args = {}
    if seed_context is not None:
        _validate_probe_context(seed_context)
        stream_args = {"seed_context": seed_context, "module": f"{representation}/{kind}", "restart": restart}
    best_score = float("-inf")
    best_head_state: dict[str, Tensor] | None = None
    best_selection: dict[str, Any] | None = None
    history = []
    for penalty in penalties:
        head = _new_affine_head(
            train_features.shape[1], _probe_output_dimension(kind), probe_seed=probe_seed, device=device, **stream_args
        )
        optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        for epoch in range(1, max_epochs + 1):
            head.train()
            permutation = _epoch_permutation(
                train_features.shape[0], probe_seed=probe_seed, epoch=epoch, **stream_args
            )
            total_data_loss = 0.0
            total_samples = 0
            for start in range(0, train_features.shape[0], batch_size):
                indices = permutation[start : start + batch_size]
                batch_features = train_features[indices].to(device)
                batch_labels = train_labels[indices].to(device)
                output = head(batch_features)
                data_loss = (
                    F.mse_loss(output, batch_labels)
                    if kind == "position"
                    else F.cross_entropy(output, batch_labels)
                )
                loss = data_loss + penalty * head.weight.square().sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_data_loss += float(data_loss.item()) * len(indices)
                total_samples += len(indices)
            validation_metrics = evaluate_mpi3d_affine_probe(
                validation_features, validation_labels, head, kind, device, batch_size
            )
            score = validation_metrics[selection_metric]
            record = {
                "lambda": penalty,
                "epoch": epoch,
                "training_data_loss": total_data_loss / total_samples,
                "validation": validation_metrics,
            }
            history.append(record)
            if score > best_score:
                best_score = score
                best_head_state = deepcopy({name: value.detach().cpu() for name, value in head.state_dict().items()})
                best_selection = {"lambda": penalty, "epoch": epoch, "validation": validation_metrics}

    assert best_head_state is not None and best_selection is not None
    selected_head = _new_affine_head(
        train_features.shape[1], _probe_output_dimension(kind), probe_seed=probe_seed, device=device, **stream_args
    )
    selected_head.load_state_dict(best_head_state)
    metadata = {
        "kind": kind,
        "selection_metric": selection_metric,
        "tie_break": "first lambda in declared grid, then earliest epoch",
        "selected": best_selection,
        "history": history,
    }

    if seed_context is not None:
        metadata["seed_streams"] = seed_context.as_dict()
        metadata["probe_init_key"] = seed_context.key(
            "probe-init", module=stream_args["module"], partition="fit", restart=restart
        ).record()
        metadata["probe_order_epoch_zero_key"] = seed_context.key(
            "probe-order", module=stream_args["module"], partition="fit", epoch=0, restart=restart
        ).record()
        metadata["probe_order_rng"] = "numpy_pcg64_full_digest"
    return selected_head, metadata
