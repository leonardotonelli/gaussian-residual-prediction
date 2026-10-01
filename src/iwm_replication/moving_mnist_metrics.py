"""Physical forecast scores. Inputs are velocities, never latent embeddings.

Return one score per source; aggregate sources equally. True future banks must
be independent of model/oracle forecast draws. Scores do not certify calibration.
"""

import torch


def physical_scores(predictions: torch.Tensor, truths: torch.Tensor, *,
                    chunk_size: int = 64) -> dict[str, torch.Tensor]:
    """ES uses ordered off-diagonal prediction pairs (unbiased U statistic).

    predictions: (Q, M, 2); truths: (Q, L, 2), pixels/frame. M=1 is a point
    forecast with exactly zero self-distance correction. Coverage is mean
    nearest-prediction squared distance, with per-future values for axis strata.
    Chunk both sample axes, bounding intermediate memory by Q * chunk_size**2.
    """
    if predictions.ndim != 3 or truths.ndim != 3 or predictions.shape[0] != truths.shape[0]:
        raise ValueError("Expected predictions (Q,M,2) and truths (Q,L,2)")
    if predictions.shape[-1] != 2 or truths.shape[-1] != 2 or min(*predictions.shape, *truths.shape) < 1:
        raise ValueError("Nonempty physical velocity arrays with last dimension 2 required")
    if predictions.device != truths.device or chunk_size < 1:
        raise ValueError("Inputs must share a device; chunk_size must be positive")
    if not torch.isfinite(predictions).all() or not torch.isfinite(truths).all():
        raise ValueError("Nonfinite physical forecasts cannot be silently dropped/clipped")
    # Float64 prevents small negative self distances from roundoff in scoring.
    p, y = predictions.double(), truths.double()
    q, m, _ = p.shape
    l = y.shape[1]
    cross = p.new_zeros(q)
    nearest = p.new_full((q, l), float("inf"))
    pairs = p.new_zeros(q)
    for a in range(0, m, chunk_size):
        pa = p[:, a:a + chunk_size]
        for b in range(0, l, chunk_size):
            distance = torch.linalg.vector_norm(pa[:, :, None] - y[:, None, b:b + chunk_size], dim=-1)
            cross += distance.sum((1, 2))
            nearest[:, b:b + chunk_size] = torch.minimum(nearest[:, b:b + chunk_size], distance.square().amin(1))
        if m > 1:
            for b in range(0, m, chunk_size):
                distance = torch.linalg.vector_norm(pa[:, :, None] - p[:, None, b:b + chunk_size], dim=-1)
                # Diagonal terms are exactly zero with explicit differences.
                pairs += distance.sum((1, 2))
    energy = cross / (m * l)
    if m > 1:
        energy -= pairs / (2 * m * (m - 1))
    return {"energy_score": energy, "coverage_squared": nearest.mean(1),
            "nearest_squared_per_future": nearest}


def estimate_source_velocity(source: torch.Tensor) -> torch.Tensor:
    """Pixel centroid displacement over observed frames, with no motion labels.

    source: (...,1,3,H,W). This provisional baseline can be biased by cropping;
    its observed-velocity error must be measured on development clips.
    """
    if source.ndim < 4 or source.shape[-4:-2] != (1, 3):
        raise ValueError("Expected (...,1,3,H,W) source clip")
    pixels = source.squeeze(-4).double()
    if not torch.isfinite(pixels).all() or (pixels < 0).any():
        raise ValueError("Expected finite nonnegative pixels")
    mass = pixels.sum((-2, -1))
    if (mass <= 0).any():
        raise ValueError("Cannot estimate motion from an empty frame")
    y, x = torch.meshgrid(torch.arange(pixels.shape[-2], device=pixels.device),
                          torch.arange(pixels.shape[-1], device=pixels.device), indexing="ij")
    centers = torch.stack(((pixels * x).sum((-2, -1)) / mass,
                           (pixels * y).sum((-2, -1)) / mass), dim=-1)
    return (centers[..., 2, :] - centers[..., 0, :]) / 2
