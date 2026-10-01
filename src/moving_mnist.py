"""Versioned concept2 simulator. Provisional rendering/law for smoke tests only.

Coordinates and velocities are (x, y), in pixels and pixels/frame. Tensor clips
are (channel=1, time=3, height=64, width=64), ready for a 3D CNN. Metadata is
returned separately from pixel observations and is for evaluation only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .moving_mnist_data import content_hash


GENERATOR_VERSION = "concept2-moving-mnist-smoke-v1"


@dataclass(frozen=True)
class GeneratorConfig:
    setting: str = "A"
    gaussian_parameter: str = "std"
    canvas_size: int = 64
    digit_size: int = 16
    initial_center_min: float = 8.0
    initial_center_max: float = 16.0
    max_initial_velocity: float = 3.0
    noise_factor: float = 2.0 / 3.0
    rendering: str = "bilinear-zero-pad"
    boundary: str = "crop-no-bounce"
    change_timing: str = "before-frame-4"
    first_frame: str = "at-initial-center"

    def __post_init__(self):
        if self.setting not in ("A", "B") or self.gaussian_parameter not in ("std", "variance"):
            raise ValueError("Expected setting A/B and Gaussian parameter std/variance")
        if (self.canvas_size, self.digit_size) != (64, 16):
            raise ValueError("Version 1 fixes 64x64 canvas and 16x16 digit")
        if (self.rendering, self.boundary, self.change_timing, self.first_frame) != (
            "bilinear-zero-pad", "crop-no-bounce", "before-frame-4", "at-initial-center"
        ):
            raise ValueError("Unsupported rendering or timing policy")
        if not (0 <= self.initial_center_min <= self.initial_center_max <= self.canvas_size):
            raise ValueError("Invalid initial-center range")
        if not np.isfinite([self.max_initial_velocity, self.noise_factor]).all() or min(
            self.max_initial_velocity, self.noise_factor
        ) < 0:
            raise ValueError("Motion scales must be finite and nonnegative")


@dataclass(frozen=True)
class SourceState:
    image_id: str
    digit: int
    center: tuple[float, float]
    velocity: tuple[float, float]


def random_stream(seed: int, query: int, stream: int) -> np.random.Generator:
    """Independent addressable streams: 0=source, 1=truth, 2=oracle, 3=training."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, query, stream])))


def sample_source(image_id: str, digit: int, rng: np.random.Generator,
                  cfg: GeneratorConfig) -> SourceState:
    if not 0 <= digit <= 9:
        raise ValueError("MNIST digit must be in [0, 9]")
    return SourceState(image_id, int(digit),
                       tuple(rng.uniform(cfg.initial_center_min, cfg.initial_center_max, 2)),
                       tuple(rng.uniform(0, cfg.max_initial_velocity, 2)))


def sample_future_velocities(state: SourceState, count: int, rng: np.random.Generator,
                             cfg: GeneratorConfig) -> tuple[np.ndarray, np.ndarray]:
    if count < 1:
        raise ValueError("At least one future is required")
    # Bernoulli=1 means y changes, exactly as Eq. 16/17.
    probability_y = 0.5 if cfg.setting == "A" else 0.1 + state.digit * 0.8 / 9
    axes = (rng.random(count) < probability_y).astype(np.int64)
    velocities = np.tile(np.asarray(state.velocity, dtype=np.float64), (count, 1))
    scale = cfg.noise_factor * velocities[np.arange(count), axes]
    if cfg.gaussian_parameter == "variance":
        scale = np.sqrt(scale)
    velocities[np.arange(count), axes] += rng.normal(size=count) * scale
    return velocities, axes


def trajectory_centers(state: SourceState, future_velocity: np.ndarray) -> np.ndarray:
    initial = np.asarray(state.center)
    velocity = np.asarray(state.velocity)
    history = initial + np.arange(3)[:, None] * velocity
    future = history[-1] + np.arange(1, 7)[:, None] * future_velocity
    return np.concatenate((history, future))


def render_centers(image: np.ndarray, centers: np.ndarray, cfg: GeneratorConfig) -> torch.Tensor:
    """Bilinear resize (antialias) then translation; zero outside the canvas.

    Pixel centers are integers. A digit centered at c has leftmost pixel center
    c - (digit_size - 1)/2. Cropping neither reflects nor alters latent velocity.
    """
    if image.shape != (28, 28) or image.dtype != np.uint8:
        raise ValueError("Expected one uint8 28x28 MNIST image")
    centers = np.asarray(centers)
    if centers.ndim != 2 or centers.shape[1] != 2 or not np.isfinite(centers).all():
        raise ValueError("Expected finite (frames, 2) centers")
    digit = torch.from_numpy(image.copy()).float()[None, None] / 255
    digit = F.interpolate(digit, size=(cfg.digit_size, cfg.digit_size), mode="bilinear",
                          align_corners=False, antialias=True)
    y, x = torch.meshgrid(torch.arange(cfg.canvas_size), torch.arange(cfg.canvas_size), indexing="ij")
    coordinates = torch.stack((x, y), dim=-1).float()[None]
    centers_tensor = torch.as_tensor(centers, dtype=torch.float32)[:, None, None]
    # align_corners=False normalized coordinates map the digit's half-pixel edges to +/-1.
    grid = 2 * (coordinates - centers_tensor) / cfg.digit_size
    frames = F.grid_sample(digit.expand(len(centers), -1, -1, -1), grid,
                           mode="bilinear", padding_mode="zeros", align_corners=False)
    return frames[:, 0]  # (T, H, W)


def render_video(image: np.ndarray, state: SourceState, future_velocity: np.ndarray,
                 cfg: GeneratorConfig) -> dict[str, torch.Tensor]:
    frames = render_centers(image, trajectory_centers(state, future_velocity), cfg)
    return {"source": frames[0:3].unsqueeze(0), "target": frames[3:6].unsqueeze(0),
            "surrogate": frames[6:9].unsqueeze(0)}


def source_manifest(states: list[SourceState], cfg: GeneratorConfig, *, seed: int,
                    split_hash: str, split: str) -> dict:
    payload = {"generator_version": GENERATOR_VERSION, "config": asdict(cfg),
               "status": "development-smoke-only", "seed": seed,
               "identity_manifest_sha256": split_hash, "split": split,
               "streams": {"source": 0, "truth": 1, "oracle": 2, "training": 3},
               "sources": [asdict(state) for state in states]}
    return {**payload, "sha256": content_hash(payload)}
