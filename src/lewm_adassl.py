"""Preprint implementation: selected components from the research codebase."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from .anti_collapse import AntiCollapseLosses, gather_global_batch


class PatchSIGReg(nn.Module):
    """Epps-Pulley SIGReg, evaluated independently at every patch position."""

    def __init__(
        self,
        *,
        global_batch_size: int,
        num_projections: int,
        knots: int,
        interval: tuple[float, float],
        rng_seed: int,
    ) -> None:
        super().__init__()
        if global_batch_size < 2 or num_projections <= 0 or knots < 2:
            raise ValueError("SIGReg requires batch >= 2, projections > 0, and knots >= 2")
        start, end = interval
        if not 0.0 <= start < end:
            raise ValueError("SIGReg interval must satisfy 0 <= start < end")
        self.global_batch_size = global_batch_size
        self.num_projections = num_projections
        self.rng_seed = rng_seed
        self._generator = torch.Generator(device="cpu").manual_seed(rng_seed)

        t = torch.linspace(start, end, knots, dtype=torch.float32)
        dt = (end - start) / (knots - 1)
        trapezoid = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        trapezoid[[0, -1]] = dt
        gaussian_characteristic = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t, persistent=True)
        self.register_buffer("phi", gaussian_characteristic, persistent=True)
        self.register_buffer(
            "integration_weights",
            trapezoid * gaussian_characteristic,
            persistent=True,
        )

    def rng_state(self) -> Tensor:
        return self._generator.get_state().clone()

    def set_rng_state(self, state: Tensor) -> None:
        self._generator.set_state(state.cpu())

    def forward(self, local_tokens: Tensor, *, generator: torch.Generator | None = None) -> AntiCollapseLosses:
        if local_tokens.ndim != 3:
            raise ValueError("SIGReg tokens must have shape [local_batch, patches, features]")
        tokens = gather_global_batch(local_tokens).float()
        if tokens.shape[0] != self.global_batch_size:
            raise ValueError(
                f"SIGReg expected global batch {self.global_batch_size}, got {tokens.shape[0]}"
            )
        feature_dim = tokens.shape[-1]
        directions = torch.randn(
            feature_dim,
            self.num_projections,
            generator=self._generator if generator is None else generator,
            dtype=torch.float32,
            device="cpu",
        ).to(tokens.device)
        directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
        projected = torch.einsum("pbd,dm->pbm", tokens.transpose(0, 1), directions)
        projected_t = projected.unsqueeze(-1) * self.t
        error = (projected_t.cos().mean(dim=1) - self.phi).square()
        error = error + projected_t.sin().mean(dim=1).square()
        sigreg = ((error @ self.integration_weights) * tokens.shape[0]).mean()

        centered = tokens.transpose(0, 1) - tokens.transpose(0, 1).mean(
            dim=1, keepdim=True
        )
        std = (centered.square().sum(dim=1) / (tokens.shape[0] - 1) + 1e-4).sqrt()
        zero = sigreg.new_zeros(())
        return AntiCollapseLosses(
            auxiliary=sigreg,
            variance=zero,
            covariance=zero,
            sigreg=sigreg,
            mean_std=std.mean(),
            min_std=std.min(),
        )
