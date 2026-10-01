"""Preprint implementation: selected components from the research codebase."""
from __future__ import annotations
import torch
from torch import Tensor, nn


class SmallViTEncoder(nn.Module):
    """Encode a fixed-size RGB image with a small ViT and one CLS latent.

    ``forward`` preserves the encoder interface used by the current IWM loop:

        image [B, 3, H, W] -> embedding [B, D]

    ``forward_tokens`` exposes the contextualized CLS and patch tokens for the
    later patch-token prediction stage:

        image [B, 3, H, W] -> tokens [B, 1 + N, token_dim]
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        embedding_dim: int = 64,
        token_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size != 0:
            raise ValueError("image_size must be positive and divisible by patch_size")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if depth <= 0 or mlp_ratio <= 0:
            raise ValueError("depth and mlp_ratio must be positive")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.patch_embedding = nn.Conv2d(  # Learn a vector for each image patch.
            3,
            token_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))  # Learnable global token.
        self.position_embeddings = nn.Parameter(  # Learned location-specific vectors.
            torch.zeros(1, self.num_patches + 1, token_dim)
        )
        block = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=int(token_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            block,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(token_dim)
        self.output_projection = nn.Linear(token_dim, embedding_dim)
        self._reset_parameters()
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def _reset_parameters(self) -> None:
        """Initialize each transformer block independently."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)

    def forward_tokens(self, image: Tensor) -> Tensor:
        """Return the contextualized CLS token followed by patch tokens."""
        patch_tokens = self._patchify(image)
        cls_token = self.cls_token.expand(image.shape[0], -1, -1)
        tokens = torch.cat([cls_token, patch_tokens], dim=1)
        tokens = tokens + self.position_embeddings
        return self.norm(self.blocks(tokens))

    def forward_context(self, image: Tensor, context_indices: Tensor) -> Tensor:
        """Encode only visible patch tokens at their original image positions.

        This is the source-encoder path for masked IWM training. The returned
        sequence has no CLS token because it supplies context to the predictor
        rather than a global image representation.
        """
        if context_indices.ndim != 1:
            raise ValueError("context_indices must have shape [num_context_patches]")
        if context_indices.dtype != torch.long:
            raise ValueError("context_indices must use torch.long dtype")
        if context_indices.numel() == 0:
            raise ValueError("context_indices cannot be empty")
        if context_indices.min().item() < 0 or context_indices.max().item() >= self.num_patches:
            raise ValueError("context_indices contain an out-of-range patch index")

        patch_tokens = self._patchify(image)
        positions = self.position_embeddings[:, 1:, :]
        context_tokens = patch_tokens[:, context_indices] + positions[:, context_indices]
        return self.norm(self.blocks(context_tokens))

    def _patchify(self, image: Tensor) -> Tensor:
        """Convert an image batch into uncontextualized patch embeddings."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        if tuple(image.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(
                f"expected images of size {self.image_size}x{self.image_size}, "
                f"got {image.shape[-2]}x{image.shape[-1]}"
            )

        return self.patch_embedding(image).flatten(2).transpose(1, 2)

    def forward(self, image: Tensor) -> Tensor:
        tokens = self.forward_tokens(image)
        embedding = self.output_projection(tokens[:, 0])
        return torch.nn.functional.normalize(embedding, dim=-1)
