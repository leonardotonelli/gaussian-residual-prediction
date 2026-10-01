"""Preprint implementation: selected components from the research codebase."""
from typing import Any, Dict
from torch import nn
from .models import SmallViTEncoder


def build_encoder(cfg: Dict[str, Any]) -> nn.Module:
    """Build the image encoder selected by the config."""
    model_cfg = cfg["model"]
    encoder_name = str(model_cfg.get("encoder", "convnet"))
    if encoder_name == "vit":
        return SmallViTEncoder(
            image_size=int(cfg["data"]["image_size"]),
            patch_size=int(model_cfg["patch_size"]),
            embedding_dim=int(model_cfg["embedding_dim"]),
            token_dim=int(model_cfg["vit_dim"]),
            depth=int(model_cfg["vit_depth"]),
            num_heads=int(model_cfg["vit_heads"]),
            mlp_ratio=float(model_cfg["vit_mlp_ratio"]),
        )
    raise ValueError(f"Unknown encoder: {encoder_name}")
