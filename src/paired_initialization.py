"""Explicit role-paired initialization for the versioned five-seed campaign.

Each logical leaf module has its own keyed CPU RNG.  Parameters are generated
on CPU and copied into the existing objects, so constructor draw order, device,
and role-specific modules cannot shift another module's initialization.  The
legacy constructors are unchanged; callers opt in before creating optimizers.

Moving-MNIST's 128-input deterministic and 130-input stochastic predictors cannot
both retain their original fan-in distribution and have equal shared weights.
The campaign uses the deterministic default U(-1/sqrt(128), 1/sqrt(128)) for the
common 128 columns.  The two stochastic-only columns use an independent key and
the original U(-1/sqrt(130), 1/sqrt(130)) distribution.  Thus the stochastic
common-column bound increases by sqrt(130/128)-1 (about 0.78%).  This explicit
pairing adjustment does not alter the predictor architecture.  Gaussian video
heads retain Xavier-uniform with ReLU gain and zero output biases; image heads
retain their ordinary PyTorch defaults.  The image ViT retains its custom
Xavier/truncated-normal initializers, including its CLS/position parameters.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from .seed_streams import seeded_rng

if TYPE_CHECKING:
    from .seed_streams import SeedContext


INITIALIZATION_VERSION = "logical-module-pairing-v1"


def _tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _fill(value: torch.Tensor | None, initializer) -> None:
    if value is not None:
        canonical = torch.empty_like(value, device="cpu")
        initializer(canonical)
        value.copy_(canonical)


def _default_linear_or_conv(module: nn.Module) -> None:
    _fill(module.weight, lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5)))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
    bound = 1 / math.sqrt(fan_in) if fan_in else 0
    _fill(module.bias, lambda b: nn.init.uniform_(b, -bound, bound))


def _reset_leaf(module: nn.Module, recipe: str) -> None:
    """Retain each architecture's distributions without constructor RNG draws."""
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if recipe == "video-gaussian" and isinstance(module, nn.Linear):
            _fill(module.weight, lambda w: nn.init.xavier_uniform_(w, gain=math.sqrt(2)))
            _fill(module.bias, nn.init.zeros_)
        elif recipe == "vit" and isinstance(module, nn.Linear):
            _fill(module.weight, lambda w: nn.init.trunc_normal_(w, std=0.02))
            _fill(module.bias, nn.init.zeros_)
        elif recipe == "vit" and isinstance(module, nn.Conv2d):
            _fill(module.weight, nn.init.xavier_uniform_)
            _fill(module.bias, nn.init.zeros_)
        else:
            _default_linear_or_conv(module)
    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
        _fill(module.weight, nn.init.ones_)
        _fill(module.bias, nn.init.zeros_)
        _fill(module.running_mean, nn.init.zeros_)
        _fill(module.running_var, nn.init.ones_)
        _fill(module.num_batches_tracked, nn.init.zeros_)
    elif isinstance(module, nn.LayerNorm):
        _fill(module.weight, nn.init.ones_)
        _fill(module.bias, nn.init.zeros_)
    elif isinstance(module, nn.MultiheadAttention) and recipe == "vit":
        if not module._qkv_same_embed_dim or module.bias_k is not None or module.bias_v is not None:
            raise ValueError("Paired ViT initialization expects the existing same-dimension attention")
        _fill(module.in_proj_weight, nn.init.xavier_uniform_)
        _fill(module.in_proj_bias, nn.init.zeros_)
        # out_proj is initialized independently as the child Linear module.
    elif recipe == "vit" and hasattr(module, "cls_token") and hasattr(module, "position_embeddings"):
        _fill(module.cls_token, lambda w: nn.init.trunc_normal_(w, std=0.02))
        _fill(module.position_embeddings, lambda w: nn.init.trunc_normal_(w, std=0.02))
    else:
        raise TypeError(f"No paired initialization recipe for {type(module).__name__}")


def _logical_blocks(model):
    from .moving_mnist_models import MovingMNISTReference
    from .moving_mnist_shared import MovingMNISTS0
    from .mpi3d_byol import MPI3DGlobal

    if isinstance(model, (MovingMNISTReference, MovingMNISTS0)):
        reference = isinstance(model, MovingMNISTReference)
        encoder = model.online_encoder if reference else model.encoder
        path = "online_encoder" if reference else "encoder"
        blocks = [
            ("encoder.backbone", path + ".backbone", encoder.backbone, "default"),
            ("encoder.projector", path + ".projector", encoder.projector, "default"),
            ("predictor", "predictor", model.predictor, "video-predictor"),
        ]
        family = "moving_mnist"
        head_recipe = "video-gaussian"
        target = (encoder, model.target_encoder, path, "target_encoder") if reference else None
    elif isinstance(model, MPI3DGlobal):
        blocks = [
            ("encoder.backbone", "online.encoder", model.online.encoder, "vit"),
            ("encoder.projector", "online.projector", model.online.projector, "default"),
            ("predictor", "predictor", model.predictor, "default"),
        ]
        family = "mpi3d"
        head_recipe = "default"
        target = ((model.online, model.target, "online", "target")
                  if getattr(model, "uses_ema", True) else None)
    else:
        raise TypeError(f"Unsupported paired model: {type(model).__name__}")
    for name in ("prior", "posterior"):
        module = getattr(model, name, None)
        if module is not None:
            blocks.append((name, name, module, head_recipe))
    return family, blocks, target


@torch.no_grad()
def initialize_paired_model(model: nn.Module, context: SeedContext) -> dict:
    """Initialize a fresh supported model and return JSON-serializable provenance.

    R0/R1 online encoders and S0/S1 shared encoders use identical logical keys.
    Initial EMA targets are exact copies including BatchNorm buffers, retaining
    their frozen parameters and the existing later parameter-only EMA policy.
    Existing parameter identities, requires_grad flags, train/eval flags, and
    caller RNG state are preserved.  Model-owned SIGReg state is not reset here;
    its separate stream is configured by the training integration.
    """
    family, blocks, target = _logical_blocks(model)
    if context.dataset != family or context.purpose == "evaluation":
        raise ValueError("Paired initialization requires a matching dataset training seed context")
    seeds = {}
    hashes = {}
    mapping = {}

    def seeded_reset(key, reset):
        seed = context.seed("initialization", module=key)
        seeds[key] = seed
        with seeded_rng(seed):
            reset()

    for logical, actual, block, recipe in blocks:
        mapping[logical] = actual
        for suffix, module in block.named_modules():
            state = dict(module.named_parameters(recurse=False))
            state.update(dict(module.named_buffers(recurse=False)))
            if not state:
                continue
            key = logical + ("." + suffix if suffix else "")
            if recipe == "video-predictor" and suffix == "0":
                if not isinstance(module, nn.Linear) or module.in_features not in (128, 130) or module.bias is not None:
                    raise ValueError("Unexpected Moving-MNIST predictor input layer")
                common = module.weight[:, :128]
                seeded_reset(key + ".common", lambda: _fill(
                    common, lambda w: nn.init.uniform_(w, -1 / math.sqrt(128), 1 / math.sqrt(128))))
                hashes[key + ".weight.common"] = _tensor_hash(common)
                if module.in_features == 130:
                    residual = module.weight[:, 128:]
                    seeded_reset(key + ".residual", lambda: _fill(
                        residual, lambda w: nn.init.uniform_(w, -1 / math.sqrt(130), 1 / math.sqrt(130))))
                    hashes[key + ".weight.residual"] = _tensor_hash(residual)
            else:
                seed_key = key + ".tokens" if recipe == "vit" and not suffix else key
                seeded_reset(seed_key, lambda: _reset_leaf(module, recipe))
            for name, value in state.items():
                hashes[key + "." + name] = _tensor_hash(value)

    target_copy = None
    if target is not None:
        source, destination, source_path, target_path = target
        destination.load_state_dict(source.state_dict(), strict=True)
        target_copy = {"source": source_path, "target": target_path, "includes_buffers": True}
    for counter in ("ema_updates", "optimizer_updates"):
        if hasattr(model, counter):
            getattr(model, counter).zero_()

    return {
        "version": INITIALIZATION_VERSION,
        "family": family,
        "logical_module_mapping": mapping,
        "initialization_seeds": seeds,
        "tensor_hashes": hashes,
        "target_copy": target_copy,
        "moving_mnist_predictor_common_fan_in": 128 if family == "moving_mnist" else None,
        "moving_mnist_predictor_residual_fan_in": 130 if family == "moving_mnist" else None,
        "parameter_generation_device": "cpu",
    }
