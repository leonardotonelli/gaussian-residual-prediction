"""Accepted Moving-MNIST campaign recipe, without changing legacy model semantics.

This is a fail-closed boundary for scientific runs, not a new model family.  The
accepted forward-only objectives and natural BatchNorm exposure already exist
in the four model classes.  This module binds those choices, structured seeds,
and their explicit local deviations to an auditable run contract.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

from .moving_mnist import GeneratorConfig
from .moving_mnist_full_training import FullTrainingConfig, ROLES, model_spec
from .seed_streams import CAMPAIGN, SeedContext, content_sha256, context_from_config


RECIPE_VERSION = "five-seed-moving-mnist-v1"
CAMPAIGN_STATUS = "five-seed-campaign-training"
IDENTITY_MANIFEST = "concept2-identities-v1-seed0.json"
GENERATOR = {
    "setting": "A", "gaussian_parameter": "std", "rendering": "bilinear-zero-pad",
    "boundary": "crop-no-bounce", "change_timing": "before-frame-4",
    "first_frame": "at-initial-center", "canvas_size": 64, "digit_size": 16,
    "initial_center_min": 8., "initial_center_max": 16.,
    "max_initial_velocity": 3., "noise_factor": 2. / 3.,
}
TRAINING = {
    "total_steps": 75000, "batch_size": 128, "seed": 0,
    "learning_rate": 1e-4, "weight_decay": 1e-4,
}
COMMON = {"spatial_strides": [2, 2, 2], "alignment": "forward-negative-cosine"}
REFERENCE = {**COMMON, "ema_decay": .996, "beta": .001, "scale_floor": 1e-6,
             "target_batchnorm": "independent-batch-stats",
             "posterior_gradients": "online-source-and-surrogate"}
SHARED = {**COMMON, "sigreg_weight": .1, "sigreg_projections": 128,
          "sigreg_knots": 17, "sigreg_interval": [.2, 4.],
          "sigreg_placement": "projector-output-after-bn-before-cosine"}
MODEL_OPTIONS = {
    "R0": REFERENCE, "R1": REFERENCE, "S0": SHARED,
    "S1": {**SHARED, "beta": .001, "scale_floor": 1e-6,
           "posterior_gradients": "shared-source-and-surrogate"},
}


def resolve_campaign_config(config: dict, seed_context: SeedContext | None) -> tuple[dict, dict]:
    """Validate all four roles before data access; return options and provenance.

    Logging/checkpoint cadence and data location can vary without changing the
    scientific recipe.  The full config is nevertheless hashed into the resume
    contract.  A software smoke uses a separate seed namespace and pauses this
    full-budget schedule; a reduced unit-test fixture is never a campaign run.
    """
    if config.get("status") != CAMPAIGN_STATUS or config.get("recipe_version") != RECIPE_VERSION:
        raise ValueError("Explicit versioned Moving-MNIST campaign recipe required")
    allowed = {"status", "recipe_version", "data_dir", "identity_manifest", "workers",
               "training", "models", "generator", "seed_streams"}
    if set(config) - allowed:
        raise ValueError(f"Unknown campaign configuration fields: {sorted(set(config) - allowed)}")
    if (not isinstance(seed_context, SeedContext) or seed_context.dataset != "moving_mnist"
            or seed_context.campaign != CAMPAIGN or seed_context.purpose == "evaluation"):
        raise ValueError("Campaign requires its explicit Moving-MNIST training seed context")
    configured_context = context_from_config(config)
    if configured_context is not None and configured_context != seed_context:
        raise ValueError("Conflicting explicit and configured campaign seed contexts")
    if config.get("identity_manifest") != IDENTITY_MANIFEST:
        raise ValueError("Campaign identity split differs from the accepted manifest")
    if config.get("generator") != GENERATOR:
        raise ValueError("Campaign requires the accepted Setting-A generator conventions")
    if config.get("models") != MODEL_OPTIONS:
        raise ValueError("All four model roles must match the accepted campaign recipe")
    if not isinstance(config.get("training"), dict):
        raise ValueError("Explicit campaign training settings required")
    for name in ("total_steps", "batch_size", "seed"):
        if type(config["training"].get(name)) is not int:
            raise ValueError(f"Campaign {name} must be an explicit integer")
    training = FullTrainingConfig(**config["training"])
    for name, expected in TRAINING.items():
        if getattr(training, name) != expected or name not in config["training"]:
            raise ValueError(f"Campaign training setting {name} must be explicit and equal {expected}")
    if not isinstance(config.get("workers"), int) or isinstance(config["workers"], bool) or config["workers"] < 0:
        raise ValueError("Campaign worker count must be a nonnegative integer")
    # The numerical seed field remains an inert legacy placeholder when a
    # structured context is supplied. No source of randomness uses it here.
    options = deepcopy(MODEL_OPTIONS)
    sigreg_seeds = {view: seed_context.seed("sigreg", view=view) for view in ("source", "target")}
    for role in ("S0", "S1"):
        # FullTrainer supplies both explicit view generators. Keep the unused
        # model-owned fallback generator in the same replication namespace too.
        options[role]["sigreg_seed"] = sigreg_seeds["source"]
    from .five_seed_campaign import manifest_binding
    binding = manifest_binding(seed_context)
    model_configs = {role: asdict(model_spec(role, options[role], training)[0]) for role in ROLES}
    descriptor = {
        "version": RECIPE_VERSION, "campaign": CAMPAIGN,
        "source_config": deepcopy(config), "configuration_sha256": content_sha256(config),
        "manifest_binding": binding, "seed_context": seed_context.as_dict(),
        "resolved_model_configs": model_configs, "generator": asdict(GeneratorConfig(**GENERATOR)),
        "training": asdict(training), "sigreg_stream_seeds": sigreg_seeds,
        "objective": {"alignment": "negative mean batch cosine; source to target only",
                      "kl": ".001 * mean examples sum residual coordinates KL(q||p)",
                      "sigreg": ".1 * .5 * (source SIGReg + target SIGReg)",
                      "sigreg_features": "real projector outputs after BN, before cosine normalization"},
        "optimizer": {"name": "AdamW", "betas": [.9, .999], "eps": 1e-8,
                      "weight_decay_policy": "all trainable parameters, including bias and BN affine",
                      "gradient_clipping": None, "mixed_precision": False, "accumulation_steps": 1},
        "schedule": {"name": "CosineAnnealingLR", "T_max": 75000, "eta_min": 0.,
                     "warmup_steps": 0, "order": ["optimizer", "parameter-only EMA for R roles", "scheduler"]},
        "initialization": {"version": "logical-module-pairing-v1",
                           "common_predictor_fan_in": 128, "extra_residual_columns_fan_in": 130},
        "fidelity": "accepted local AdaSSL-V-inspired recipe; unreleased author video details unresolved",
    }
    descriptor["sha256"] = content_sha256(descriptor)
    return options, descriptor


def model_recipe_report(model) -> dict:
    """Measure capacity and disclose the intentional forward/BN differences.

    Linear MAC counts exclude normalization, activation, and backbone work, and
    are an architecture-level accounting aid, not measured GPU throughput.
    """
    from torch import nn
    role = model.config.role
    reference = role.startswith("R")
    variational = role.endswith("1")
    encoder = model.online_encoder if reference else model.encoder
    modules = {"encoder": encoder, "predictor": model.predictor,
               "prior": getattr(model, "prior", None), "posterior": getattr(model, "posterior", None)}
    from math import prod
    shape, convolution_macs = (3, 64, 64), 0
    for layer in encoder.backbone.modules():
        if isinstance(layer, nn.Conv3d):
            shape = tuple((n + 2*p - d*(k-1) - 1)//s + 1 for n, p, d, k, s in
                          zip(shape, layer.padding, layer.dilation, layer.kernel_size, layer.stride))
            convolution_macs += prod(shape) * layer.out_channels * (layer.in_channels // layer.groups) * prod(layer.kernel_size)
    counts = {name: sum(p.numel() for p in module.parameters()) if module is not None else 0
              for name, module in modules.items()}
    macs = {name: sum(layer.in_features * layer.out_features for layer in module.modules()
                      if isinstance(layer, nn.Linear)) if module is not None else 0
            for name, module in modules.items()}
    bn_policy = {"online_encoder": 1 + int(variational), "target_encoder": 1} if reference else {
        "encoder": 2 + int(variational)}
    bn_policy.update({"predictor": 1})
    if variational:
        bn_policy.update(prior=1, posterior=1)
    return {
        # Architectural training roles remain the same after frozen evaluation
        # sets requires_grad=False on every parameter.
        "role": role, "parameter_counts": {
            "trainable": sum(counts.values()), "target": counts["encoder"] if reference else 0,
            "total": sum(p.numel() for p in model.parameters())}, "parameters_by_module": counts,
        "stochastic_extra_predictor_weights": 2048 if variational else 0,
        "linear_macs_per_call": macs,
        "convolution_macs_per_encoder_call": convolution_macs,
        "forward_weight_macs_per_training_example": ((convolution_macs + macs["encoder"]) * (3 if variational else 2)
                                                     + macs["predictor"] + macs["prior"] + macs["posterior"]),
        "compute_scope": "weighted forward operations only; excludes BN/activation/backward/optimizer/SIGReg; not wall-clock throughput",
        "encoder_forward_calls_per_training_example": 3 if variational else 2,
        "encoder_backward_calls_per_training_example": (2 if variational else 1) if reference else (3 if variational else 2),
        "bn_updates_per_optimizer_step": bn_policy,
        "target_parameter_gradients": not reference, "ema_parameter_only": reference,
        "training_posterior_inputs": ["source pixels", "later surrogate pixels"] if variational else [],
        "inference_inputs": ["source pixels"], "training_labels": [],
    }


def validate_campaign_contract(contract: dict, *, model=None) -> dict:
    """Revalidate a serialized recipe before campaign resume or frozen use.

    JSON round trips may turn tuples into lists; canonical hashes compare their
    serialized meaning. Legacy contracts do not call this opt-in validator.
    """
    from .moving_mnist_full_training import SEEDED_FULL_TRAINING_VERSION
    recipe = contract.get("campaign_recipe")
    if not isinstance(recipe, dict):
        raise ValueError("Missing campaign recipe in checkpoint contract")
    context = SeedContext.from_dict(contract.get("seed_context"))
    try:
        _, expected = resolve_campaign_config(recipe["source_config"], context)
        role = contract["model"]["role"]
        correct = (content_sha256(recipe) == content_sha256(expected)
                   and contract["version"] == SEEDED_FULL_TRAINING_VERSION
                   and contract["purpose"] == CAMPAIGN_STATUS
                   and content_sha256(contract["model"]) == content_sha256(expected["resolved_model_configs"][role])
                   and contract["training"] == expected["training"]
                   and contract["train_data"]["generator"] == expected["generator"]
                   and contract["train_data"]["seed_context"] == context.as_dict())
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete Moving-MNIST campaign contract") from exc
    if not correct:
        raise ValueError("Moving-MNIST campaign contract differs from the accepted recipe")
    if model is not None and contract.get("model_recipe") != model_recipe_report(model):
        raise ValueError("Checkpoint architecture/capacity report mismatch")
    return expected
