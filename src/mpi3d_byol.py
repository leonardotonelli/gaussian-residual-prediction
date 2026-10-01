"""Preprint implementation: selected components from the research codebase."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .builders import build_encoder


SOURCE_FILES = (
    "scripts/train_mpi3d_byol.py", "scripts/evaluate_mpi3d_byol.py",
    "scripts/evaluate_five_seed_mpi3d.py",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def provenance(cfg):
    # Include transitive data/scoring helpers, not just the directly imported files.
    paths = (*sorted(Path("src").glob("*.py")),
             *SOURCE_FILES, *cfg["data"]["position_manifest_paths"].values())
    return {str(p): file_hash(p) for p in paths}


def verify_provenance(expected):
    for path, digest in expected.items():
        if file_hash(path) != digest:
            raise ValueError(f"Run source/manifest changed: {path}. Restore the recorded version.")


def mlp(sizes, *, output_bn=False):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        last = i == len(sizes) - 2
        layers.append(nn.Linear(a, b, bias=last and not output_bn))
        if not last or output_bn:
            layers.append(nn.BatchNorm1d(b))
        if not last:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ImageBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = build_encoder(cfg)
        # The full patch-context path does not use either legacy CLS parameter.
        self.encoder.cls_token.requires_grad_(False)
        self.encoder.output_projection.requires_grad_(False)
        m = cfg["model"]
        self.projector = mlp((m["vit_dim"], m["head_hidden_dim"],
                              m["head_hidden_dim"], m["projector_dim"]), output_bn=True)

    def forward(self, images):
        indices = torch.arange(self.encoder.num_patches, device=images.device)
        tokens = self.encoder.forward_context(images, indices)
        pooled = tokens.mean(1)
        projected = self.projector(pooled)
        return {"flattened": tokens.flatten(1), "pooled": pooled, "projector": projected,
                "normalized_projector": F.normalize(projected, dim=-1)}


class GaussianDistribution(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.net = mlp((input_dim, hidden, 2))

    def forward(self, inputs):
        mean, log_std = self.net(inputs).chunk(2, -1)
        return mean, log_std.clamp(-5.0, 2.0)


MATCHED_VERSION = "mpi3d-global-four-role-v2"


MATCHED_ROLES = ("R0", "R1", "S0", "S1")


MATCHED_TARGET_BN = "role_specific_independent_ema_or_shared_separate_passes"


_MATCHED_MODEL = {
    "encoder": "vit", "embedding_dim": 64, "patch_size": 8,
    "vit_dim": 256, "vit_depth": 6, "vit_heads": 8, "vit_mlp_ratio": 2.0,
    "action_dim": 2, "residual_dim": 1, "projector_dim": 128, "head_hidden_dim": 1024,
}


_MATCHED_OPTIM = {
    "lr": .001, "betas": [.9, .999], "learning_rate_schedule_scale": 1.25,
    "weight_decay_start": .04, "weight_decay_end": .4,
    "ema_decay_start": .996, "ema_decay_end": 1.0,
}


_MATCHED_SIGREG = {"weight": .1, "projections": 128, "knots": 17, "interval": [.2, 4.]}


def resolve_matched_config(base, *, role, replication, purpose="main-training"):
    """Resolve one of all five main replications or separate development/smoke 0."""
    from .seed_streams import SeedContext
    from .five_seed_campaign import manifest_binding

    if role not in MATCHED_ROLES or purpose not in (
            "main-training", "development-training", "software-smoke"):
        raise ValueError("Expected one of R0/R1/S0/S1 and a training purpose")
    context = SeedContext("mpi3d", purpose, replication)
    cfg = deepcopy(base)
    cfg.update(protocol=MATCHED_VERSION, role=role,
               variant="adassl" if role.endswith("1") else "byol", seed=replication,
               stage=purpose, target_bn=MATCHED_TARGET_BN,
               seed_streams=context.as_dict(), campaign_manifest=manifest_binding(context))
    cfg["data"]["condition"] = "S"
    cfg.setdefault("sigreg", deepcopy(_MATCHED_SIGREG))
    cfg["run_name"] = f"mpi3d_s_{role.lower()}_v2_{purpose}_rep{replication}_{cfg['train']['epochs']}ep"
    validate_matched_config(cfg)
    return cfg


def validate_matched_config(cfg):
    """Reject changed scientific recipes and mislabeled or unbound seed contexts.

    Software-smoke namespace alone permits reduced dimensions/budgets for local
    synthetic tests. Main and full development recipes use accepted fixed values.
    """
    from .seed_streams import SeedContext
    from .five_seed_campaign import validate_manifest_binding

    if cfg.get("protocol") != MATCHED_VERSION or cfg.get("role") not in MATCHED_ROLES:
        raise ValueError("Unsupported matched MPI3D protocol/role")
    context = SeedContext.from_dict(cfg.get("seed_streams"))
    if (context.dataset != "mpi3d" or context.purpose == "evaluation"
            or context.replication != cfg.get("seed") or cfg.get("stage") != context.purpose):
        raise ValueError("Matched MPI3D configuration must identify its exact training replication")
    validate_manifest_binding(cfg.get("campaign_manifest"), context)
    if cfg.get("variant") != ("adassl" if cfg["role"].endswith("1") else "byol"):
        raise ValueError("Matched role and deterministic/stochastic variant disagree")
    data, model = cfg["data"], cfg["model"]
    if data.get("dataset") != "mpi3d" or data.get("condition") != "S":
        raise ValueError("Matched campaign uses balanced stochastic MPI3D-S only")
    if any(k.startswith("execution_success_probability") for k in data):
        raise ValueError("Matched campaign uses the canonical balanced outcome law")
    if (model.get("encoder") != "vit" or model.get("action_dim") != 2
            or model.get("residual_dim") != 1 or cfg.get("target_bn") != MATCHED_TARGET_BN):
        raise ValueError("Matched architecture/action/residual/BatchNorm contract changed")
    if (cfg["loss"] != {"alignment": "negative_cosine", "beta": .001}
            or cfg.get("sigreg") != _MATCHED_SIGREG):
        raise ValueError("Matched alignment, KL or SIGReg recipe changed")
    if cfg["optim"] != _MATCHED_OPTIM:
        raise ValueError("Matched MPI3D optimizer/schedule recipe changed")
    if min(model["projector_dim"], model["head_hidden_dim"], model["vit_dim"], model["vit_depth"],
           cfg["train"]["epochs"], cfg["train"]["transition_samples_per_epoch"]) <= 0:
        raise ValueError("Matched dimensions and budget must be positive")
    if data["batch_size"] < 2 or cfg["train"]["transition_samples_per_epoch"] % data["batch_size"]:
        raise ValueError("Use complete batches with at least two examples")
    if cfg["train"]["epochs"] > 100:
        raise ValueError("Epoch budget exceeds the fixed seed manifest domain")
    if cfg["train"]["transition_samples_per_epoch"] % 2:
        raise ValueError("Balanced stochastic training requires an even epoch size")
    if context.purpose != "software-smoke":
        if model != _MATCHED_MODEL or data["image_size"] != 64 or data["batch_size"] != 64:
            raise ValueError("Main/development requires the accepted full MPI3D architecture and global batch")
        for key, value in {"epochs": 100, "warmup_epochs": 10, "transition_samples_per_epoch": 65536}.items():
            if cfg["train"][key] != value:
                raise ValueError("Main/development requires the accepted full MPI3D optimization budget")
    if cfg["eval"]["quantiles"] < 2 or cfg["eval"]["gap_epsilon"] <= 0:
        raise ValueError("Invalid forecast quadrature or gap threshold")


class MPI3DGlobal(nn.Module):
    """Matched global image/action models; unsupervised scalar Gaussian residual.

    Shared roles run source then target once each, reusing target features for
    S1's posterior. EMA roles retain the old independent-BN target pass and R1's
    differentiable extra online target pass. Inputs never include outcome labels.
    """
    def __init__(self, cfg):
        from .lewm_adassl import PatchSIGReg
        from .seed_streams import SeedContext

        super().__init__()
        validate_matched_config(cfg)
        self.role = cfg["role"]
        self.uses_ema = self.role.startswith("R")
        self.is_stochastic = self.role.endswith("1")
        self.variant = cfg["variant"]
        self.online = ImageBranch(cfg)
        if self.uses_ema:
            self.target = deepcopy(self.online).requires_grad_(False)
            self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))
        m = cfg["model"]
        d, h = m["projector_dim"], m["head_hidden_dim"]
        self.predictor = mlp((d + 3, h, h, d))
        self.prior = GaussianDistribution(d + 2, h) if self.is_stochastic else None
        self.posterior = GaussianDistribution(2 * d + 2, h) if self.is_stochastic else None
        self.beta = cfg["loss"]["beta"] if self.is_stochastic else 0.0
        self.register_buffer("optimizer_updates", torch.zeros((), dtype=torch.long))
        self.sigreg_generators = ()
        self.regularizer = None
        self.sigreg_weight = 0.0
        if not self.uses_ema:
            context = SeedContext.from_dict(cfg["seed_streams"])
            self.sigreg_generators = tuple(context.torch_generator("sigreg", view=view)
                                           for view in ("source", "target"))
            s = cfg["sigreg"]
            self.sigreg_weight = s["weight"]
            self.regularizer = PatchSIGReg(global_batch_size=cfg["data"]["batch_size"],
                num_projections=s["projections"], knots=s["knots"], interval=tuple(s["interval"]),
                rng_seed=context.seed("sigreg", view="source"))

    @property
    def target_branch(self):
        """The actual representation space predicted by this role."""
        return self.target if self.uses_ema else self.online

    def get_extra_state(self):
        return {"version": MATCHED_VERSION, "role": self.role,
                "sigreg_rng": {view: g.get_state().clone() for view, g in
                               zip(("source", "target"), self.sigreg_generators)}}

    def set_extra_state(self, state):
        if (state.get("version") != MATCHED_VERSION or state.get("role") != self.role
                or set(state.get("sigreg_rng", {})) !=
                (set(("source", "target")) if self.sigreg_generators else set())):
            raise ValueError("Matched MPI3D SIGReg/role snapshot mismatch")
        for view, generator in zip(("source", "target"), self.sigreg_generators):
            generator.set_state(state["sigreg_rng"][view].cpu())

    def forward(self, source, target, action, *, generator=None):
        if action.shape != (len(source), 2) or source.shape != target.shape:
            raise ValueError("Expected matched images and (B,2) commands")
        z = self.online(source)["projector"]
        if self.uses_ema:
            with torch.no_grad():
                target_z = self.target(target)["projector"]
            future_z = self.online(target)["projector"] if self.is_stochastic else None
        else:
            target_z = self.online(target)["projector"]
            future_z = target_z
        residual = z.new_zeros(len(z), 1)
        kl = z.new_zeros(())
        if self.is_stochastic:
            pm, ps = self.prior(torch.cat((z, action), -1))
            qm, qs = self.posterior(torch.cat((z, action, future_z), -1))
            residual = qm + qs.exp() * torch.randn(qm.shape, device=qm.device,
                                                   dtype=qm.dtype, generator=generator)
            kl = (ps - qs + .5 * ((qs - ps).mul(2).exp()
                  + (qm - pm).square() * (-2 * ps).exp() - 1)).sum(-1).mean()
        prediction = self.predictor(torch.cat((z, action, residual), -1))
        alignment = -F.cosine_similarity(prediction, target_z, dim=-1).mean()
        source_reg = target_reg = z.new_zeros(())
        if self.regularizer is not None:
            source_reg = self.regularizer(z[:, None], generator=self.sigreg_generators[0]).sigreg
            target_reg = self.regularizer(target_z[:, None], generator=self.sigreg_generators[1]).sigreg
        sigreg = .5 * (source_reg + target_reg)
        return {"loss": alignment + self.beta * kl + self.sigreg_weight * sigreg,
                "alignment": alignment, "kl": kl, "sigreg": sigreg,
                "source_sigreg": source_reg, "target_sigreg": target_reg}

    @torch.no_grad()
    def after_optimizer_step(self, decay):
        if self.uses_ema:
            for name, parameter in self.target.named_parameters():
                parameter.lerp_(self.online.get_parameter(name), 1 - decay)
            self.ema_updates.add_(1)
        self.optimizer_updates.add_(1)

    def _require_eval(self):
        if any(m.training for m in self.modules()):
            raise RuntimeError("Forecasting/encoding requires frozen evaluation mode")

    @torch.no_grad()
    def encode(self, images, *, branch="online"):
        self._require_eval()
        if branch not in ("online", "target"):
            raise ValueError("Expected online or target branch")
        return (self.online if branch == "online" else self.target_branch)(images)

    @torch.no_grad()
    def forecast(self, source, action, *, quantiles=16, fixed_residual=False, normalize=True):
        self._require_eval()
        z = self.online(source)["projector"]
        return self.forecast_features(z, action, quantiles=quantiles,
                                      fixed_residual=fixed_residual, normalize=normalize)

    @torch.no_grad()
    def forecast_features(self, z, action, *, quantiles=16, fixed_residual=False, normalize=True):
        self._require_eval()
        if action.shape != (len(z), 2):
            raise ValueError("Expected (B,2) command")
        if not self.is_stochastic:
            residuals = z.new_zeros(len(z), 1, 1)
        else:
            mean, log_std = self.prior(torch.cat((z, action), -1))
            if fixed_residual:
                residuals = mean[:, None]
            else:
                if quantiles < 2:
                    raise ValueError("Use at least two Gaussian quadrature points")
                u = (torch.arange(quantiles, device=z.device, dtype=z.dtype) + .5) / quantiles
                q = torch.distributions.Normal(0., 1.).icdf(u)
                residuals = mean[:, None] + log_std.exp()[:, None] * q[None, :, None]
        k = residuals.shape[1]
        inputs = torch.cat((z[:, None].expand(-1, k, -1),
                            action[:, None].expand(-1, k, -1), residuals), -1)
        predicted = self.predictor(inputs.flatten(0, 1)).reshape(len(z), k, -1)
        if normalize:
            predicted = F.normalize(predicted, dim=-1)
        return predicted, predicted.new_full((len(z), k), 1 / k)

    def parameter_counts(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        stored = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "frozen": stored - trainable, "stored": stored,
                "ema_target": sum(p.numel() for p in self.target.parameters()) if self.uses_ema else 0}


def validate_config(cfg):
    return validate_matched_config(cfg)


def build_model(cfg):
    validate_config(cfg)
    return MPI3DGlobal(cfg)
