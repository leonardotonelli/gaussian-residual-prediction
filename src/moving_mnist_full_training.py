"""All four Moving-MNIST roles: fixed-budget training and exact resume, no evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import tempfile

import torch

from .moving_mnist_models import MODEL_VERSION, ModelConfig, MovingMNISTReference
from .moving_mnist_shared import SHARED_MODEL_VERSION, MovingMNISTS0, S0Config
from .moving_mnist_shared_variational import S1_MODEL_VERSION, MovingMNISTS1, S1Config
from .seed_streams import SeedContext, seeded_rng
from .paired_initialization import initialize_paired_model


FULL_TRAINING_VERSION = "concept2-four-role-full-training-v1"
SEEDED_FULL_TRAINING_VERSION = "concept2-four-role-full-training-structured-v2"
ROLES = ("R0", "R1", "S0", "S1")


@dataclass(frozen=True)
class FullTrainingConfig:
    total_steps: int = 75000
    batch_size: int = 128
    seed: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    checkpoint_interval: int = 2500
    log_interval: int = 100
    timing_warmup_steps: int = 10

    def __post_init__(self):
        if self.total_steps < 1 or self.batch_size < 2 or self.seed < 0:
            raise ValueError("Positive training budget, batch size >=2 and nonnegative seed required")
        if min(self.checkpoint_interval, self.log_interval) < 1 or self.timing_warmup_steps < 0:
            raise ValueError("Invalid checkpoint/log/timing interval")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative")


def model_spec(role, model_options, training):
    if role not in ROLES:
        raise ValueError("Unknown Moving-MNIST role")
    if role.startswith("R"):
        return ModelConfig(role=role, **model_options), MODEL_VERSION
    cls, version = (S0Config, SHARED_MODEL_VERSION) if role == "S0" else (S1Config, S1_MODEL_VERSION)
    return cls(batch_size=training.batch_size, **model_options), version


def atomic_checkpoint(path, state):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(state, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class FullTrainer:
    def __init__(self, config, model_config, device, *, seed_context: SeedContext | None = None):
        self.config, self.device, self.seed_context = config, device, seed_context
        if hasattr(model_config, "batch_size") and model_config.batch_size != config.batch_size:
            raise ValueError("Shared model and trainer batch sizes must match")
        if seed_context is not None and (seed_context.dataset != "moving_mnist"
                                         or seed_context.purpose == "evaluation"):
            raise ValueError("FullTrainer requires a Moving-MNIST training seed context")
        self.version = FULL_TRAINING_VERSION if seed_context is None else SEEDED_FULL_TRAINING_VERSION
        if seed_context is None:
            torch.manual_seed(config.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(config.seed)
        self.role = model_config.role
        cls = MovingMNISTReference if self.role.startswith("R") else MovingMNISTS0 if self.role == "S0" else MovingMNISTS1
        self.initialization = None
        if seed_context is None:
            self.model = cls(model_config).to(device).train()
        else:
            with seeded_rng(seed_context.seed("initialization", module="construction")):
                self.model = cls(model_config).to(device).train()
                self.initialization = initialize_paired_model(self.model, seed_context)
        self.optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad),
                                          lr=config.learning_rate, weight_decay=config.weight_decay,
                                          betas=(.9, .999), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.total_steps, eta_min=0.0)
        self.residual_rng = (torch.Generator(device=device).manual_seed(config.seed + 1000)
                             if seed_context is None else seed_context.torch_generator("latent", device=device))
        # Stateful streams avoid repeated CPU reseeding (whose effective seed has
        # only 32 bits) and preserve paired source/target directions across roles.
        self.sigreg_generators = (None if seed_context is None or not self.role.startswith("S") else
                                  tuple(seed_context.torch_generator("sigreg", view=view)
                                        for view in ("source", "target")))
        self.step = 0

    def normalization_counts(self):
        """Observed and expected natural BN exposures; no recalibration passes."""
        result = {}
        for name, value in self.model.named_buffers():
            if name.endswith("num_batches_tracked"):
                multiplier = (2 if self.role == "R1" and name.startswith("online_encoder.") else
                              2 if self.role == "S0" and name.startswith("encoder.") else
                              3 if self.role == "S1" and name.startswith("encoder.") else 1)
                result[name] = {"observed": int(value), "expected": multiplier * self.step}
        return result

    def update(self, batch):
        cfg = self.config
        if self.step >= cfg.total_steps:
            raise ValueError("Training budget is complete")
        expected = torch.arange(self.step * cfg.batch_size, (self.step + 1) * cfg.batch_size)
        if not torch.equal(batch["sample_index"].cpu(), expected):
            raise ValueError("Wrong next sample addresses for this checkpoint")
        keys = ("source", "target", "surrogate") if self.role in ("R1", "S1") else ("source", "target")
        pixels = [batch[key].to(self.device, non_blocking=True) for key in keys]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        kwargs = {"generator": self.residual_rng} if self.role in ("R1", "S1") else {}
        if self.sigreg_generators is not None:
            kwargs["sigreg_generators"] = self.sigreg_generators
        out = self.model(*pixels, **kwargs)
        if not torch.isfinite(out["loss"]):
            raise RuntimeError(f"Nonfinite loss before update {self.step + 1}")
        out["loss"].backward()
        for name, p in self.model.named_parameters():
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                raise RuntimeError(f"Missing/nonfinite gradient: {name}")
        lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise RuntimeError("Nonfinite parameters after update; resume last completed checkpoint")
        if self.role.startswith("R"):
            self.model.update_target()
        self.scheduler.step()
        self.step += 1
        return {"step": self.step, "next_sample_index": self.step * cfg.batch_size,
                "learning_rate": lr,
                **{key: out[key].detach().item() if key in out else 0.0
                   for key in ("loss", "alignment", "kl", "sigreg")}}

    def save(self, path, contract):
        if (self.scheduler.last_epoch != self.step
                or (self.role.startswith("R") and int(self.model.ema_updates) != self.step)
                or any(row["observed"] != row["expected"] for row in self.normalization_counts().values())):
            raise ValueError("Incomplete training update or extra BN forwards; checkpoint counts mismatch")
        atomic_checkpoint(path, {
            "version": self.version, "contract": contract,
            **({"seed_context": self.seed_context.as_dict(), "initialization": self.initialization,
                "sigreg_rng": [g.get_state() for g in self.sigreg_generators or ()]}
               if self.seed_context is not None else {}),
            "training_config": asdict(self.config), "model_config": asdict(self.model.config),
            "step": self.step, "next_sample_index": self.step * self.config.batch_size,
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "residual_rng": self.residual_rng.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            "device_type": self.device.type})

    def load(self, path, contract):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (state.get("version") != self.version or state.get("contract") != contract
                or state.get("seed_context") != (None if self.seed_context is None else self.seed_context.as_dict())
                or state.get("initialization") != self.initialization
                or state.get("training_config") != asdict(self.config)
                or state.get("model_config") != asdict(self.model.config)
                or state.get("device_type") != self.device.type):
            raise ValueError("Resume contract mismatch (role/code/data/recipe/runtime); no smoke checkpoint resume")
        step = state["step"]
        if not 0 <= step <= self.config.total_steps or state["next_sample_index"] != step * self.config.batch_size:
            raise ValueError("Invalid checkpoint step/sample cursor")
        if state["scheduler"]["last_epoch"] != step:
            raise ValueError("Checkpoint scheduler count mismatch")
        if self.role.startswith("R") and int(state["model"]["ema_updates"]) != step:
            raise ValueError("Checkpoint EMA count mismatch")
        for name, value in state["model"].items():
            if name.endswith("num_batches_tracked"):
                multiplier = (2 if self.role == "R1" and name.startswith("online_encoder.") else
                              2 if self.role == "S0" and name.startswith("encoder.") else
                              3 if self.role == "S1" and name.startswith("encoder.") else 1)
                if int(value) != multiplier * step:
                    raise ValueError(f"Checkpoint BN count mismatch: {name}")
        if self.seed_context is not None:
            saved_sigreg = state.get("sigreg_rng", [])
            if len(saved_sigreg) != len(self.sigreg_generators or ()):
                raise ValueError("Checkpoint SIGReg stream count mismatch")
            for generator, rng_state in zip(self.sigreg_generators or (), saved_sigreg):
                generator.set_state(rng_state)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.residual_rng.set_state(state["residual_rng"])
        torch.set_rng_state(state["torch_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.step = step
