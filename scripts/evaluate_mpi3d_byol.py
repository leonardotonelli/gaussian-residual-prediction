"""Preprint implementation: selected components from the research codebase."""
import json
import torch
from iwm_replication.mpi3d_byol import MATCHED_VERSION, build_model, validate_config, verify_provenance, contract_hash, file_hash
from iwm_replication.seed_streams import preserve_rng


@preserve_rng()
def load_endpoint(path, device):
    sidecar = json.loads(path.with_suffix(".json").read_text())
    digest = file_hash(path)
    if sidecar["sha256"] != digest:
        raise ValueError("Checkpoint SHA256 disagrees with training sidecar")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    validate_config(cfg)
    if (checkpoint["version"] != cfg["protocol"] or checkpoint["epoch"] != cfg["train"]["epochs"]
            or checkpoint["config_sha256"] != contract_hash(cfg)
            or sidecar["config_sha256"] != contract_hash(cfg)
            or sidecar["epoch"] != checkpoint["epoch"]):
        raise ValueError("Evaluation requires the prespecified, complete terminal checkpoint")
    expected_steps = cfg["train"]["epochs"] * cfg["train"]["transition_samples_per_epoch"] // cfg["data"]["batch_size"]
    counter = "optimizer_updates" if cfg["protocol"] == MATCHED_VERSION else "ema_updates"
    if checkpoint["global_step"] != expected_steps or int(checkpoint["model"][counter]) != expected_steps:
        raise ValueError("Incomplete training/update budget")
    if cfg["protocol"] == MATCHED_VERSION and cfg["role"].startswith("R") and int(checkpoint["model"]["ema_updates"]) != expected_steps:
        raise ValueError("Incomplete EMA budget")
    verify_provenance(checkpoint["source_sha256"])
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, cfg, digest
