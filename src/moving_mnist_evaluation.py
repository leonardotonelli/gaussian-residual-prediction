"""Frozen development probes and physical forecasts for the four trained roles.

All normalizers fit on train identities. Selection and reporting identities are
disjoint. Physical predictions use the actual target projector readout; online
representation probes are a separate track. No final-test access or BN updates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .moving_mnist import (GeneratorConfig, SourceState, random_stream,
                          render_centers, sample_future_velocities, trajectory_centers)
from .moving_mnist_data import content_hash, load_digits
from .moving_mnist_full_training import FULL_TRAINING_VERSION, SEEDED_FULL_TRAINING_VERSION
from .moving_mnist_models import ModelConfig, MovingMNISTReference
from .moving_mnist_shared import S0Config, MovingMNISTS0
from .moving_mnist_shared_variational import S1Config, MovingMNISTS1
from .moving_mnist_metrics import physical_scores, estimate_source_velocity
from .moving_mnist_stream import OnlineClips
from .seed_streams import SeedContext, preserve_rng, seeded_rng


EVALUATION_VERSION = "concept2-development-evaluation-v1"
SEEDED_EVALUATION_VERSION = "concept2-development-evaluation-structured-v2"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(model):
    """Include every parameter/buffer and nested extra state (SIGReg RNG)."""
    h = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            a = value.detach().cpu().contiguous().numpy()
            h.update(str((a.dtype, a.shape)).encode())
            h.update(a.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                h.update(key.encode())
                visit(value[key])
        else:
            h.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    visit(model.state_dict())
    return h.hexdigest()


def _evaluation_context(seed_context):
    if seed_context is not None and (seed_context.dataset != "moving_mnist"
                                     or seed_context.purpose != "evaluation"):
        raise ValueError("Use a fixed Moving-MNIST evaluation context, not a training replication")


@preserve_rng()
def load_frozen(checkpoint, *, expected_sha256, expected_step, role, repo, device, allow_partial_software_smoke=False):
    actual = file_hash(checkpoint)
    if actual != expected_sha256:
        raise ValueError("Checkpoint SHA256 mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract = state["contract"]
    metadata = json.loads((checkpoint.parent.parent / "config.json").read_text())
    if (metadata["contract_sha256"] != content_hash(contract)
            or content_hash(metadata["contract"]) != content_hash(contract)):
        raise ValueError("Checkpoint/training metadata contract mismatch")
    from .moving_mnist_checkpoint_policy import allow_partial_checkpoint
    partial_smoke = allow_partial_checkpoint(state, expected_step, role, allow_partial_software_smoke)
    if (state["version"] not in (FULL_TRAINING_VERSION, SEEDED_FULL_TRAINING_VERSION) or state["step"] != expected_step
            or state["model_config"]["role"] != role
            or (not partial_smoke and state["training_config"]["total_steps"] != expected_step)
            or state["next_sample_index"] != expected_step * state["training_config"]["batch_size"]
            or content_hash(state["model_config"]) != content_hash(contract["model"])
            or content_hash(state["training_config"]) != content_hash(contract["training"])):
        raise ValueError("Expected the specified completed full-training checkpoint")
    if state["version"] == SEEDED_FULL_TRAINING_VERSION:
        context = SeedContext.from_dict(state["seed_context"])
        if (context.dataset != "moving_mnist" or context.purpose == "evaluation"
                or contract.get("seed_context") != context.as_dict()
                or contract.get("version") != SEEDED_FULL_TRAINING_VERSION):
            raise ValueError("Checkpoint seed context contract mismatch")
    elif state.get("seed_context") is not None:
        raise ValueError("Legacy checkpoint cannot contain structured seed context")
    for relative, expected in contract["code_sha256"].items():
        if file_hash(repo / relative) != expected:
            raise ValueError(f"Training source has changed: {relative}")
    cfg_class, model_class = ((ModelConfig, MovingMNISTReference) if role.startswith("R") else
                              (S0Config, MovingMNISTS0) if role == "S0" else (S1Config, MovingMNISTS1))
    model = model_class(cfg_class(**state["model_config"]))
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    if "campaign_recipe" in contract or contract.get("purpose") == "five-seed-campaign-training":
        from .moving_mnist_campaign import validate_campaign_contract
        validate_campaign_contract(contract, model=model)
    return model, contract


def make_banks(data_dir, manifest, train_contract, sizes, seed, *, seed_context=None):
    _evaluation_context(seed_context)
    train = load_digits(data_dir, manifest, "train")
    development = load_digits(data_dir, manifest, "development")
    split = len(development[2]) // 2
    selection = tuple(x[:split] for x in development)
    report = tuple(x[split:] for x in development)
    if any(set(a[2]) & set(b[2]) for a, b in ((train, selection), (train, report), (selection, report))):
        raise ValueError("Probe identity partitions overlap")
    result = {}
    for offset, (name, raw, partition) in enumerate((
            ("fit", train, "train"), ("selection", selection, "development"),
            ("report", report, "development"), ("queries", report, "development"))):
        result[name] = OnlineClips(*raw, split=partition, seed=seed + offset,
            generator_config=GeneratorConfig(**train_contract["generator"]),
            identity_hash=manifest["sha256"], size=sizes[name],
            seed_context=seed_context, seed_partition=name)
    for key in ("images_sha256", "labels_sha256", "image_ids_sha256", "identity_manifest_sha256"):
        if result["fit"].contract[key] != train_contract[key]:
            raise ValueError(f"Training data fingerprint mismatch: {key}")
    return result


@torch.inference_mode()
def extract_features(model, dataset, batch_size, device, *, clip="target"):
    if clip not in ("source", "target"):
        raise ValueError("Probe population must be source or target clips")
    banks = {name: [] for name in ("online_backbone", "online_projector", "target_projector")}
    records = [dataset.record(i) for i in range(len(dataset))]
    for start in range(0, len(dataset), batch_size):
        clips = torch.stack([dataset[i][clip] for i in range(start, min(start + batch_size, len(dataset)))])
        online = model.encode(clips.to(device), branch="online")
        target = model.encode(clips.to(device), branch="target") if getattr(model, "uses_ema", model.config.role.startswith("R")) else online
        for name, value in (("online_backbone", online["backbone"]),
                            ("online_projector", online["projector"]), ("target_projector", target["projector"])):
            banks[name].append(value.cpu().numpy().copy())
    return {**{k: np.concatenate(v) for k, v in banks.items()},
            "velocity": np.asarray([r["state"]["velocity"] if clip == "source" else r["future_velocity"] for r in records]),
            "source_velocity": np.asarray([r["state"]["velocity"] for r in records]),
            "digit": np.asarray([r["state"]["digit"] for r in records]),
            "records_sha256": content_hash({"records": records})}


def standardizer(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Need at least two finite feature rows")
    mean, std = x.mean(0), x.std(0)
    active = std > 1e-8
    return mean, np.where(active, std, 1.), active


def transform(x, head):
    from .campaign_evaluation import feature_transform
    x = feature_transform(x, str(head.get("feature_transform", "raw")))
    return (np.asarray(x, dtype=np.float64) - head["mean"]) / head["scale"] * head["active"]


def predict_velocity(x, head):
    return transform(x, head) @ head["weight"] + head["bias"]


def fit_ridge(x, y, selection_x, selection_y, alphas):
    mean, scale, active = standardizer(x)
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Ridge penalties must be finite and positive")
    head = {"mean": mean, "scale": scale, "active": active, "bias": y.mean(0)}
    z = transform(x, head)
    values, vectors = np.linalg.eigh(z.T @ z / len(z))
    values = np.maximum(values, 0)
    rhs = vectors.T @ (z.T @ (y - head["bias"]) / len(z))
    candidates = []
    for alpha in alphas:
        weight = vectors @ (rhs / (values[:, None] + alpha))
        mse = float(np.mean((transform(selection_x, head) @ weight + head["bias"] - selection_y) ** 2))
        candidates.append((mse, alpha, weight))
    mse, alpha, weight = min(candidates, key=lambda c: c[0])
    head["weight"] = weight
    norms = np.linalg.norm(z, axis=1)
    diagnostics = {"alpha": float(alpha), "selection_mse": mse,
                   "candidate_selection_mse": [{"alpha": float(a), "mse": v} for v, a, _ in candidates],
                   "regularized_condition": float((values[-1] + alpha) / (values[0] + alpha)),
                   "standardized_operator_norm": float(np.linalg.norm(weight, ord=2)),
                   "raw_operator_norm": float(np.linalg.norm(weight * (active / scale)[:, None], ord=2)),
                   "train_standardized_norm_p99": float(np.quantile(norms, .99)),
                   "constant_features": int((~active).sum())}
    return head, diagnostics


def regression_metrics(predicted, truth):
    residual = ((predicted - truth) ** 2).sum(0)
    total = ((truth - truth.mean(0)) ** 2).sum(0)
    r2 = [float(1 - e / t) if t > 1e-12 else None for e, t in zip(residual, total)]
    return {"rmse_xy": np.sqrt(residual / len(truth)).tolist(),
            "vector_rmse": float(np.sqrt(residual.sum() / len(truth))), "r2_xy": r2,
            "r2_mean": float(np.mean(r2)) if None not in r2 else None}


def motion_strata(source_velocity):
    norm = np.linalg.norm(source_velocity, axis=-1)
    return {"all": np.ones(len(norm), dtype=bool), "speed_lt_1": norm < 1,
            "speed_1_to_2": (norm >= 1) & (norm < 2), "speed_ge_2": norm >= 2}


def readout_report(predicted, truth, source_velocity, generator_config):
    # Setting A, std convention: E_axis Var(innovation | axis, source).
    if generator_config.setting != "A" or generator_config.gaussian_parameter != "std":
        raise ValueError("This evaluation protocol is defined for Setting A/std")
    variance = .5 * np.sum((generator_config.noise_factor * source_velocity) ** 2, axis=1)
    result = {}
    for name, mask in motion_strata(source_velocity).items():
        result[name] = {"queries": int(mask.sum())}
        if mask.any():
            error = np.sum((predicted[mask] - truth[mask]) ** 2, axis=1).mean()
            scale = float(variance[mask].mean())
            result[name].update(regression_metrics(predicted[mask], truth[mask]),
                                mean_innovation_variance=scale,
                                squared_error_over_innovation_variance=float(error / scale) if scale > 0 else None)
    return result


@preserve_rng()
def fit_digit_probe(x, labels, selection_x, selection_labels, report_x, report_labels, *, config, seed, device,
                    seed_context=None, probe_name="digit", restart=0):
    _evaluation_context(seed_context)
    mean, scale, active = standardizer(x)
    normalization = {"mean": mean, "scale": scale, "active": active}
    arrays = [torch.as_tensor(transform(a, normalization), dtype=torch.float32, device=device)
              for a in (x, selection_x, report_x)]
    targets = [torch.as_tensor(a, dtype=torch.long, device=device)
               for a in (labels, selection_labels, report_labels)]
    initialization_seed = (seed if seed_context is None else
                           seed_context.seed("probe-init", module=probe_name, restart=restart))
    with seeded_rng(initialization_seed):
        classifier = nn.Linear(x.shape[1], 10).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    rng = (torch.Generator().manual_seed(seed) if seed_context is None else
           seed_context.torch_generator("probe-order", module=probe_name, restart=restart))
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, config["epochs"] + 1):
        order = torch.randperm(len(x), generator=rng)
        for idx in order.split(config["batch_size"]):
            idx = idx.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(classifier(arrays[0][idx]), targets[0][idx])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            value = nn.functional.cross_entropy(classifier(arrays[1]), targets[1]).item()
        if not np.isfinite(value):
            raise RuntimeError("Nonfinite digit probe")
        history.append(value)
        # Always retain the lowest selection CE; patience uses a material decrease.
        meaningful = value < best - config["min_delta"]
        if value < best:
            best, best_epoch = value, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
        stale = 0 if meaningful else stale + 1
        if stale >= config["patience"]:
            break
    classifier.load_state_dict(best_state)
    with torch.no_grad():
        predictions = classifier(arrays[2]).argmax(-1).cpu().numpy()
    return {"accuracy": float(np.mean(predictions == report_labels)), "selected_epoch": best_epoch,
            "epochs_run": epoch, "selection_cross_entropy": best, "selection_curve": history}, \
           {**normalization, **{k: v.numpy() for k, v in best_state.items()}}


@preserve_rng()
@torch.inference_mode()
def forecast_bank(model, dataset, head, *, samples, futures, seed, device, batch_size, train_norm_p99,
                  seed_context=None):
    _evaluation_context(seed_context)
    if samples < 2 or futures < 2:
        raise ValueError("Independent oracle and truth banks require at least two draws")
    saved = {name: [] for name in ("truth", "axis", "source_velocity", "pixel_velocity", "query_records")}
    forecasts, support = {}, []
    partition = getattr(dataset, "seed_partition", dataset.split)
    forecast_rng = (None if seed_context is None else
                    seed_context.torch_generator("evaluation-forecast", device=device, partition=partition))
    variational = getattr(model, "is_stochastic", model.config.role in ("R1", "S1"))
    for q in range(len(dataset)):
        record = dataset.record(q)
        state = SourceState(**record["state"])
        source = dataset[q]["source"].unsqueeze(0).to(device)
        truth_rng = (random_stream(seed, q, 51) if seed_context is None else
                     seed_context.numpy_rng("evaluation-truth", partition=partition, sample_index=q))
        oracle_rng = (random_stream(seed, q, 52) if seed_context is None else
                      seed_context.numpy_rng("evaluation-oracle", partition=partition, sample_index=q))
        truth, axes = sample_future_velocities(state, futures, truth_rng, dataset.config)
        oracle, _ = sample_future_velocities(state, samples, oracle_rng, dataset.config)
        rng = (torch.Generator(device=device).manual_seed(seed + 10000 + q)
               if seed_context is None else forecast_rng)
        kwargs = {"num_samples": samples, "generator": rng} if variational else {}
        latent = model.forecast(source, **kwargs)["prediction"][0].cpu().numpy()
        pixel = estimate_source_velocity(source)[0].cpu().numpy()
        predictions = {"model_prior": predict_velocity(latent, head),
                       "pixel_constant_velocity": pixel[None], "independent_privileged_oracle": oracle,
                       "target_space_persistence": predict_velocity(model.encode(source, branch="target")["projector"].cpu().numpy(), head)}
        if variational:
            fixed = model.forecast(source, fixed_residual=True)["prediction"][0].cpu().numpy()
            predictions["model_fixed_prior_mean"] = predict_velocity(fixed, head)
        decoded = []
        image = dataset.images[record["image_index"]]
        for start in range(0, samples, batch_size):
            clips = torch.stack([render_centers(image, trajectory_centers(state, v)[3:6], dataset.config).unsqueeze(0)
                                 for v in oracle[start:start + batch_size]])
            features = model.encode(clips.to(device), branch="target")["projector"].cpu().numpy()
            decoded.append(predict_velocity(features, head))
        predictions["encoded_decoded_oracle"] = np.concatenate(decoded)
        norms = np.linalg.norm(transform(latent, head), axis=1)
        support.append({"mean_standardized_norm": float(norms.mean()),
                        "fraction_above_train_p99": float(np.mean(norms > train_norm_p99))})
        for name, value in predictions.items():
            if not np.isfinite(value).all():
                raise RuntimeError(f"Nonfinite physical predictions: {name}, query {q}")
            forecasts.setdefault(name, []).append(value)
        for key, value in (("truth", truth), ("axis", axes), ("source_velocity", state.velocity),
                           ("pixel_velocity", pixel), ("query_records", record)):
            saved[key].append(value)
        if q == 0 or (q + 1) % 64 == 0 or q + 1 == len(dataset):
            print(f"{model.config.role} physical queries {q + 1}/{len(dataset)}", flush=True)
    records = saved.pop("query_records")
    arrays = {k: np.asarray(v) for k, v in saved.items()}
    arrays.update({f"prediction_{k}": np.stack(v) for k, v in forecasts.items()})
    summaries, per_query = {}, {}
    truth_tensor = torch.from_numpy(arrays["truth"])
    for name in forecasts:
        values = arrays[f"prediction_{name}"]
        # Bound memory across queries as well as sample dimensions.
        chunks = [physical_scores(torch.from_numpy(values[a:a + 16]), truth_tensor[a:a + 16])
                  for a in range(0, len(values), 16)]
        score = {key: torch.cat([c[key] for c in chunks]).numpy() for key in chunks[0]}
        per_query[name] = {k: v.tolist() for k, v in score.items() if k != "nearest_squared_per_future"}
        aggregate = {}
        for stratum, mask in motion_strata(arrays["source_velocity"]).items():
            row = {"queries": int(mask.sum())}
            if mask.any():
                row.update({k: float(v[mask].mean()) for k, v in score.items() if k != "nearest_squared_per_future"})
                for axis, label in ((0, "x_changed"), (1, "y_changed")):
                    # Average futures within source, then give sources equal weight.
                    axis_means = [float(nearest[changed == axis].mean()) for nearest, changed in
                                  zip(score["nearest_squared_per_future"][mask], arrays["axis"][mask]) if np.any(changed == axis)]
                    row[f"coverage_squared_{label}"] = float(np.mean(axis_means)) if axis_means else None
                    row[f"sources_with_{label}"] = len(axis_means)
            aggregate[stratum] = row
        summaries[name] = {"samples": values.shape[1], "strata": aggregate,
                           "decoded_min_xy": values.min(axis=(0, 1)).tolist(),
                           "decoded_max_xy": values.max(axis=(0, 1)).tolist()}
    distortion = np.sum((arrays["prediction_encoded_decoded_oracle"] - arrays["prediction_independent_privileged_oracle"]) ** 2, axis=-1)
    diagnostics = {"pixel_source_velocity": regression_metrics(arrays["pixel_velocity"], arrays["source_velocity"]),
                   "oracle_decode_vector_rmse": float(np.sqrt(distortion.mean())),
                   "oracle_energy_score_distortion": summaries["encoded_decoded_oracle"]["strata"]["all"]["energy_score"] - summaries["independent_privileged_oracle"]["strata"]["all"]["energy_score"],
                   "predicted_feature_support": support}
    if variational:
        diagnostics["delta_fixed_es"] = summaries["model_fixed_prior_mean"]["strata"]["all"]["energy_score"] - summaries["model_prior"]["strata"]["all"]["energy_score"]
    return summaries, diagnostics, arrays, {"scores": per_query, "records": records}
