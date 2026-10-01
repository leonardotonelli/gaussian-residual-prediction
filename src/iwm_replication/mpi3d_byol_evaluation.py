"""Preprint implementation: selected components from the research codebase."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from .data import build_mpi3d_repeated_future_dataloader
from .mpi3d_probes import MPI3D_PROBE_LABEL_KEYS, fit_feature_standardizer, apply_feature_standardizer, normalize_mpi3d_positions, fit_mpi3d_affine_probe, evaluate_mpi3d_affine_probe
from .seed_streams import SeedContext, preserve_rng


def evaluation_seed_context(cfg):
    """Return a replication-independent evaluation namespace only on explicit opt-in."""
    if "evaluation_seed_streams" in cfg:
        context = SeedContext.from_dict(cfg["evaluation_seed_streams"])
    elif "seed_streams" in cfg:
        training = SeedContext.from_dict(cfg["seed_streams"])
        if training.dataset != "mpi3d" or training.purpose == "evaluation":
            raise ValueError("Derived MPI3D evaluation requires an MPI3D training SeedContext")
        context = SeedContext(campaign=training.campaign, dataset="mpi3d", purpose="evaluation", replication=None)
    else:
        return None
    if context.dataset != "mpi3d" or context.purpose != "evaluation":
        raise ValueError("MPI3D evaluation requires a fixed evaluation SeedContext")
    return context


def feature_diagnostics(features):
    # Full covariance is unnecessary for the 16k-dimensional flattened view.
    x = features.float()
    std = x.std(0, unbiased=False)
    report = {"num_images": len(x), "dimension": x.shape[1],
              "mean_coordinate_std": float(std.mean()),
              "mean_coordinate_variance": float(std.square().mean()),
              "negligible_std_coordinates": int((std < 1e-6).sum())}
    if x.shape[1] <= 1024:
        singular = torch.linalg.svdvals((x - x.mean(0)).double())
        mass = singular / singular.sum().clamp_min(1e-30)
        report["effective_rank"] = float((-(mass * mass.clamp_min(1e-30).log()).sum()).exp())
    return report


def fit_probes(train, validation, cfg, device):
    settings = cfg["probe"]
    seed_context = evaluation_seed_context(cfg)
    state, report = {}, {}
    for view, raw in train["features"].items():
        stats = fit_feature_standardizer(raw)
        x = apply_feature_standardizer(raw, stats)
        v = apply_feature_standardizer(validation["features"][view], stats)
        heads, details = {}, {}
        for kind in MPI3D_PROBE_LABEL_KEYS:
            yt, yv = train["labels"][kind], validation["labels"][kind]
            if kind == "position":
                yt, yv = normalize_mpi3d_positions(yt), normalize_mpi3d_positions(yv)
            head, metadata = fit_mpi3d_affine_probe(x, yt, v, yv, kind=kind, probe_seed=cfg["seed"],
                lambda_grid=settings["lambda_grid"], max_epochs=settings["maximum_epochs"], device=device,
                batch_size=settings["batch_size"], learning_rate=settings["learning_rate"],
                **({} if seed_context is None else {"seed_context": seed_context, "representation": view}))
            heads[kind] = {k: t.cpu() for k, t in head.state_dict().items()}
            details[kind] = metadata
            print(f"probe_selected view={view} factor={kind} selection={metadata['selected']}", flush=True)
        state[view] = {"standardizer": stats, "heads": heads}
        if seed_context is not None:
            state[view]["seed_streams"] = seed_context.as_dict()
        report[view] = {"training_features": feature_diagnostics(raw), "probes": details}
    return state, report


def evaluate_probes(bank, state, device, *, seed_context=None):
    result = {}
    for view, raw in bank["features"].items():
        view_context = seed_context
        if "seed_streams" in state[view]:
            saved_context = SeedContext.from_dict(state[view]["seed_streams"])
            if view_context is not None and view_context != saved_context:
                raise ValueError("Probe evaluation seed context disagrees with fitted head")
            view_context = saved_context
        x = apply_feature_standardizer(raw, state[view]["standardizer"])
        metrics = {}
        for kind, weights in state[view]["heads"].items():
            if view_context is None:
                head = nn.Linear(x.shape[1], weights["weight"].shape[0]).to(device)
            else:
                if view_context.dataset != "mpi3d" or view_context.purpose != "evaluation":
                    raise ValueError("MPI3D evaluation requires a fixed evaluation SeedContext")
                with preserve_rng():
                    head = nn.Linear(x.shape[1], weights["weight"].shape[0]).to(device)
            head.load_state_dict(weights)
            labels = bank["labels"][kind]
            if kind == "position":
                labels = normalize_mpi3d_positions(labels)
            metrics[kind] = evaluate_mpi3d_affine_probe(x, labels, head, kind, device)
        result[view] = {"features": feature_diagnostics(raw), "probes": metrics}
    return result


def distribution_metrics(predictions, weights, targets, truth_weights, *, gap_epsilon):
    """Exact score of a finite weighted distribution, NOT an IID-sample U statistic.

    All target atoms are enumerated. Gaussian midpoint quantiles define a finite
    approximation. Nondegenerate balanced targets have singleton ES >= 0.5 and
    oracle ES = 0.25 after normalization by the target RMS gap.
    """
    p, w, y, q = (v.double() for v in (predictions, weights, targets, truth_weights))
    if p.ndim != 3 or y.shape != (len(p), 2, p.shape[-1]) or w.shape != p.shape[:2] or q.shape != (len(p), 2):
        raise ValueError("Expected (B,K,D) predictions and (B,2,D) targets with matching weights")
    if not all(torch.isfinite(v).all() for v in (p, w, y, q)):
        raise ValueError("Nonfinite forecast/target")
    for mass in (w, q):
        if (mass < 0).any() or not torch.allclose(mass.sum(1), torch.ones(len(p), device=p.device, dtype=p.dtype)):
            raise ValueError("Probability mass must be nonnegative and sum to one")
    squared = (p[:, :, None] - y[:, None]).square().mean(-1)
    distances = squared.sqrt()
    self_dist = (p[:, :, None] - p[:, None]).square().mean(-1).sqrt()
    es = (distances * w[:, :, None] * q[:, None]).sum((1, 2)) - 0.5 * (self_dist * w[:, :, None] * w[:, None]).sum((1, 2))
    coverage = squared.masked_fill(w[:, :, None] == 0, float("inf")).amin(1)
    precision = (squared.masked_fill(q[:, None] == 0, float("inf")).amin(2) * w).sum(1)
    gap = (y[:, 0] - y[:, 1]).square().mean(-1).sqrt()
    valid = gap > gap_epsilon
    # Mark undefined values rather than silently creating favorable scores by clamping.
    norm_es = torch.full_like(es, float("nan"))
    norm_coverage = norm_es.clone()
    norm_precision = norm_es.clone()
    norm_es[valid] = es[valid] / gap[valid]
    norm_coverage[valid] = (coverage * q).sum(1)[valid] / gap[valid].square()
    norm_precision[valid] = precision[valid] / gap[valid].square()
    return {"energy_score_rms": es, "coverage_mse": (coverage * q).sum(1),
            "coverage_failure_mse": coverage[:, 0], "coverage_success_mse": coverage[:, 1],
            "precision_mse": precision, "true_gap_rms": gap, "valid_gap": valid,
            "normalized_energy_score": norm_es, "normalized_coverage": norm_coverage,
            "normalized_precision": norm_precision}


def summarize_records(records):
    result = {}
    for key, values in records.items():
        if key.endswith("valid_gap"):
            result[key] = {"valid": int(values.sum()), "invalid": int((~values).sum())}
        else:
            good = np.isfinite(values)
            result[key] = {"mean": float(values.mean()) if good.all() else None,
                           "mean_over_defined_queries": float(values[good].mean()) if good.any() else None,
                           "defined_queries": int(good.sum()), "total_queries": len(values)}
    return result


def forecast_loader(cfg, split):
    # D is the degenerate success-only law over the same two enumerated atoms.
    data_cfg = deepcopy(cfg)
    data_cfg["data"]["condition"] = "S"
    loader = build_mpi3d_repeated_future_dataloader(data_cfg, split)
    context = evaluation_seed_context(cfg)
    if context is not None:
        loader.generator = context.torch_generator("loader", module="forecast-bank", partition=split)
    return loader
