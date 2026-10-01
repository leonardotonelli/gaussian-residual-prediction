"""Train fixed-budget BYOL/AdaSSL-V baselines; no validation or test selection."""

import argparse
import json
from pathlib import Path
import platform
import subprocess

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from iwm_replication.data import build_iwm_dataset, build_mpi3d_training_sampler
from iwm_replication.distributed import initialize_distributed, destroy_distributed, barrier, mean_across_ranks
from iwm_replication.mpi3d_byol import (MATCHED_VERSION, MATCHED_ROLES,
    build_model, resolve_matched_config, validate_config, provenance,
    verify_provenance, contract_hash, file_hash)
from iwm_replication.optimization import IWMOptimizationSchedule, apply_optimization_values, build_adamw_optimizer
from iwm_replication.utils import load_yaml, save_yaml, save_json, seed_everything, tee_console_to_file
from iwm_replication.seed_streams import context_from_config, seeded_rng
from iwm_replication.paired_initialization import initialize_paired_model


def primary_action(context, action):
    """Propagate rank-zero filesystem failures before any worker can wait forever."""
    result = [None]
    if context.is_primary:
        try:
            result[0] = {"value": action()}
        except Exception as error:
            result[0] = {"error": f"{type(error).__name__}: {error}"}
    if context.is_distributed:
        dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise RuntimeError(result[0]["error"])
    return result[0]["value"]


def atomic_save(state, path):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def wrap_distributed_model(model, context, *, fixed_buckets=False):
    if not context.is_distributed:
        return model
    # DDP otherwise starts with one reduction bucket and rebuilds it after the
    # first backward pass. A resumed process repeats that first-pass layout,
    # changing NCCL's floating-point reduction order relative to uninterrupted
    # training. Unused-parameter traversal keeps the construction-time buckets
    # fixed. train_step still rejects every missing trainable gradient.
    return DistributedDataParallel(
        model,
        device_ids=[context.local_rank] if context.device.type == "cuda" else None,
        broadcast_buffers=False,
        find_unused_parameters=fixed_buckets,
    )


def train_step(model, wrapped, batch, optimizer, generator, device, decay):
    model.train()
    # These are the only fields available to the training objective.
    losses = wrapped(batch["x_source"].to(device), batch["y_target"].to(device),
                     batch["action"].to(device), generator=generator)
    if not all(torch.isfinite(v).all() for v in losses.values()):
        raise RuntimeError("Nonfinite training objective")
    optimizer.zero_grad(set_to_none=True)
    losses["loss"].backward()
    gradients = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                raise RuntimeError(f"Missing gradient: {name}")
            gradients.append(torch.isfinite(p.grad).all())
    # One CUDA synchronization rather than one per parameter on every step.
    if not torch.stack(gradients).all():
        raise RuntimeError("Nonfinite gradient")
    optimizer.step()
    if hasattr(model, "after_optimizer_step"):
        model.after_optimizer_step(decay)
    else:
        model.update_target(decay)
    return {k: float(v.detach()) for k, v in losses.items()}


def run(args, cfg, context):
    if getattr(args, "smoke_runtime_audit", False):
        from iwm_replication.mpi3d_smoke import enable_smoke_determinism
        enable_smoke_determinism(cfg)
    if cfg.get("protocol") == MATCHED_VERSION:
        validate_config(cfg)
        if context.world_size not in (1, 2, 4) or context.rank not in range(context.world_size):
            raise ValueError("Matched topology exceeds the fixed seed manifest rank domain")
    seed_context = context_from_config(cfg)
    if seed_context is None:
        return _run(args, cfg, context, None)
    if (seed_context.dataset != "mpi3d" or seed_context.purpose == "evaluation"
            or seed_context.replication != cfg["seed"]):
        raise ValueError("MPI3D training seed context must match the configured replication")
    # Single-GPU device resolution can leave CUDA lazy. Initialize before the
    # context snapshots/seeds generators, including the first training call.
    if context.device.type == "cuda":
        torch.cuda.init()
    # Global APIs (e.g. dropout) are isolated from loader, construction, latent
    # draws, and the calling process. Their rank-local state is checkpointed.
    with seeded_rng(seed_context.seed("dropout", rank=context.rank)):
        return _run(args, cfg, context, seed_context)


def _run(args, cfg, context, seed_context):
    if seed_context is None:
        seed_everything(cfg["seed"])
    device = context.device
    version = cfg["protocol"]
    batch_size = cfg["data"]["batch_size"]
    stop_after = getattr(args, "stop_after_epoch", None)
    if stop_after is not None and (type(stop_after) is not int or
            not 1 <= stop_after <= cfg["train"]["epochs"] or args.preflight or args.synthetic_preflight):
        raise ValueError("stop-after-epoch must be inside the unchanged training budget and cannot accompany preflight")
    if batch_size % context.world_size or batch_size // context.world_size < 2:
        raise ValueError("Global batch must divide over ranks with >=2 examples per rank")
    loader_generator = None if seed_context is None else seed_context.torch_generator("loader", rank=context.rank)
    if args.synthetic_preflight:
        n = batch_size // context.world_size
        loader = [{"x_source": torch.rand(n, 3, 64, 64, generator=loader_generator), "y_target": torch.rand(n, 3, 64, 64, generator=loader_generator),
                   "action": torch.tensor([[1., 0.], [-1., 0.]]).repeat(n // 2, 1)} for _ in range(2)]
    else:
        def check_data():
            if file_hash(cfg["data"]["images_path"]) != cfg["data"]["images_sha256"]:
                raise ValueError("MPI3D images checksum mismatch")
            return True
        primary_action(context, check_data)
        dataset = build_iwm_dataset(cfg, split="train")
        sampler = build_mpi3d_training_sampler(cfg, dataset, num_replicas=context.world_size, rank=context.rank)
        loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size // context.world_size,
                            num_workers=cfg["data"]["num_workers"], pin_memory=device.type == "cuda",
                            **({} if seed_context is None else {"generator": loader_generator}))
    initialization = None
    if seed_context is None:
        model = build_model(cfg).to(device)
    else:
        with seeded_rng(seed_context.seed("initialization", module="construction")):
            model = build_model(cfg).to(device)
        initialization = initialize_paired_model(model, seed_context)
    schedule = IWMOptimizationSchedule.from_config(cfg, steps_per_epoch=len(loader))
    # Build groups before conversion so BN affine weights retain zero decay.
    optimizer = build_adamw_optimizer([model], cfg["optim"]["lr"],
                                      cfg["optim"]["weight_decay_start"], cfg["optim"]["betas"])
    if context.is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    wrapped = wrap_distributed_model(model, context, fixed_buckets=version == MATCHED_VERSION)
    generator = (torch.Generator(device=device).manual_seed(cfg["seed"] + 10000 + context.rank)
                 if seed_context is None else seed_context.torch_generator("latent", device=device, rank=context.rank))
    if args.preflight or args.synthetic_preflight:
        for step, batch in enumerate(loader):
            values = schedule.values_at(step)
            apply_optimization_values(optimizer, values)
            metrics = train_step(model, wrapped, batch, optimizer, generator, device, values.ema_decay)
            if step == 1:
                break
        model.eval()
        predictions, weights = model.forecast(batch["x_source"][:2].to(device), batch["action"][:2].to(device))
        if not torch.isfinite(predictions).all() or not torch.allclose(weights.sum(1), torch.ones(2, device=device)):
            raise RuntimeError("Invalid prior forecast")
        primary_action(context, lambda: print(json.dumps({"preflight": "passed", "synthetic": args.synthetic_preflight,
            "world_size": context.world_size, "variant": cfg["variant"], "losses": metrics,
            **({} if version != MATCHED_VERSION else {"role": cfg["role"],
                "purpose": cfg["stage"], "replication": cfg["seed"]})}), flush=True))
        return

    run_path = Path(cfg["output_dir"]) / cfg["run_name"]
    hashes = primary_action(context, lambda: provenance(cfg))
    start_epoch, global_step, history = 1, 0, []
    if args.resume:
        checkpoint = torch.load(run_path / "checkpoints/checkpoint_latest.pt", map_location="cpu", weights_only=True)
        if (checkpoint["version"] != version or checkpoint["config"] != cfg
                or checkpoint["world_size"] != context.world_size or checkpoint["source_sha256"] != hashes
                or checkpoint["config_sha256"] != contract_hash(cfg)):
            raise ValueError("Resume requires identical code, config, data manifests and world size")
        if seed_context is not None and (
                checkpoint.get("seed_streams") != seed_context.as_dict()
                or checkpoint.get("seed_context_sha256") != seed_context.sha256):
            raise ValueError("Resume requires identical versioned seed context")
        if version == MATCHED_VERSION and (
                checkpoint.get("role") != cfg["role"]
                or checkpoint.get("campaign_manifest") != cfg["campaign_manifest"]
                or checkpoint.get("initialization") != initialization):
            raise ValueError("Matched resume role/manifest/initialization contract mismatch")
        verify_provenance(checkpoint["source_sha256"])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, global_step = checkpoint["epoch"] + 1, checkpoint["global_step"]
        updates = model.optimizer_updates if hasattr(model, "optimizer_updates") else model.ema_updates
        if (global_step != checkpoint["epoch"] * len(loader) or int(updates) != global_step
                or (getattr(model, "uses_ema", True) and int(model.ema_updates) != global_step)):
            raise ValueError("Checkpoint epoch/step/EMA counters disagree")
        if version == MATCHED_VERSION:
            for name, value in checkpoint["model"].items():
                if name.endswith("num_batches_tracked"):
                    multiplier = 2 if name.startswith("online.") and cfg["role"] != "R0" else 1
                    if int(value) != multiplier * global_step:
                        raise ValueError("Checkpoint BatchNorm exposure counter mismatch")
        history = checkpoint["history"]
        rng = checkpoint["rng_by_rank"][context.rank]
        generator.set_state(rng["residual"])
        if seed_context is not None:
            loader_generator.set_state(rng["loader"])
        torch.set_rng_state(rng["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"], device)
    else:
        def claim():
            run_path.mkdir(parents=True, exist_ok=False)
            for name in ("checkpoints", "metrics", "logs"):
                (run_path / name).mkdir()
            save_yaml(cfg, run_path / "config.yaml")
            try:
                revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                revision = "unavailable"
            save_json({"version": version, "config_sha256": contract_hash(cfg), "source_sha256": hashes,
                "revision": revision, "python": platform.python_version(), "torch": str(torch.__version__),
                "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                "world_size": context.world_size, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "outcome_supervision": False, "validation_or_test_loaded_during_training": False,
                "checkpoint_selection": "prespecified terminal epoch",
                **({} if version != MATCHED_VERSION else {
                    "role": model.role, "uses_ema": model.uses_ema,
                    "parameter_counts": model.parameter_counts(),
                    "campaign_manifest": cfg["campaign_manifest"],
                    "target_space": "ema_projector" if model.uses_ema else "shared_projector",
                    "batchnorm_passes": {"online": 2 if model.role != "R0" else 1,
                                         "ema": 1 if model.uses_ema else 0},
                    "sigreg_streams": [seed_context.key("sigreg", view=view).record()
                                       for view in ("source", "target")] if not model.uses_ema else [],
                }),
                **({} if seed_context is None else {
                    "seed_streams": seed_context.as_dict(), "seed_context_sha256": seed_context.sha256,
                    "initialization": initialization,
                    "model_noise_policy": "checkpointed_rank_local_streams_fixed_world_size",
                    "latent_stream": seed_context.key("latent", rank=context.rank).record(),
                    "dropout_stream": seed_context.key("dropout", rank=context.rank).record(),
                    "loader_policy": "epoch_and_rank_addressed_generator",
                })}, run_path / "metadata.json")
        primary_action(context, claim)
    if context.is_primary:
        tee_console_to_file(run_path / "logs/training.log")
    if start_epoch > cfg["train"]["epochs"]:
        def finish_interrupted_publication():
            # A timeout may occur after latest is saved but before its endpoint
            # or sidecar is published. Resume must recover this boundary too.
            endpoint = run_path / f"checkpoints/checkpoint_epoch_{checkpoint['epoch']:04d}.pt"
            sidecar = endpoint.with_suffix(".json")
            try:
                metadata = json.loads(sidecar.read_text())
                valid = (endpoint.exists() and metadata["sha256"] == file_hash(endpoint)
                         and metadata["config_sha256"] == contract_hash(cfg)
                         and metadata["epoch"] == checkpoint["epoch"])
            except (OSError, ValueError, KeyError):
                valid = False
            if not valid:
                atomic_save(checkpoint, endpoint)
                save_json({"epoch": checkpoint["epoch"], "sha256": file_hash(endpoint),
                           "config_sha256": contract_hash(cfg)}, sidecar)
            save_json({"status": "training-complete", "history": history}, run_path / "metrics/training.json")
            print(f"training_complete={run_path}", flush=True)
        primary_action(context, finish_interrupted_publication)
        barrier(context)
        return
    if stop_after is not None and stop_after < start_epoch:
        raise ValueError("stop-after-epoch precedes the next resumable epoch")
    runtime = None
    if getattr(args, "smoke_runtime_audit", False):
        from iwm_replication.mpi3d_smoke import SmokeRuntime
        runtime = SmokeRuntime(device, context.rank, context.world_size, batch_size)
    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        sampler.set_epoch(epoch - 1)
        if seed_context is not None:
            loader_generator.manual_seed(seed_context.seed("loader", epoch=epoch - 1, rank=context.rank))
        metric_names = (("loss", "alignment", "kl", "sigreg", "source_sigreg", "target_sigreg")
                        if version == MATCHED_VERSION else ("loss", "alignment", "kl"))
        totals = {k: 0. for k in metric_names}
        for batch in loader:
            values = schedule.values_at(global_step)
            apply_optimization_values(optimizer, values)
            started = runtime.start() if runtime is not None else None
            metrics = train_step(model, wrapped, batch, optimizer, generator, device, values.ema_decay)
            if runtime is not None:
                runtime.finish(started, epoch=epoch, step=global_step + 1, metrics=metrics)
            for key, value in metrics.items():
                totals[key] += value
            global_step += 1
            if context.is_primary and global_step % cfg["train"]["log_every"] == 0:
                print(json.dumps({"epoch": epoch, "step": global_step, "learning_rate": values.learning_rate,
                                  **metrics}), flush=True)
        averaged = {k: mean_across_ranks(v / len(loader), context) for k, v in totals.items()}
        history.append({"epoch": epoch, "step": global_step, **averaged})
        rng = {"cpu": torch.get_rng_state(), "residual": generator.get_state(),
               "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
        if seed_context is not None:
            rng["loader"] = loader_generator.get_state()
        all_rng = [None] * context.world_size
        if context.is_distributed:
            dist.all_gather_object(all_rng, rng)
        else:
            all_rng[0] = rng
        all_runtime = None
        if runtime is not None:
            all_runtime = [None] * context.world_size
            if context.is_distributed:
                dist.all_gather_object(all_runtime, runtime.report())
            else:
                all_runtime[0] = runtime.report()
        def save_epoch():
            state = {"version": version, "config": cfg, "config_sha256": contract_hash(cfg), "source_sha256": hashes,
                "epoch": epoch, "global_step": global_step, "world_size": context.world_size,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng_by_rank": all_rng,
                "history": history}
            if version == MATCHED_VERSION:
                state.update(role=model.role, campaign_manifest=cfg["campaign_manifest"],
                             initialization=initialization)
            if seed_context is not None:
                state.update(seed_streams=seed_context.as_dict(), seed_context_sha256=seed_context.sha256,
                             sampler_metadata=sampler.metadata_for_epoch(epoch - 1))
            latest = run_path / "checkpoints/checkpoint_latest.pt"
            atomic_save(state, latest)
            if epoch % cfg["train"]["checkpoint_every"] == 0 or epoch == cfg["train"]["epochs"]:
                endpoint = latest.parent / f"checkpoint_epoch_{epoch:04d}.pt"
                atomic_save(state, endpoint)
                save_json({"epoch": epoch, "sha256": file_hash(endpoint), "config_sha256": contract_hash(cfg)},
                          endpoint.with_suffix(".json"))
            status = ("training-complete" if epoch == cfg["train"]["epochs"] else
                      "training-paused" if stop_after == epoch else "training")
            save_json({"status": status, "history": history}, run_path / "metrics/training.json")
            if all_runtime is not None:
                runtime_path = run_path / "metrics" / f"smoke_runtime_{start_epoch:04d}_{epoch:04d}.json"
                with runtime_path.open("x") as handle:
                    json.dump({"config_sha256": contract_hash(cfg), "source_sha256": hashes,
                               "start_epoch": start_epoch, "end_epoch": epoch,
                               "ranks": all_runtime}, handle, indent=2, sort_keys=True, allow_nan=False)
                    handle.write("\n")
            print(json.dumps(history[-1]), flush=True)
        primary_action(context, save_epoch)
        if stop_after == epoch:
            break
    barrier(context)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/campaigns/five_seed_v1/mpi3d.yaml")
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--role", choices=MATCHED_ROLES, help="Matched four-role campaign")
    identity.add_argument("--array-index", type=int, help="Matched role-major run mapping")
    parser.add_argument("--replication", type=int, help="Matched main 1..5; development/smoke 0")
    parser.add_argument("--purpose", choices=("main-training", "development-training", "software-smoke"),
                        default="main-training")
    parser.add_argument("--run-dir", type=Path, help="Explicit output directory for the portable preprint runner")
    parser.add_argument("--resume", action="store_true", help="Resume latest completed epoch, preserving all contracts")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Pause after this absolute epoch; keep the full budget/schedule for exact resume")
    parser.add_argument("--smoke-runtime-audit", action="store_true",
                        help="Software-smoke only: strict determinism and measured step/memory sidecars")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--synthetic-preflight", action="store_true")
    return parser


def config_from_arguments(args):
    base = load_yaml(args.config)
    if args.array_index is not None:
        from iwm_replication.five_seed_campaign import run_at_index
        if args.replication is not None:
            raise ValueError("Array index already determines its replication")
        row = run_at_index(args.array_index, dataset="mpi3d", purpose=args.purpose)
        role, replication = row["role"], row["replication"]
    else:
        role, replication = args.role, args.replication
    cfg = resolve_matched_config(base, role=role, replication=replication, purpose=args.purpose)
    if args.synthetic_preflight:
        if cfg["protocol"] == MATCHED_VERSION and args.purpose != "software-smoke":
            raise ValueError("Matched synthetic preflight requires the separate software-smoke namespace")
        cfg["device"] = "cpu"
        cfg["data"]["batch_size"] = 4
        cfg["train"].update(epochs=2, warmup_epochs=0, transition_samples_per_epoch=8)
        cfg["model"].update(vit_dim=16, vit_depth=1, vit_heads=2, projector_dim=8, head_hidden_dim=16)
        torch.set_num_threads(1)
    if getattr(args, "run_dir", None) is not None:
        location = args.run_dir.resolve()
        cfg["output_dir"], cfg["run_name"] = str(location.parent), location.name
    validate_config(cfg)
    return cfg


def main():
    args = argument_parser().parse_args()
    cfg = config_from_arguments(args)
    context = initialize_distributed(cfg["device"])
    try:
        run(args, cfg, context)
    finally:
        destroy_distributed(context)


if __name__ == "__main__":
    main()
