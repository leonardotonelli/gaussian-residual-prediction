"""Train R0/R1/S0/S1 from scratch or resume a full-training checkpoint. No evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from src.moving_mnist import GeneratorConfig
from src.moving_mnist_data import content_hash, load_digits, load_identity_manifest, write_once_json
from src.moving_mnist_stream import OnlineClips
from src.seed_streams import SeedContext, context_from_config
from src.moving_mnist_training import clip_loader
from src.moving_mnist_full_training import ROLES, FullTrainer, FullTrainingConfig, model_spec
from src.moving_mnist_campaign import (
    CAMPAIGN_STATUS, model_recipe_report, resolve_campaign_config,
)


def run(config, *, role, data_dir, output_dir, device, workers, stop_after=None, resume=None,
        stop_requested=lambda: None, seed_context=None):
    if config["status"] not in ("exploratory-full-training", "training-integration-test-only", CAMPAIGN_STATUS):
        raise ValueError("Versioned full-training configuration required")
    if "recipe_version" in config and config["status"] != CAMPAIGN_STATUS:
        raise ValueError("A campaign recipe cannot fall back to a legacy training status")
    configured_context = context_from_config(config)
    if seed_context is not None and configured_context is not None:
        if seed_context != configured_context:
            raise ValueError("Conflicting explicit and configured seed contexts")
    if seed_context is None:
        seed_context = configured_context
    campaign_recipe = None
    model_options = config["models"]
    if config["status"] == CAMPAIGN_STATUS:
        model_options, campaign_recipe = resolve_campaign_config(config, seed_context)
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("Moving-MNIST campaign requires the declared float32 numerical recipe")
    cfg = FullTrainingConfig(**config["training"])
    if config["status"] == "exploratory-full-training" and (cfg.total_steps != 75000 or cfg.batch_size != 128):
        raise ValueError("This full-training protocol uses 75,000 steps and batch 128")
    mc, model_version = model_spec(role, model_options[role], cfg)
    end = cfg.total_steps if stop_after is None else stop_after
    if not 1 <= end <= cfg.total_steps or workers < 0:
        raise ValueError("Invalid stop step or worker count")
    if output_dir.exists():
        raise FileExistsError("Use a NEW output segment directory")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "torch": str(torch.__version__), "device": str(device), "cuda_build": torch.version.cuda}
    if device.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(device)
    print("Runtime:", json.dumps(environment), flush=True)
    manifest = load_identity_manifest(data_dir / config["identity_manifest"])
    if campaign_recipe is not None and manifest.get("seed") != 0:
        raise ValueError("Campaign identity manifest must retain the accepted seed-0 split")
    dataset = OnlineClips(*load_digits(data_dir, manifest, "train"), split="train", seed=cfg.seed,
                          generator_config=GeneratorConfig(**config["generator"]),
                          identity_hash=manifest["sha256"], size=cfg.total_steps * cfg.batch_size,
                          seed_context=seed_context)
    trainer = FullTrainer(cfg, mc, device, seed_context=seed_context)
    repo = Path(__file__).resolve().parents[1]
    # Hash imported project dependencies, not future unrelated evaluator files.
    files = {Path(__file__).resolve()}
    files.update(Path(module.__file__).resolve() for name, module in sys.modules.items()
                 if name.startswith("src") and getattr(module, "__file__", "").endswith(".py"))
    contract = json.loads(json.dumps({
        "version": trainer.version, "purpose": config["status"], "model_version": model_version,
        **({"seed_context": seed_context.as_dict(), "initialization": trainer.initialization}
           if seed_context is not None else {}),
        **({"campaign_recipe": campaign_recipe, "model_recipe": model_recipe_report(trainer.model)}
           if campaign_recipe is not None else {}),
        "training": asdict(cfg), "model": asdict(mc), "train_data": dataset.contract,
        "environment": environment, "evaluation": "deferred; train identities only",
        "code_sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}))
    parent_hash = hashlib.sha256(resume.read_bytes()).hexdigest() if resume else None
    if resume:
        sidecar = resume.with_suffix(".json")
        if sidecar.exists() and json.loads(sidecar.read_text())["sha256"] != parent_hash:
            raise ValueError("Resume checkpoint checksum mismatch")
        trainer.load(resume, contract)
    if end <= trainer.step:
        raise ValueError("Requested stop must be after the checkpoint step")
    start = trainer.step
    output_dir.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False)
    write_once_json(output_dir / "config.json", {
        "contract": contract, "contract_sha256": content_hash(contract), "data_dir": str(data_dir.resolve()),
        "workers": workers, "start_step": start, "stop_after": end, "git_revision": revision.stdout.strip(),
        "resume_from": str(resume.resolve()) if resume else None, "resume_checkpoint_sha256": parent_hash})
    loader = clip_loader(dataset, batch_size=cfg.batch_size, start=start*cfg.batch_size,
                         end=end*cfg.batch_size, workers=workers,
                         seed=cfg.seed+4000 if seed_context is None else seed_context.seed("loader"),
                         pin_memory=device.type == "cuda")
    iterator = iter(loader)
    checkpoints, last, saved_step = [], None, None
    timing = {"measured_steps": 0, "total_data_seconds": 0., "total_update_seconds": 0.,
              "max_cuda_allocated_bytes": None}
    def checkpoint():
        nonlocal saved_step
        if saved_step == trainer.step:
            return
        path = output_dir / "checkpoints" / f"checkpoint_step_{trainer.step:06d}.pt"
        trainer.save(path, contract)
        metadata = {"step": trainer.step, "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        write_once_json(path.with_suffix(".json"), metadata)
        checkpoints.append(metadata)
        saved_step = trainer.step
        print(f"{role} checkpoint: {path}", flush=True)
    print(f"{role}: training updates {start+1}..{end}/{cfg.total_steps}, batch={cfg.batch_size}; evaluation deferred", flush=True)
    with (output_dir / "learning_curve.jsonl").open("w") as log:
        for local_step in range(end-start):
            if stop_requested():
                break
            fetched = time.perf_counter()
            batch = next(iterator)
            data_seconds = time.perf_counter() - fetched
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            began = time.perf_counter()
            last = trainer.update(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_seconds = time.perf_counter() - began
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            measured = local_step >= cfg.timing_warmup_steps
            last.update(data_wait_seconds=data_seconds, update_seconds=update_seconds,
                        timed_after_warmup=measured, peak_cuda_allocated_bytes=peak)
            log.write(json.dumps(last, allow_nan=False) + "\n")
            if measured:
                timing["measured_steps"] += 1
                timing["total_data_seconds"] += data_seconds
                timing["total_update_seconds"] += update_seconds
                if peak is not None:
                    timing["max_cuda_allocated_bytes"] = max(timing["max_cuda_allocated_bytes"] or 0, peak)
            if local_step == 0 or trainer.step % cfg.log_interval == 0 or trainer.step == end:
                print(f"{role} step {trainer.step}/{cfg.total_steps}: loss={last['loss']:.6f}, "
                      f"alignment={last['alignment']:.6f}, KL={last['kl']:.6f}, SIGReg={last['sigreg']:.6f}, "
                      f"lr={last['learning_rate']:.8f}, data+update={data_seconds+update_seconds:.3f}s", flush=True)
                log.flush()
            if trainer.step % cfg.checkpoint_interval == 0:
                log.flush()
                checkpoint()
        log.flush()
        checkpoint()
    del iterator, loader
    stopped = stop_requested()
    status = "training-complete" if trainer.step == cfg.total_steps else "paused-on-signal" if stopped else "paused-at-requested-step"
    n = timing["measured_steps"]
    timing["mean_data_plus_update_seconds"] = (timing["total_data_seconds"] + timing["total_update_seconds"])/n if n else None
    summary = {"role": role, "status": status, "purpose": config["status"], "evaluation": "deferred",
               "start_step": start, "completed_step": trainer.step, "total_steps": cfg.total_steps,
               "next_sample_index": trainer.step*cfg.batch_size, "scheduler_updates": trainer.scheduler.last_epoch,
               "ema_updates": int(trainer.model.ema_updates) if role.startswith("R") else None,
               "next_learning_rate": trainer.optimizer.param_groups[0]["lr"], "batch_size": cfg.batch_size,
               "parameter_counts": trainer.model.parameter_counts(), "last_training": last,
               **({"campaign_recipe_sha256": campaign_recipe["sha256"],
                   "model_recipe": model_recipe_report(trainer.model)} if campaign_recipe is not None else {}),
               "timing": timing, "checkpoints": checkpoints, "stop_signal": stopped,
               "contract_sha256": content_hash(contract)}
    write_once_json(output_dir / "metrics.json", summary)
    print(f"{role} {status}. Artifacts: {output_dir}", flush=True)
    if status != "training-complete":
        print(f"Resume in a new segment with --resume {checkpoints[-1]['path']}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/campaigns/five_seed_v1/moving_mnist.yaml"))
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--stop-after", type=int, help="Absolute update at which to pause without changing the schedule")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed-context", type=Path,
                        help="JSON SeedContext entry from the fixed campaign manifest")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    requested = {"signal": None}
    def stop(signum, frame):
        requested["signal"] = signal.Signals(signum).name
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, stop)
    run(config, role=args.role, data_dir=args.data_dir or Path(config["data_dir"]),
        output_dir=args.output_dir, device=torch.device(args.device),
        workers=config["workers"] if args.workers is None else args.workers,
        stop_after=args.stop_after, resume=args.resume, stop_requested=lambda: requested["signal"],
        seed_context=None if args.seed_context is None else SeedContext.from_dict(json.loads(args.seed_context.read_text())))


if __name__ == "__main__":
    main()
