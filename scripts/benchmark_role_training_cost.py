"""Controlled per-update training-cost benchmark for the four campaign roles.

Post hoc cost measurement for the preprint; it does not train a campaign model
and never reads datasets. Each role is built with the accepted campaign recipe
and trained on synthetic tensors of the real shapes, so the timed work is the
exact forward/backward/optimizer/EMA/SIGReg path of the executed trainers.

Reported per role and dataset:
  * stored/trainable/EMA parameter counts;
  * analytic training FLOPs per update (forward + backward, torch FlopCounterMode;
    counts matmul/conv-like ops only, excludes elementwise/BN/optimizer work);
  * analytic inference FLOPs per forecast (source encoding + prior + predictor);
  * median/IQR wall-clock seconds per full update after warmup (CUDA-synchronized);
  * peak CUDA allocated memory during an update.

Run every role in one process on one GPU so hardware is held fixed. MPI3D
campaign training used 4 GPUs at global batch 64; this benchmark uses the same
global batch on one device, so it measures per-update work, not DDP overhead.

Example:
  python -s scripts/benchmark_role_training_cost.py \
      --device cuda --output results/compute_cost/benchmark_$(hostname).json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from src.moving_mnist_campaign import MODEL_OPTIONS, TRAINING
from src.moving_mnist_full_training import FullTrainer, FullTrainingConfig, ROLES, model_spec
from src.mpi3d_byol import build_model, resolve_matched_config
from src.optimization import build_adamw_optimizer
from src.seed_streams import SeedContext
from src.utils import load_yaml

# Benchmark models live in the separate smoke namespace; no campaign stream is reused.
PURPOSE = "software-smoke"


def flop_count(fn):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    with FlopCounterMode(display=False) as counter:
        fn()
    return int(counter.get_total_flops())


def saved_activation_bytes(model, loss_fn):
    """Bytes of non-parameter tensors autograd saves for backward in one update.

    Hardware-independent proxy for activation memory: unique storages captured
    by saved-tensor hooks during the forward pass, excluding parameters.
    """
    parameters = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        if storage.data_ptr() not in parameters:
            seen[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = loss_fn()
    loss.backward()
    return int(sum(seen.values()))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def time_updates(step, device, warmup, measured):
    for _ in range(warmup):
        step()
    seconds, peaks = [], []
    for _ in range(measured):
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        began = time.perf_counter()
        step()
        sync(device)
        seconds.append(time.perf_counter() - began)
        if device.type == "cuda":
            peaks.append(torch.cuda.max_memory_allocated(device))
    q = statistics.quantiles(seconds, n=4) if len(seconds) > 1 else [seconds[0]] * 3
    return {"measured_updates": measured, "warmup_updates": warmup,
            "median_update_seconds": statistics.median(seconds),
            "q1_update_seconds": q[0], "q3_update_seconds": q[2],
            "mean_update_seconds": statistics.fmean(seconds),
            "peak_cuda_allocated_bytes": max(peaks) if peaks else None}


def counts(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stored = sum(p.numel() for p in model.parameters())
    return {"trainable_parameters": trainable, "stored_parameters": stored,
            "non_trainable_parameters": stored - trainable}


def moving_mnist_role(role, device, args):
    training = FullTrainingConfig(**{**TRAINING, "total_steps": 10 ** 7})
    model_config = model_spec(role, MODEL_OPTIONS[role], training)[0]
    context = SeedContext("moving_mnist", PURPOSE, 0)
    trainer = FullTrainer(training, model_config, device, seed_context=context)
    b = training.batch_size
    clip = lambda: torch.rand(b, 1, 3, 64, 64, device=device)
    keys = ("source", "target", "surrogate") if role in ("R1", "S1") else ("source", "target")
    batch = {key: clip() for key in keys}

    def step():
        batch["sample_index"] = torch.arange(trainer.step * b, (trainer.step + 1) * b)
        trainer.update(batch)

    model = trainer.model
    kwargs = {"generator": trainer.residual_rng} if role in ("R1", "S1") else {}
    if trainer.sigreg_generators is not None:
        kwargs["sigreg_generators"] = trainer.sigreg_generators
    train_flops = flop_count(lambda: model(*[batch[k] for k in keys], **kwargs)["loss"].backward())
    activations = saved_activation_bytes(model, lambda: model(*[batch[k] for k in keys], **kwargs)["loss"])
    trainer.optimizer.zero_grad(set_to_none=True)
    timing = time_updates(step, device, args.warmup, args.measured)

    model.eval()
    encoder = model.online_encoder if role.startswith("R") else model.encoder
    source = batch["source"][: args.inference_batch]

    def infer():
        with torch.no_grad():
            z = encoder(source)["projector"]
            if role.endswith("1"):
                mean, std = model.prior(z)
                r = mean + std * torch.randn_like(std)
                model.predictor(torch.cat((z, r), -1))
            else:
                model.predictor(z)
    inference_flops = flop_count(infer)
    return {**counts(model), "batch_size": b, "encoder_passes_per_update": len(keys),
            "train_flops_per_update": train_flops, "saved_activation_bytes_per_update": activations,
            "inference_flops_per_example": None if inference_flops is None else inference_flops / len(source),
            **timing}


def mpi3d_role(role, device, args):
    cfg = resolve_matched_config(load_yaml(args.mpi3d_config), role=role, replication=0, purpose=PURPOSE)
    # The base config carries the full accepted architecture and batch; smoke only names the seed namespace.
    cfg["device"] = str(device)
    model = build_model(cfg).to(device).train()
    optimizer = build_adamw_optimizer([model], cfg["optim"]["lr"], cfg["optim"]["weight_decay_start"],
                                      cfg["optim"]["betas"])
    generator = torch.Generator(device=device).manual_seed(0)
    b, s = cfg["data"]["batch_size"], cfg["data"]["image_size"]
    source, target = (torch.rand(b, 3, s, s, device=device) for _ in range(2))
    action = torch.nn.functional.one_hot(torch.randint(0, 2, (b,), device=device), 2).float()

    def loss():
        return model(source, target, action, generator=generator)["loss"]

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss().backward()
        optimizer.step()
        model.after_optimizer_step(cfg["optim"]["ema_decay_start"])

    train_flops = flop_count(lambda: loss().backward())
    activations = saved_activation_bytes(model, loss)
    optimizer.zero_grad(set_to_none=True)
    timing = time_updates(step, device, args.warmup, args.measured)
    model.eval()
    n = args.inference_batch
    inference_flops = flop_count(lambda: model.forecast(source[:n], action[:n], quantiles=16))
    passes = 3 if role == "R1" else 2
    return {**counts(model), "batch_size": b, "encoder_passes_per_update": passes,
            "train_flops_per_update": train_flops, "saved_activation_bytes_per_update": activations,
            "inference_flops_per_example_16_supports": None if inference_flops is None else inference_flops / n,
            **timing}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="+", default=["moving_mnist", "mpi3d"])
    parser.add_argument("--roles", nargs="+", default=list(ROLES))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measured", type=int, default=200)
    parser.add_argument("--inference-batch", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repeat the role sweep in alternating order to expose drift")
    parser.add_argument("--mpi3d-config", default="config/campaigns/five_seed_v1/mpi3d.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    rows = []
    for repeat in range(args.repeats):
        order = args.roles if repeat % 2 == 0 else list(reversed(args.roles))
        for dataset in args.datasets:
            for role in order:
                run = moving_mnist_role if dataset == "moving_mnist" else mpi3d_role
                row = {"dataset": dataset, "role": role, "repeat": repeat, **run(role, device, args)}
                rows.append(row)
                print(json.dumps(row), flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = {"host": platform.node(), "torch": torch.__version__,
            "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda"
                            else "Apple MPS" if device.type == "mps" else "cpu"),
            "status": "post hoc cost benchmark on synthetic tensors; not campaign evidence of accuracy",
            "flop_note": "FlopCounterMode counts matmul/convolution/attention ops, not elementwise/BN/optimizer",
            "argv": vars(args)}
    output.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
