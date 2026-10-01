"""Consistent analytic FLOP and saved-activation recount for the four campaign roles.

Post hoc, secondary cost analysis for the preprint. It trains nothing, reads no
dataset, and never writes under ``results/``. Models are built with the same
constructors as ``scripts/benchmark_role_training_cost.py`` (accepted campaign
recipes, synthetic tensors of the real shapes).

Why a recount: ``torch.utils.flop_counter.FlopCounterMode`` only counts the
operators it has formulas for. On CPU, fused scaled-dot-product-attention
kernels selected for gradient-carrying passes are not counted, the native
multi-head-attention path used by no-grad EMA passes is counted only partly,
and the eval-mode ``nn.TransformerEncoderLayer`` fast path hides whole
transformer blocks. Forcing the math attention backend and disabling the
fast path makes every role's matmuls visible to the counter in the same way.
Moving-MNIST has no attention, so its counts are unaffected.

Scope of the count: matmul/convolution/attention-matmul FLOPs (forward and
backward); elementwise, normalization, optimizer, EMA and SIGReg
characteristic-function work are excluded, as in the benchmark.

Usage (from the repository root):
  PYTHONPATH=src python paper/scripts/recount_flops.py \
      --output paper/data/flop_recount.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import platform
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from iwm_replication.moving_mnist_campaign import MODEL_OPTIONS, TRAINING
from iwm_replication.moving_mnist_full_training import FullTrainer, FullTrainingConfig, ROLES, model_spec
from iwm_replication.mpi3d_byol import build_model, resolve_matched_config
from iwm_replication.seed_streams import SeedContext
from iwm_replication.utils import load_yaml

PURPOSE = "software-smoke"  # separate namespace; no campaign stream is reused


def count(fn):
    with FlopCounterMode(display=False) as counter:
        fn()
    by_op = {str(op): int(v) for op, v in counter.get_flop_counts()["Global"].items()}
    return int(counter.get_total_flops()), by_op


def saved_activation_bytes(model, loss_fn):
    """Same definition as the benchmark: unique non-parameter storages saved for backward."""
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
    model.zero_grad(set_to_none=True)
    return int(sum(seen.values()))


def moving_mnist(role):
    training = FullTrainingConfig(**{**TRAINING, "total_steps": 10 ** 7})
    model_config = model_spec(role, MODEL_OPTIONS[role], training)[0]
    trainer = FullTrainer(training, model_config, torch.device("cpu"),
                          seed_context=SeedContext("moving_mnist", PURPOSE, 0))
    model, b = trainer.model.train(), training.batch_size
    keys = ("source", "target", "surrogate") if role in ("R1", "S1") else ("source", "target")
    batch = {key: torch.rand(b, 1, 3, 64, 64) for key in keys}
    kwargs = {"generator": trainer.residual_rng} if role in ("R1", "S1") else {}
    if trainer.sigreg_generators is not None:
        kwargs["sigreg_generators"] = trainer.sigreg_generators
    loss = lambda: model(*[batch[k] for k in keys], **kwargs)["loss"]
    train_flops, by_op = count(lambda: loss().backward())
    model.zero_grad(set_to_none=True)
    activations = saved_activation_bytes(model, loss)
    model.eval()
    encoder = model.online_encoder if role.startswith("R") else model.encoder
    source = batch["source"][:8]
    with torch.no_grad():
        z = encoder(source)["projector"]
    encode_flops, _ = count(lambda: encoder(source))
    if role.endswith("1"):
        r = torch.zeros(len(z), 2)
        head_flops, _ = count(lambda: model.prior(z))
        draw_flops, _ = count(lambda: model.predictor(torch.cat((z, r), -1)))
    else:
        head_flops, (draw_flops, _) = 0, count(lambda: model.predictor(z))
    n = len(source)
    return {"batch_size": b, "train_flops_per_update": train_flops, "train_flops_by_operator": by_op,
            "saved_activation_bytes_per_update": activations,
            "inference_encoder_flops_per_source": encode_flops / n,
            "inference_prior_flops_per_source": head_flops / n,
            "inference_predictor_flops_per_draw": draw_flops / n}


def mpi3d(role, config_path):
    cfg = resolve_matched_config(load_yaml(config_path), role=role, replication=0, purpose=PURPOSE)
    cfg["device"] = "cpu"
    model = build_model(cfg).train()
    generator = torch.Generator().manual_seed(0)
    b, s = cfg["data"]["batch_size"], cfg["data"]["image_size"]
    source, target = torch.rand(b, 3, s, s), torch.rand(b, 3, s, s)
    action = torch.nn.functional.one_hot(torch.randint(0, 2, (b,)), 2).float()
    loss = lambda: model(source, target, action, generator=generator)["loss"]
    result = {"batch_size": b}
    for label, backend in (("default_backend", None), ("math_attention_backend", SDPBackend.MATH)):
        context = sdpa_kernel(backend) if backend is not None else contextlib.nullcontext()
        with context:
            flops, by_op = count(lambda: loss().backward())
            model.zero_grad(set_to_none=True)
            activations = saved_activation_bytes(model, loss)
        result[label] = {"train_flops_per_update": flops, "train_flops_by_operator": by_op,
                         "saved_activation_bytes_per_update": activations}
    model.eval()
    n = 8
    x, a = source[:n], action[:n]
    # Grad-enabled eval mode disables the TransformerEncoderLayer fast path so
    # that every block is counted; BatchNorm uses running statistics as at inference.
    with sdpa_kernel(SDPBackend.MATH), torch.enable_grad():
        encode_flops, _ = count(lambda: model.online(x))
        z = model.online(x)["projector"].detach()
        if model.is_stochastic:
            head_flops, _ = count(lambda: model.prior(torch.cat((z, a), -1)))
        else:
            head_flops = 0
        residual = torch.zeros(n, 1)
        draw_flops, _ = count(lambda: model.predictor(torch.cat((z, a, residual), -1)))
    result.update(inference_encoder_flops_per_source=encode_flops / n,
                  inference_prior_flops_per_source=head_flops / n,
                  inference_predictor_flops_per_draw=draw_flops / n)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mpi3d-config", default="config/campaigns/five_seed_v1/mpi3d.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    rows = []
    for role in ROLES:
        rows.append({"dataset": "moving_mnist", "role": role, **moving_mnist(role)})
        rows.append({"dataset": "mpi3d", "role": role, **mpi3d(role, args.mpi3d_config)})
    meta = {"host_platform": platform.platform(), "torch": torch.__version__, "device": "cpu",
            "status": "post hoc analytic recount on synthetic tensors; not campaign evidence of accuracy",
            "scope": "FlopCounterMode matmul/convolution/attention-matmul FLOPs; excludes elementwise, "
                     "normalization, optimizer, EMA and SIGReg characteristic-function work",
            "mpi3d_note": "math_attention_backend forces countable attention matmuls in every pass; "
                          "default_backend reproduces results/compute_cost/benchmark_local_cpu_flops.json"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
