"""Portable training, readout fitting and final scoring for the preprint recipes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml
from iwm_replication.seed_streams import SeedContext

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/campaigns/five_seed_v1"
EVIDENCE = ROOT / "results/campaigns/five_seed_v1/20260925_eval_fix_v3"


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def execute(command, dry_run):
    print(json.dumps({"cwd": str(ROOT), "command": command}), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def train(args):
    if args.dataset == "moving_mnist" and args.nproc != 1:
        raise ValueError("Moving-MNIST uses one process")
    if args.device == "cpu" and args.nproc != 1:
        raise ValueError("CPU software runs use one process; the MPI3D campaign uses four CUDA GPUs")
    context = SeedContext(args.dataset, args.purpose, args.replication)
    cfg = yaml.safe_load((CONFIG / (args.dataset + ".yaml")).read_text())
    out = args.output.resolve()
    config_path = out.parent / (out.name + ".config.yaml")
    if args.dataset == "moving_mnist":
        cfg["data_dir"] = str(args.mnist_dir.resolve())
        cfg["seed_streams"] = context.as_dict()
        cfg["workers"] = args.workers
        command = [sys.executable, str(ROOT / "scripts/train_moving_mnist_full.py"), "--config", str(config_path),
                   "--role", args.role, "--output-dir", str(out), "--device", args.device]
        if args.stop_after is not None:
            command += ["--stop-after", str(args.stop_after)]
        if args.resume:
            command += ["--resume", str(args.resume.resolve())]
    else:
        cfg["device"] = args.device
        cfg["data"]["images_path"] = str(args.images.resolve())
        cfg["data"]["num_workers"] = args.workers
        cfg["output_dir"] = str(out.parent)
        command = [sys.executable]
        if args.nproc > 1:
            command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(args.nproc)]
        command += [str(ROOT / "scripts/train_mpi3d_byol.py"), "--config", str(config_path),
                    "--role", args.role, "--replication", str(args.replication), "--purpose", args.purpose,
                    "--run-dir", str(out)]
        if args.stop_after is not None:
            command += ["--stop-after-epoch", str(args.stop_after)]
        if args.resume:
            if args.resume.resolve() != out / "checkpoints/checkpoint_latest.pt":
                raise ValueError("MPI3D resumes checkpoint_latest.pt in the same run directory")
            command += ["--resume"]
    if not args.dry_run:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(cfg, sort_keys=False)
        if config_path.exists() and config_path.read_text() != text:
            raise ValueError("Existing run configuration differs; use a new output directory")
        config_path.write_text(text)
    execute(command, args.dry_run)


def evaluate(args):
    checkpoint = args.checkpoint.resolve()
    command = [sys.executable, str(ROOT / ("scripts/evaluate_five_seed_" + args.dataset + ".py")),
               "--checkpoint", str(checkpoint), "--expected-sha256", file_hash(checkpoint),
               "--config", str(CONFIG / "evaluation.yaml"), "--output", str(args.output.resolve()),
               "--device", args.device, "--mode", "development" if args.stage == "fit" else "final"]
    if args.dataset == "moving_mnist":
        command += ["--data-dir", str(args.mnist_dir.resolve())]
    if args.stage == "final":
        if args.fitted is None:
            raise ValueError("Final scoring requires --fitted from a separate fit stage")
        fitted = args.fitted.resolve()
        analysis = json.loads((EVIDENCE / "frozen_analysis.json").read_text())
        command += ["--fitted-artifacts", str(fitted), "--fitted-sha256", file_hash(fitted),
                    "--analysis-contract", str(EVIDENCE / "frozen_analysis.json"),
                    "--frozen-protocol-sha256", analysis["protocol_sha256"]]
    execute(command, args.dry_run)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_subparsers(dest="action", required=True)
    t = actions.add_parser("train")
    e = actions.add_parser("evaluate")
    for sub in (t, e):
        sub.add_argument("--dataset", choices=("moving_mnist", "mpi3d"), required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        sub.add_argument("--mnist-dir", type=Path, default=ROOT / "data/concept2-mnist")
        sub.add_argument("--dry-run", action="store_true")
    t.add_argument("--role", choices=("R0", "R1", "S0", "S1"), required=True)
    t.add_argument("--replication", type=int, required=True)
    t.add_argument("--purpose", choices=("main-training", "development-training", "software-smoke"), default="main-training")
    t.add_argument("--images", type=Path, default=ROOT / "data/mpi3d/images.npy")
    t.add_argument("--nproc", type=int, choices=(1, 2, 4), default=1)
    t.add_argument("--workers", type=int, default=0)
    t.add_argument("--stop-after", type=int, help="Absolute step for Moving-MNIST; absolute epoch for MPI3D")
    t.add_argument("--resume", type=Path)
    e.add_argument("--stage", choices=("fit", "final"), required=True)
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--fitted", type=Path)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    (train if args.action == "train" else evaluate)(args)
