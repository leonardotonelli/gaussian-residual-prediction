"""Preprint implementation: selected components from the research codebase."""
import math
import os
import time
import torch


SMOKE_VERSION = "mpi3d-training-replay-smoke-v1"


def enable_smoke_determinism(cfg):
    if cfg.get("stage") != "software-smoke":
        raise ValueError("Runtime audit is restricted to the software-smoke namespace")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Smoke subprocess requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def runtime_policy():
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED")}


class SmokeRuntime:
    """One invocation's synchronized local training-step observations."""
    def __init__(self, device, rank, world_size, global_batch):
        self.device = torch.device(device)
        self.rank, self.world_size, self.global_batch = rank, world_size, global_batch
        self.rows = []
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def finish(self, started, *, epoch, step, metrics):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        if not seconds > 0 or not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError("Invalid measured smoke step")
        row = {"epoch": epoch, "step": step, "elapsed_seconds": seconds,
               "local_examples": self.global_batch // self.world_size,
               "global_batch": self.global_batch, "finite_objective_and_gradients": True,
               "losses": metrics}
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
            row[name + "_bytes"] = int(getattr(torch.cuda, name)(self.device)) if self.device.type == "cuda" else None
        self.rows.append(row)

    def report(self):
        return {"schema": SMOKE_VERSION, "rank": self.rank, "world_size": self.world_size,
                "device": str(self.device), "policy": runtime_policy(), "steps": self.rows,
                "timing_scope": "synchronized forward/backward/optimizer/EMA; excludes input fetch and checkpoint I/O",
                "memory_scope": "allocator peaks since process start, including construction and checkpoint restoration"}
