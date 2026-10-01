"""Small utilities shared by scripts and training code."""

import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


class _TeeStream:
    """Write console output to its original stream and one run log file."""

    def __init__(self, console, log_file) -> None:
        self.console = console
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.console.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file into a Python dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return config


def ensure_dir(path: str) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def create_run_dir(output_dir: str, run_name: str = None) -> Path:
    """Create a unique run directory under an experiment output root."""
    output_root = ensure_dir(output_dir)
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    for attempt in range(100):
        suffix = "" if attempt == 0 else f"_{datetime.now().strftime('%f')}_{attempt}"
        run_dir = output_root / f"{run_name}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError(f"Could not create a unique run directory under {output_root}")

    (run_dir / "checkpoints").mkdir()
    (run_dir / "metrics").mkdir()
    (run_dir / "logs").mkdir()
    return run_dir


def tee_console_to_file(path: Path) -> None:
    """Keep console output visible while also recording it in a run log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log_file = path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_file)
    sys.stderr = _TeeStream(sys.stderr, log_file)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save a dictionary as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    """Save a dictionary as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def seed_everything(seed: int) -> None:
    """Seed common random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve an explicit or automatic torch device."""
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
