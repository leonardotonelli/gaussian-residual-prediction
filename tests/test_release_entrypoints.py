"""Exercise release path handling, training contracts and streamed dataset integrity."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import yaml

import prepare_mpi3d
import run_preprint
from src.mpi3d_byol import resolve_matched_config
from src.seed_streams import SeedContext


@pytest.mark.parametrize("dataset", ["moving_mnist", "mpi3d"])
@pytest.mark.parametrize("role", ["R0", "R1", "S0", "S1"])
def test_training_wrapper_from_other_directory(tmp_path, monkeypatch, dataset, role):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(run_preprint.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    args = run_preprint.parser().parse_args([
        "train", "--dataset", dataset, "--role", role, "--replication", "1",
        "--output", "runs/endpoint", "--device", "cpu", "--mnist-dir", "digits", "--images", "images.npy"])
    run_preprint.train(args)
    command, kwargs = calls[0]
    assert kwargs["cwd"] == run_preprint.ROOT
    cfg = yaml.safe_load((tmp_path / "runs/endpoint.config.yaml").read_text())
    assert not (tmp_path / "runs/endpoint").exists()  # The trainer owns the fresh run directory.
    if dataset == "moving_mnist":
        assert cfg["data_dir"] == str(tmp_path / "digits")
        assert SeedContext.from_dict(cfg["seed_streams"]) == SeedContext(dataset, "main-training", 1)
        assert cfg["training"]["total_steps"] == 75000
    else:
        assert cfg["data"]["images_path"] == str(tmp_path / "images.npy")
        bound = resolve_matched_config(cfg, role=role, replication=1, purpose="main-training")
        assert bound["train"]["epochs"] == 100
        assert command[command.index("--run-dir") + 1] == str(tmp_path / "runs/endpoint")
    changed = (tmp_path / "runs/endpoint.config.yaml")
    changed.write_text(changed.read_text() + "extra: true\n")
    with pytest.raises(ValueError, match="configuration differs"):
        run_preprint.train(args)


def test_final_wrapper_binds_files_and_frozen_contract(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "checkpoint.pt").write_bytes(b"checkpoint")
    (tmp_path / "fitted.pt").write_bytes(b"fitted")
    calls = []
    monkeypatch.setattr(run_preprint.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    args = run_preprint.parser().parse_args([
        "evaluate", "--dataset", "mpi3d", "--stage", "final", "--checkpoint", "checkpoint.pt",
        "--fitted", "fitted.pt", "--output", "final", "--device", "cpu"])
    run_preprint.evaluate(args)
    command, kwargs = calls[0]
    assert kwargs["cwd"] == run_preprint.ROOT
    assert command[command.index("--fitted-sha256") + 1] == hashlib.sha256(b"fitted").hexdigest()
    contract = Path(command[command.index("--analysis-contract") + 1])
    assert contract.is_file()
    assert command[command.index("--frozen-protocol-sha256") + 1] == json.loads(contract.read_text())["protocol_sha256"]


def test_mpi3d_extraction_preserves_npy_bytes_and_rejects_corruption(tmp_path, monkeypatch):
    array = np.arange(48, dtype=np.uint8).reshape(4, 2, 2, 3)
    payload = io.BytesIO()
    np.save(payload, array, allow_pickle=False)
    original = payload.getvalue()
    archive = tmp_path / "realistic.npz"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("images.npy", original)
    monkeypatch.setattr(prepare_mpi3d, "SHAPE", array.shape)
    monkeypatch.setattr(prepare_mpi3d, "EXPECTED_SHA256", hashlib.sha256(original).hexdigest())
    output = tmp_path / "data/images.npy"
    prepare_mpi3d.extract_images(archive, output)
    assert output.read_bytes() == original
    prepare_mpi3d.extract_images(archive, output)  # Safe verified retry.
    output.unlink()
    monkeypatch.setattr(prepare_mpi3d, "EXPECTED_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        prepare_mpi3d.extract_images(archive, output)
    assert not output.exists()
    assert list(output.parent.iterdir()) == []
