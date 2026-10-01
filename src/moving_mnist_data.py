"""MNIST preparation and identity splits for concept2 (no video download).

Raw MNIST stays in the supplied data directory. Videos are sampled on demand; no labels enter a
world-model objective. The manifest is immutable and safe to validate offline.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

import numpy as np


MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
# Official MNIST resource checksums, also used by torchvision.datasets.MNIST.
RESOURCES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
SPLIT_VERSION = "concept2-mnist-identities-v1"


def content_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_once_json(path: Path, value: dict) -> None:
    """Allow identical retries, refuse to replace a different manifest."""
    # Normalize tuples from dataclass source states to their persisted JSON form.
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing to replace different manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_idx(path: Path) -> np.ndarray:
    """Validate IDX type, dimensionality and exact payload size before loading."""
    payload = path.read_bytes()
    if len(payload) < 4 or payload[:3] != b"\x00\x00\x08" or payload[3] not in (1, 3):
        raise ValueError(f"Invalid uint8 MNIST IDX header: {path}")
    ndim = payload[3]
    header_size = 4 + ndim * 4
    if len(payload) < header_size:
        raise ValueError(f"Truncated IDX dimensions: {path}")
    shape = struct.unpack(">" + "I" * ndim, payload[4:header_size])
    if any(n == 0 for n in shape) or len(payload) - header_size != int(np.prod(shape)):
        raise ValueError(f"IDX payload size does not match dimensions: {path}")
    return np.frombuffer(payload, dtype=np.uint8, offset=header_size).reshape(shape)


def prepare_mnist(root: Path, *, download: bool = False) -> dict:
    """Verify all four original archives and raw files; network only by opt-in.

    Existing corrupt files cause an error instead of being silently replaced.
    Downloads use a temporary sibling file and are promoted only after checksum
    verification. This matches torchvision's root/MNIST/raw layout.
    """
    raw = root / "MNIST" / "raw"
    if download:
        raw.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, expected_md5 in RESOURCES.items():
        archive = raw / name
        if not archive.exists():
            if not download:
                raise FileNotFoundError(f"Missing {archive}; run prepare with --download on the login node")
            temporary = archive.with_suffix(".gz.part")
            # Exclusive creation prevents concurrent preparations trampling each other.
            with temporary.open("xb") as output:
                with urllib.request.urlopen(MNIST_URL + name, timeout=60) as response:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            if hashlib.md5(temporary.read_bytes()).hexdigest() != expected_md5:
                raise ValueError(f"MNIST download checksum mismatch: {temporary}")
            temporary.rename(archive)
        compressed = archive.read_bytes()
        if hashlib.md5(compressed).hexdigest() != expected_md5:
            raise ValueError(f"MNIST archive checksum mismatch: {archive}")
        unpacked = gzip.decompress(compressed)
        idx = raw / name.removesuffix(".gz")
        if idx.exists():
            if idx.read_bytes() != unpacked:
                raise ValueError(f"Raw MNIST differs from verified archive: {idx}")
        elif download:
            with idx.open("xb") as output:
                output.write(unpacked)
        else:
            raise FileNotFoundError(f"Missing extracted MNIST: {idx}")
        array = read_idx(idx)
        count = 60000 if name.startswith("train") else 10000
        expected_shape = (count, 28, 28) if "images" in name else (count,)
        if array.shape != expected_shape or ("labels" in name and array.max() > 9):
            raise ValueError(f"Invalid MNIST dimensions or labels: {idx}")
        records[name] = {"url": MNIST_URL + name, "md5": expected_md5,
                         "sha256": hashlib.sha256(compressed).hexdigest(),
                         "raw_sha256": hashlib.sha256(unpacked).hexdigest(),
                         "shape": list(array.shape)}
    return {"dataset": "MNIST", "resources": records}


def identity_manifest(resources: dict, seed: int = 0) -> dict:
    """50k train / 10k development from official train; official test untouched.

    The partition seed is a local choice; the reference does not release IDs.
    Namespacing prevents test index 0 from colliding with train index 0.
    """
    order = np.random.Generator(np.random.PCG64(seed)).permutation(60000)
    payload = {
        "version": SPLIT_VERSION, "seed": seed, "rng": "numpy.PCG64",
        "resources": resources,
        "splits": {
            "train": [f"mnist-train:{i:05d}" for i in sorted(order[:50000])],
            "development": [f"mnist-train:{i:05d}" for i in sorted(order[50000:])],
            "test": [f"mnist-test:{i:05d}" for i in range(10000)],
        },
    }
    return {**payload, "sha256": content_hash(payload)}


def load_identity_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("sha256") != content_hash(payload):
        raise ValueError("Identity manifest hash mismatch")
    expected = identity_manifest(manifest["resources"], manifest["seed"])
    if manifest != expected:
        raise ValueError("Identity manifest does not match versioned split policy")
    return manifest


def load_digits(root: Path, manifest: dict, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Offline loading only. Preparation performs full archive verification."""
    ids = manifest["splits"][split]
    prefix = "t10k" if split == "test" else "train"
    raw = root / "MNIST" / "raw"
    for suffix in ("images-idx3-ubyte", "labels-idx1-ubyte"):
        path = raw / f"{prefix}-{suffix}"
        expected = manifest["resources"]["resources"][path.name + ".gz"]["raw_sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"MNIST content does not match identity manifest: {path}")
    indices = np.array([int(key.split(":")[1]) for key in ids])
    images = read_idx(raw / f"{prefix}-images-idx3-ubyte")[indices]
    labels = read_idx(raw / f"{prefix}-labels-idx1-ubyte")[indices]
    return images, labels, ids
