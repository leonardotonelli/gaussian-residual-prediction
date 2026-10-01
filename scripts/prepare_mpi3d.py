"""Extract and verify MPI3D-realistic images.npy without loading the array into RAM."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

import numpy as np

URL = "https://huggingface.co/datasets/waleedgondal/mpi3d/resolve/main/mpi3d_realistic.npz"
EXPECTED_SHA256 = "f6124f358e02846695cb3795acf8714f990c84cf02d495796898c133cf1ae866"
SHAPE = (1036800, 64, 64, 3)


def verify(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError("images.npy SHA256 differs from the preprint dataset. Use MPI3D-realistic and extract its images.npy member without rewriting it.")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != SHAPE or array.dtype != np.uint8:
        raise ValueError("Unexpected MPI3D shape or dtype")
    return digest.hexdigest()


def extract_images(archive, output):
    if output.exists():
        return verify(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".mpi3d-", suffix=".npy", delete=False) as dst:
        temporary = Path(dst.name)
        try:
            with zipfile.ZipFile(archive) as zipped, zipped.open("images.npy") as src:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
            dst.flush()
            digest = verify(temporary)
            temporary.replace(output)
            return digest
        finally:
            temporary.unlink(missing_ok=True)


def download_archive(archive):
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=archive.parent, prefix=".mpi3d-", suffix=".download", delete=False) as dst:
        temporary = Path(dst.name)
        try:
            with urllib.request.urlopen(URL, timeout=60) as src:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
            dst.flush()
            with zipfile.ZipFile(temporary) as zipped:
                if "images.npy" not in zipped.namelist():
                    raise ValueError("Archive contains no images.npy")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, help="Downloaded mpi3d_realistic.npz")
    p.add_argument("--output", type=Path, default=Path("data/mpi3d/images.npy"))
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        print("Verified:", args.output, verify(args.output))
        return
    archive = args.archive or args.output.parent / "mpi3d_realistic.npz"
    if not archive.exists():
        if not args.download:
            p.error("Provide --archive or use --download to download MPI3D-realistic")
        download_archive(archive)
    print("Verified:", args.output, extract_images(archive, args.output))


if __name__ == "__main__":
    main()
