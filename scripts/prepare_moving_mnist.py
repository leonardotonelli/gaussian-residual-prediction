"""Download/verify raw MNIST and seal identity splits; generate no videos."""

import argparse
from pathlib import Path

from iwm_replication.moving_mnist_data import identity_manifest, prepare_mnist, write_once_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Explicit network/write opt-in")
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()
    resources = prepare_mnist(args.data_dir, download=args.download)
    manifest = identity_manifest(resources, args.split_seed)
    path = args.data_dir / f"concept2-identities-v1-seed{args.split_seed}.json"
    if args.download:
        write_once_json(path, manifest)
    elif not path.exists():
        raise FileNotFoundError(f"Missing {path}; create it using --download")
    else:
        # Identical retry verifies without writing, also with read-only data.
        write_once_json(path, manifest)
    print(f"Verified MNIST: {args.data_dir}")
    print(f"Identity manifest: {path}")
    print(f"SHA256: {manifest['sha256']}")
    print("Images: train=50000, development=10000, test=10000. No videos generated.")


if __name__ == "__main__":
    main()
