# Release verification

Checked on 1 October 2026 in a fresh Python 3.12.14 virtual environment on macOS ARM64, using the pinned `requirements.txt` and the README's editable installation command.

- Dependency resolution and `pip check`: passed.
- Relevant implementation and release entry point tests: **150 passed**. Tests exercise all four roles on both datasets, paired initialization, exact checkpoint resume, worker-invariant streams, synthetic fitting/final scoring and frozen contract guards.
- Every shipped package module imports; every public script's `--help` succeeds.
- Downloaded MNIST into a new temporary directory using the documented preparation command. All checksums passed and the identity manifest matched the supplied metadata.
- Trained one full-architecture, batch-128 Shared-Var update on real MNIST in the separate software-smoke namespace, then resumed successfully for a second update through the portable wrapper. This is a software check, not a new scientific result.
- Recomputed the published analysis: **40 complete outcomes, no failures or missing entries**, with all four paired contrasts matching the stored analysis. The 11 original numerical export hashes match; redundant earlier plots/cluster notes are excluded, and the missing training-curves export is disclosed.
- Rebuilt the assets: **19 table/macro files match the latest preprint byte for byte**, 13 figures and 227 number macros generated. The independently rerun FLOP recount matches every stored measurement row.
- The final evaluation configuration hash matches the frozen analysis contract.
- The manuscript compiles to **27 pages**, with no LaTeX warnings or overfull/underfull boxes. The current source includes the official repository URL.

GitHub Actions repeats installation, all tests and offline result/asset reproduction on Linux CPU with PyTorch 2.5.1 and 2.8.0.

The full 40 GPU trainings were not rerun. The original checkpoints and fitted heads remain unavailable, and the full MPI3D download/real-array evaluation was not exercised locally. MPI3D extraction is tested with a small NPZ fixture for byte preservation, corruption rejection and safe retry; real data requires the declared checksum. CUDA/DDP execution requires the documented hardware and is not certified by the CPU checks.
