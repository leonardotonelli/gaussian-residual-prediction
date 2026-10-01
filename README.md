# Gaussian Residual Prediction without an EMA Target Network

Official code and reproducibility materials for the preprint by **Leonardo Tonelli and Makoto Yamada**.

[Read the preprint](paper/main.pdf) · [Reproduction guide](docs/REPRODUCING.md) · [Data and attribution](docs/DATA.md)

Shared-Var combines a shared encoder, a learned Gaussian prior/posterior, and SIGReg. This repository contains the four models evaluated in the paper on Moving-MNIST and stochastic command execution on MPI3D:

| Paper name | Code identifier | Target recipe | Prediction |
|---|---|---|---|
| BYOL reference | R0 | EMA target | Deterministic |
| AdaSSL reference | R1 | EMA target | Gaussian residual |
| Shared-Det | S0 | Shared encoder + SIGReg | Deterministic |
| Shared-Var | S1 | Shared encoder + SIGReg | Gaussian residual |

On Moving-MNIST, Shared-Var improves the forecasting pipeline over Shared-Det in all five paired replications. Its difference from the AdaSSL reference remains uncertain. Much of the gain is already present in the target representation/readout. On MPI3D, none of the ten variational runs beats its own latent persistence forecast; deterministic models cannot beat this control by construction. The EMA models are local adaptations of the reference methods.

## Install

Use Python 3.9–3.12 in a new virtual environment. The pinned local analysis/test environment uses PyTorch 2.8.0:

```bash
git clone https://github.com/leonardotonelli/gaussian-residual-prediction.git
cd gaussian-residual-prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'setuptools>=68' wheel
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

Commands below are run from the repository root. An editable install makes the package available without setting `PYTHONPATH`.

For the original CUDA training stack (PyTorch 2.5.1, CUDA 11.8), see [the reproduction guide](docs/REPRODUCING.md#training-environment). No torchvision, notebook kernel, OpenCV, HDF5 or Slurm Python package is required.

## Reproduce the reported results without training

The compact evidence export contains all 40 run-level outcomes. The following checks the recorded numerical export hashes and recomputes the primary analysis, including paired intervals:

```bash
python scripts/reproduce_results.py --output outputs/reproduced-analysis
python paper/scripts/build_assets.py
```

The first command requires a new output directory. The second regenerates the empirical tables, plots, plot data and number macros. It retains the supplied architecture and dataset illustration, so it does not require dataset downloads. Recreating the dataset illustration is optional and documented in the reproduction guide.

Compile the manuscript with a TeX installation containing `latexmk`, or use the supplied PDF:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/main.tex
```

## Test the implementation

```bash
python -m pytest -q
```

Tests include training updates for all four models on both datasets, exact resume checks, paired initialization, synthetic readout fitting/final evaluation, and analysis/partition guards. They require no dataset download or GPU. GitHub Actions tests installation, the regression suite and the exported-results reproduction on CPU.

## Train and evaluate fresh models

Download/verify MNIST and prepare MPI3D-realistic as described in [Data](docs/DATA.md). For example:

```bash
python scripts/prepare_moving_mnist.py --data-dir data/concept2-mnist --download
python scripts/prepare_mpi3d.py --archive /path/to/mpi3d_realistic.npz
python scripts/run_preprint.py train --dataset moving_mnist --role S1 --replication 1 --output runs/moving_mnist/S1/rep1
python scripts/run_preprint.py train --dataset mpi3d --role S1 --replication 1 --nproc 4 --output runs/mpi3d/S1/rep1
```

These are full training runs, not quick smoke tests. Moving-MNIST uses 75,000 updates; MPI3D uses 100 epochs and 102,400 updates. The guide supplies all 40 runs, checkpoint locations, separate readout fitting and final scoring, aggregation, and resume commands. Use `--dry-run` on the training wrapper to inspect a command without launching it.

## What is preserved

- `src/iwm_replication/`: the preprint implementations and shared data/scoring utilities. The original package identifier is retained.
- `config/`: final training/evaluation recipes, paired seed manifest and position manifests.
- `scripts/`: dataset preparation, training, fitting/scoring, analysis and cost benchmark entry points.
- `results/`: immutable run-level evidence and post hoc cost measurements.
- `paper/`: latest manuscript, generated tables/figures, plotted values and asset scripts.
- `tests/`: regression tests relevant to the preprint.

The original checkpoints, fitted readouts, per-query forecast arrays, training curves and MPI3D image archive are **not included and were unavailable for the manuscript analysis**. Rebuilding reported results from the exports is supported; regenerating them from models requires fresh training and datasets. This release is a cleaned version of the available implementation, not a recovered byte-exact snapshot of the original executed code. Hardware, software and topology changes can change fresh-run scores. See [reproducibility limits](docs/REPRODUCING.md#reproducibility-limits).

Code is released under the [MIT license](LICENSE). Dataset and third-party asset terms are listed in [Data and attribution](docs/DATA.md). Please cite the accompanying preprint; GitHub's citation menu uses [CITATION.cff](CITATION.cff).
