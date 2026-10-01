# Gaussian Residual Prediction without an EMA Target Network

Official code and reproducibility materials for the preprint by **Leonardo Tonelli and Makoto Yamada**.

[Read the preprint](paper/main.pdf) · [Install](#install) · [Reproduce results](#reproduce-the-reported-results) · [Train and evaluate](#train-and-evaluate) · [Data](#data-and-attribution)

Shared-Var combines a shared encoder, a learned Gaussian prior/posterior, and SIGReg. This repository contains the four models evaluated on Moving-MNIST and stochastic command execution on MPI3D:

| Paper name | Code identifier | Target recipe | Prediction |
|---|---|---|---|
| BYOL reference | R0 | EMA target | Deterministic |
| AdaSSL reference | R1 | EMA target | Gaussian residual |
| Shared-Det | S0 | Shared encoder + SIGReg | Deterministic |
| Shared-Var | S1 | Shared encoder + SIGReg | Gaussian residual |

On Moving-MNIST, Shared-Var improves the forecasting pipeline over Shared-Det in all five paired replications. Its difference from the AdaSSL reference remains uncertain. Much of the gain is already present in the target representation/readout. On MPI3D, none of the ten variational runs beats its own latent persistence forecast; deterministic models cannot beat this control by construction. The EMA models are local adaptations of the reference methods.

## Install

Use Python 3.9–3.12 in a new virtual environment. The pinned analysis/test environment uses PyTorch 2.8.0:

```bash
git clone https://github.com/leonardotonelli/gaussian-residual-prediction.git
cd gaussian-residual-prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'setuptools>=68' wheel
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

Run the commands below from the repository root. The editable install makes `src` importable without setting `PYTHONPATH`; all implementation files are directly in `src/`. No torchvision, notebook kernel, OpenCV, HDF5 or Slurm Python package is required.

## Reproduce the reported results

Recompute the analysis from the compact exports, without downloading datasets or training models:

```bash
python scripts/reproduce_results.py --output outputs/reproduced-analysis
python paper/scripts/build_assets.py
```

The first command verifies 11 numerical export hashes and checks all 40 outcomes and four paired contrasts against the saved analysis. Use a new output directory. The second rebuilds the empirical tables, plots, plot data and 227 number macros while retaining the supplied architecture and dataset illustration. Earlier redundant plot exports and cluster notes are omitted; the unavailable training-curves export is not used for a reported number.

Compile the manuscript with a TeX installation containing `latexmk`, or use the supplied PDF:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/main.tex
```

## Test

```bash
python -m pytest -q
```

The 150 regression tests cover updates for all four roles on both datasets, paired initialization, exact resume, worker-invariant streams, synthetic fitting/final evaluation, dataset integrity and analysis/partition guards. They require no dataset download or GPU. GitHub Actions repeats installation, tests and offline reproduction on Linux CPU with PyTorch 2.5.1 and 2.8.0.

## Train and evaluate

The original campaign used Python 3.9, PyTorch 2.5.1 with CUDA 11.8, V100 GPUs and FP32. To match that software stack, use a separate environment:

```bash
python3.9 -m venv .venv-training
source .venv-training/bin/activate
python -m pip install 'setuptools>=68' wheel
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-training.txt
python -m pip install --no-deps --no-build-isolation -e .
```

Prepare the datasets as described below. This loop runs all 40 main trainings sequentially:

```bash
for role in R0 R1 S0 S1; do
  for replication in 1 2 3 4 5; do
    python scripts/run_preprint.py train --dataset moving_mnist \
      --role "$role" --replication "$replication" \
      --output "runs/moving_mnist/$role/rep$replication"
    python scripts/run_preprint.py train --dataset mpi3d \
      --role "$role" --replication "$replication" --nproc 4 \
      --output "runs/mpi3d/$role/rep$replication"
  done
done
```

Moving-MNIST uses 75,000 updates at batch 128. MPI3D uses 100 epochs, 102,400 updates and global batch 64 (16 per rank on four GPUs). Replications 1–5 share initialization/example streams across roles; replication 0 is reserved for development/software checks. Main recipes reject reduced scientific budgets. Use `--dry-run` to inspect training commands. The wrapper defaults to zero loader workers; `--workers 4` matches the original Moving-MNIST setting. MPI3D's original setting was zero.

Override dataset locations with `--mnist-dir` and `--images`. Paths are resolved before launching the trainer. CPU software checks use `--device cpu`; MPI3D can use one CUDA GPU with `--nproc 1`, retaining global batch 64. Changes in hardware, software or process count can change scores. Train and evaluate using the same runtime/device, and keep one fixed checkout because source hashes are verified.

Fit/select readouts and probes on training/development populations, then score the final banks using the separately saved heads:

```bash
for role in R0 R1 S0 S1; do
  for replication in 1 2 3 4 5; do
    for dataset in moving_mnist mpi3d; do
      run="runs/$dataset/$role/rep$replication"
      if [ "$dataset" = moving_mnist ]; then
        checkpoint="$run/checkpoints/checkpoint_step_075000.pt"
      else
        checkpoint="$run/checkpoints/checkpoint_epoch_0100.pt"
      fi
      python scripts/run_preprint.py evaluate --dataset "$dataset" \
        --stage fit --checkpoint "$checkpoint" --output "$run/fit"
      python scripts/run_preprint.py evaluate --dataset "$dataset" \
        --stage final --checkpoint "$checkpoint" --fitted "$run/fit/fitted.pt" \
        --output "$run/final"
    done
  done
done
python scripts/summarize_five_seed_campaign.py \
  --contract results/campaigns/five_seed_v1/20260925_eval_fix_v3/frozen_analysis.json \
  --results runs/moving_mnist/*/rep*/final/summary.json runs/mpi3d/*/rep*/final/summary.json \
  --output outputs/fresh-training-analysis
```

The fit stage is internally called `development`, even for a main checkpoint. Final scoring checks the head/checkpoint hashes and frozen query-bank contract and never refits on test data. Supply the same non-default `--mnist-dir` to both Moving-MNIST evaluations; MPI3D reads its image location from the checkpoint config. The asset builder continues to use the published exports, keeping fresh measurements separate.

<details>
<summary>Resume, illustration and cost commands</summary>

`--stop-after` pauses at an absolute optimizer step (Moving-MNIST) or completed epoch (MPI3D), preserving the full schedule. Moving-MNIST resumes into a new segment; MPI3D resumes in the same directory with the same process count:

```bash
python scripts/run_preprint.py train --dataset moving_mnist --role S1 --replication 1 \
  --output runs/moving_mnist/S1/rep1-resume \
  --resume runs/moving_mnist/S1/rep1/checkpoints/checkpoint_step_002500.pt
python scripts/run_preprint.py train --dataset mpi3d --role S1 --replication 1 --nproc 4 \
  --output runs/mpi3d/S1/rep1 \
  --resume runs/mpi3d/S1/rep1/checkpoints/checkpoint_latest.pt
python paper/scripts/build_assets.py --regenerate-illustration --mnist-dir data/concept2-mnist
mkdir -p outputs
python scripts/benchmark_role_training_cost.py --device cpu --warmup 1 --measured 2 \
  --inference-batch 8 --output outputs/cost-cpu.json
python paper/scripts/recount_flops.py --output outputs/flop-recount.json
```

Do not modify source/configuration/manifests between resume segments. Evaluate the terminal checkpoint from the resumed segment. Illustration regeneration uses a training identity and the supplied MPI3D rendering. CPU timings differ from the paper's Apple MPS benchmark (10 warm-up updates, 100 measured updates, two sweeps). FLOP counts exclude elementwise/normalization, optimizer, EMA and SIGReg characteristic-function work.

</details>

## Data and attribution

Moving-MNIST is generated online from raw MNIST digits. Download and verify the archives and create the fixed identity manifest:

```bash
python scripts/prepare_moving_mnist.py --data-dir data/concept2-mnist --download
```

The checksum-verified split has 50,000 train, 10,000 development and 10,000 official test identities. `assets/mnist_identity_manifest.json` supplies metadata only. MNIST attribution: LeCun, Bottou, Bengio and Haffner, *Gradient-Based Learning Applied to Document Recognition*, 1998. The official distribution page does not state a dataset license; MIT applies to our code, not the digit data.

Use MPI3D's **realistic rendered** variant from the [official dataset repository](https://github.com/rr-learning/disentanglement_dataset), which links the [MPI3D-realistic archive](https://huggingface.co/datasets/waleedgondal/mpi3d/resolve/main/mpi3d_realistic.npz):

```bash
python scripts/prepare_mpi3d.py --archive /path/to/mpi3d_realistic.npz
# Alternatively, download it explicitly:
python scripts/prepare_mpi3d.py --download
```

The preparer streams the archive's `images.npy` to `data/mpi3d/images.npy`, preserving its header, and verifies its checksum, shape `(1036800, 64, 64, 3)` and `uint8` dtype. Allow space for both the archive and the approximately 12.7 GB array. Existing arrays are verified. Expected SHA256:

```text
f6124f358e02846695cb3795acf8714f990c84cf02d495796898c133cf1ae866
```

MPI3D and its supplied illustration retain [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) terms and attribution to Gondal et al., *On the Transfer of Inductive Bias from Simulation to the Real World: a New Disentanglement Dataset*, 2019. Raw dataset pixels are not redistributed. The unchanged NeurIPS 2026 style retains its original notice. Method references and task definitions are in the paper; these are local adaptations of BYOL, AdaSSL, LeJEPA and LeWorldModel.

## Repository and limits

`src/` contains the implementations; `scripts/` the entry points; `config/` the final recipes and manifests; `results/` the immutable evidence; `paper/` the manuscript/assets; and `tests/` the regression suite.

Original checkpoints, fitted heads, per-query arrays, training curves and the MPI3D array were unavailable for the manuscript analysis and are not included. The release reproduces the reported analysis from run-level exports; fresh model reproduction requires training. It is not certified as the byte-exact original executed snapshot. `release_provenance.json` records source mappings; historical paths in evidence are provenance only. The paper discloses earlier MPI3D split inspection and that earlier MNIST test access cannot be ruled out. Full 40-run GPU training and real MPI3D evaluation were not rerun for the release.

The authors' software uses the [MIT license](LICENSE). Please cite the accompanying preprint; GitHub's citation menu uses [CITATION.cff](CITATION.cff).
