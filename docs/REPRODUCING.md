# Reproducing the preprint

There are two supported routes: recompute the published analysis from the preserved exports, or train new checkpoints and evaluate them with the preprint protocol. The second route produces new measurements.

## Rebuild the published analysis and manuscript

After installing the pinned requirements and this package as in the README:

```bash
python scripts/reproduce_results.py --output outputs/reproduced-analysis
python paper/scripts/build_assets.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/main.tex
```

The analysis script verifies the available original numerical export hashes, reproduces all 40 outcomes and the four paired contrasts, and checks agreement with `results/campaigns/five_seed_v1/20260925_eval_fix_v3/analysis.json`. Redundant earlier plot exports and cluster runbooks listed in the historical provenance are intentionally omitted. The only unavailable numerical export is `training_curves.csv`. It is not used to reproduce a reported number. Output directories are never silently overwritten.

The asset script independently checks primary values against `diagnostics.csv`, regenerates tables and empirical figures, and writes a manifest linking inputs, metric keys and plotted values. It retains the supplied Figma architecture export and dataset illustration. To regenerate the illustration from MNIST and the supplied MPI3D rendering:

```bash
python scripts/prepare_moving_mnist.py --data-dir data/concept2-mnist --download
python paper/scripts/build_assets.py --regenerate-illustration --mnist-dir data/concept2-mnist
```

The illustration uses a training identity and the documented selection rule, not an evaluation query. Plot fonts and PDF metadata can vary across systems; the numerical tables and macros are the comparison targets. LaTeX is needed only to compile the paper. Ghostscript is optional for an architecture PNG preview.

## Training environment

The original campaign used Python 3.9, PyTorch 2.5.1 with CUDA 11.8 and V100 GPUs, FP32. Use a separate environment to match that stack:

```bash
python3.9 -m venv .venv-training
source .venv-training/bin/activate
python -m pip install 'setuptools>=68' wheel
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-training.txt
python -m pip install --no-deps --no-build-isolation -e .
```

The [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/) describe platform-specific wheel alternatives. The pinned `requirements.txt` is for the tested local analysis and CPU tests; it does not claim to match the original training runtime. Evaluation checks that Moving-MNIST's runtime matches its checkpoint's recorded training runtime: train and evaluate using the same Python, PyTorch, NumPy and device.

Prepare the datasets following [DATA.md](DATA.md). Default paths are `data/concept2-mnist/` and `data/mpi3d/images.npy`. Override them with `--mnist-dir` and `--images` on the training wrapper. The wrapper anchors source/configuration paths to this checkout and resolves supplied data/output paths before launching the trainer. Run from one fixed checkout throughout training and evaluation because source hashes are verified.

## The 40 main training runs

The four roles share module initialization and training example streams within each replication. Replications are 1–5; replication 0 is reserved for development/software smoke. The existing seed manifest is used unchanged. The final recipes reject reductions of scientific architecture or training budgets in the main namespace.

This loop runs the 40 main trainings sequentially. MPI3D launches four local CUDA processes for each run; the original global batch is 64 (16 per rank):

```bash
for role in R0 R1 S0 S1; do
  for replication in 1 2 3 4 5; do
    python scripts/run_preprint.py train \
      --dataset moving_mnist --role "$role" --replication "$replication" \
      --output "runs/moving_mnist/$role/rep$replication"
    python scripts/run_preprint.py train \
      --dataset mpi3d --role "$role" --replication "$replication" --nproc 4 \
      --output "runs/mpi3d/$role/rep$replication"
  done
done
```

Moving-MNIST defaults to zero loader workers in the portable wrapper; `--workers 4` matches the original loader setting. Addressed data streams are tested for worker invariance. MPI3D defaults to zero workers, as in the original recipe. A single CUDA GPU can run MPI3D with `--nproc 1`, retaining the global batch, but changes the distributed topology and does not promise identical numbers. CPU training is supported for software checks with `--device cpu`; full CPU runs are computationally expensive.

Terminal checkpoint paths are:

```text
runs/moving_mnist/ROLE/repN/checkpoints/checkpoint_step_075000.pt
runs/mpi3d/ROLE/repN/checkpoints/checkpoint_epoch_0100.pt
```

Each run records its resolved configuration, code/configuration hashes, seed context and checkpoint sidecars. Training and evaluation are separate. No Slurm account, cluster storage mount or automatic job submission is needed.

## Fit readouts, then score the final banks

For a trained main checkpoint, `--stage fit` fits/selects the readout and probes on the training/development populations only. The internal evaluator calls this mode `development`; it is applied to each main checkpoint and does not mean the checkpoint is a development replication. It writes `fitted.pt`.

Final scoring reuses that fitted artifact, verifies its SHA256 and checkpoint binding, checks the preserved analysis/query-bank contract, and opens the final test population. It never refits on test data.

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
```

Supply the same `--mnist-dir` on both Moving-MNIST evaluation commands if using a non-default MNIST location. MPI3D's image location is read from the checkpoint's training config; keep the array at that location. Use the same `--device` as training.

Aggregate the 40 new `summary.json` records under the preprint's frozen analysis contract:

```bash
python scripts/summarize_five_seed_campaign.py \
  --contract results/campaigns/five_seed_v1/20260925_eval_fix_v3/frozen_analysis.json \
  --results runs/moving_mnist/*/rep*/final/summary.json runs/mpi3d/*/rep*/final/summary.json \
  --output outputs/fresh-training-analysis
```

This command reports incomplete/missing outcomes explicitly. Keep fresh results separate from the supplied paper evidence. `paper/scripts/build_assets.py` deliberately continues to read the published exports.

## Pause and resume

`--stop-after` pauses at an absolute optimizer step on Moving-MNIST or at an absolute completed epoch on MPI3D while preserving the full schedule. This does not produce a final scientific endpoint.

Moving-MNIST resumes into a new output segment:

```bash
python scripts/run_preprint.py train --dataset moving_mnist --role S1 --replication 1 \
  --output runs/moving_mnist/S1/rep1-resume \
  --resume runs/moving_mnist/S1/rep1/checkpoints/checkpoint_step_002500.pt
```

MPI3D resumes its latest completed epoch in the same output directory and with the same process count:

```bash
python scripts/run_preprint.py train --dataset mpi3d --role S1 --replication 1 --nproc 4 \
  --output runs/mpi3d/S1/rep1 \
  --resume runs/mpi3d/S1/rep1/checkpoints/checkpoint_latest.pt
```

Use a terminal checkpoint from the resumed segment for evaluation. Do not modify source, scientific configuration or dataset manifests between segments.

## Cost analysis

The supplied cost JSON files reproduce the paper's post hoc figures. To make new analytic/timing measurements:

```bash
mkdir -p outputs
python scripts/benchmark_role_training_cost.py --device cpu --warmup 1 --measured 2 \
  --inference-batch 8 --output outputs/cost-cpu.json
python paper/scripts/recount_flops.py --output outputs/flop-recount.json
```

The paper's controlled timing benchmark used Apple MPS, 10 warm-up updates, 100 measured updates and two sweeps. CPU timings do not reproduce that hardware measurement. FLOP counts exclude elementwise/normalization, optimizer, EMA and SIGReg characteristic-function work. The recount controls attention backends and inference fast paths. Newly measured files are kept separate from the historical measurements used in the manuscript.

## Reproducibility limits

Original checkpoints, fitted heads, per-query arrays, training curves and the MPI3D array were not recovered for the manuscript analysis. They are not downloadable from this repository. Only the original run-level exports and measured cost records are preserved. Original evidence JSON includes historical cluster paths; these are provenance records, not operational paths.

The public implementation retains the available model/training/scoring bodies, removes unrelated model/dataset adapters, and adds portable entry points. It is not certified as the byte-exact snapshot executed on the cluster. `release_provenance.json` records the source export. The paper also discloses earlier MPI3D split inspection and that earlier MNIST test access cannot be ruled out.

Fresh training has been software-tested on CPU with synthetic inputs. Full 40-run training and real MPI3D evaluation were not rerun for the release, and GPU/DDP execution needs the declared hardware. Exact numerical reproduction across PyTorch versions, GPUs or process counts is not guaranteed. The paper's scientific claims refer to the supplied original exports.
