# Scripts — Cluster Job Submission Guide

This directory contains environment setup scripts, dataset downloaders, and
SLURM sbatch templates for submitting experiments on the cluster.

## Prerequisites

All commands below are run **on the cluster login node** unless otherwise noted.
Replace `ab1234` with your own NetID throughout.

### 1. Set up scratch environment variables

```bash
NETID=ab1234 source scripts/set_scratch_env.sh
```

This creates the directory layout under `/scratch/<NETID>` and exports
`CONDA_ENVS_DIRS`, `PIP_CACHE_DIR`, `HF_HOME`, `TMPDIR`, etc., so that
nothing is written to `/home` (which has a small quota on this cluster).

### 2. Create the conda environment

```bash
NETID=ab1234 bash scripts/setup_env.sh
```

This creates a conda env at `/scratch/<NETID>/conda-envs/gpc` with Python 3.12
and installs all project dependencies from `requirements.txt` (numpy, scipy,
scikit-learn, torch, torchvision, h5py, tabpfn, etc.).

To also install WILDS dataset download dependencies:

```bash
NETID=ab1234 bash scripts/setup_env.sh --with-datasets
```

### 3. Download datasets (optional, for Experiments 3–4)

```bash
python scripts/download_datasets.py --datasets pcam --root datasets
python scripts/download_datasets.py --datasets camelyon17 --root datasets
```

---

## Submitting Experiments

### Experiment 1: GPyTorch SVGP Binary GP Classification

Runs the sparse variational GP classifier on frozen embeddings. By default the
job prefers the HG-style embedding layout, so on PCam it will use:

- `datasets/pcam-hg/train/embeddings/projected_512.npy`
- `datasets/pcam-hg/valid/embeddings/projected_512.npy`
- `datasets/pcam-hg/test/embeddings/projected_512.npy`

```bash
sbatch --export=ALL,NETID=ab1234,DATASET=pcam scripts/exp1_gpytorch_svgp_gpc.sbatch
```

| Variable               | Default         | Description                                      |
| ---------------------- | --------------- | ------------------------------------------------ |
| `NETID`                | auto-detected   | Your cluster NetID                               |
| `DATASET`              | `pcam`          | `pcam`, `camelyon17`, or `embed`                 |
| `EMBEDDING_DIR`        | auto-detected   | Defaults to `datasets/<dataset>-hg` if present   |
| `CONDA_ENV`            | auto-detected   | Override conda env path                          |
| `PROJECT_ROOT`         | auto-detected   | Override project directory                       |
| `NUM_INDUCING`         | `256`           | Number of inducing points                        |
| `BATCH_SIZE`           | `2048`          | Training batch size                              |
| `PREDICT_BATCH_SIZE`   | `4096`          | Validation/test prediction batch size            |
| `EPOCHS`               | `15`            | Maximum number of epochs                         |
| `PATIENCE`             | `4`             | Early stopping patience on validation AUROC      |
| `LEARNING_RATE`        | `0.01`          | Adam learning rate                               |
| `MAX_TRAIN_SAMPLES`    | `0`             | Subsample train set if nonzero                   |
| `MAX_VAL_SAMPLES`      | `0`             | Subsample validation set if nonzero              |
| `MAX_TEST_SAMPLES`     | `0`             | Subsample test set if nonzero                    |
| `DISABLE_STANDARDIZE`  | `0`             | Set to `1` to disable train-stat standardization |

**Examples:**

```bash
# Full PCam run on the existing 512-d projected embeddings
sbatch --export=ALL,NETID=ab1234,DATASET=pcam scripts/exp1_gpytorch_svgp_gpc.sbatch

# Faster smoke test with a capped train set and fewer inducing points
sbatch --export=ALL,NETID=ab1234,DATASET=pcam,MAX_TRAIN_SAMPLES=50000,MAX_VAL_SAMPLES=10000,MAX_TEST_SAMPLES=10000,NUM_INDUCING=128 \
       scripts/exp1_gpytorch_svgp_gpc.sbatch
```

**Outputs:** `data/exp1_gpytorch_svgp_<dataset>_results.json`,
`data/exp1_gpytorch_svgp_<dataset>_posterior.npz`,
`data/exp1_gpytorch_svgp_<dataset>_calibration.png`,
`data/exp1_gpytorch_svgp_<dataset>_roc.png`

---

### Experiment 0: RPCholesky Algorithm Verification

CPU-only job. No GPU required.

```bash
sbatch --export=ALL,NETID=ab1234 scripts/exp0_algorithm_verification.sbatch
```

**Outputs:** `data/exp0_results.mat`, `data/exp0_*.png`

---

### Experiment 3: Deterministic Neural Network Baseline

Two-step GPU job: (1) extract frozen DenseNet-121 embeddings, then (2) train a
2-layer MLP classifier on the embeddings.

```bash
sbatch --export=ALL,NETID=ab1234,DATASET=pcam scripts/exp3_nn_baseline.sbatch
```

| Variable       | Default        | Description                                      |
| -------------- | -------------- | ------------------------------------------------ |
| `NETID`        | *(required)*   | Your cluster NetID                               |
| `DATASET`      | `pcam`         | `pcam`, `camelyon17`, or `embed`                 |
| `ENCODER`      | `densenet121`  | `densenet121` or `dinov2_vitl14`                 |
| `SKIP_EMBED`   | `0`            | Set to `1` to skip embedding extraction          |
| `CONDA_ENV`    | auto-detected  | Override conda env path                          |
| `PROJECT_ROOT` | auto-detected  | Override project directory                       |

**Examples:**

```bash
# Run on PCam with default encoder
sbatch --export=ALL,NETID=ab1234,DATASET=pcam scripts/exp3_nn_baseline.sbatch

# Run on CAMELYON17 with DINOv2 encoder
sbatch --export=ALL,NETID=ab1234,DATASET=camelyon17,ENCODER=dinov2_vitl14 \
       scripts/exp3_nn_baseline.sbatch

# Reuse existing embeddings (skip extraction step)
sbatch --export=ALL,NETID=ab1234,DATASET=pcam,SKIP_EMBED=1 \
       scripts/exp3_nn_baseline.sbatch
```

**Outputs:** `data/exp3_<dataset>_results.json`, `data/exp3_<dataset>_calibration.png`,
`data/exp3_<dataset>_roc.png`

---

### Experiment 4: TabPFN Tabular Model Baseline

Runs TabPFN on the same frozen embeddings produced by Experiment 3. **You must
run Experiment 3 first** (or at least the embedding extraction step) so that
`data/embeddings/` is populated.

```bash
sbatch --export=ALL,NETID=ab1234,DATASET=pcam scripts/exp4_tabpfn_baseline.sbatch
```

| Variable             | Default        | Description                                       |
| -------------------- | -------------- | ------------------------------------------------- |
| `NETID`              | *(required)*   | Your cluster NetID                                |
| `DATASET`            | `pcam`         | `pcam`, `camelyon17`, or `embed`                  |
| `MAX_TRAIN_SAMPLES`  | `50000`        | Subsample training set to this size for TabPFN    |
| `CONDA_ENV`          | auto-detected  | Override conda env path                           |
| `PROJECT_ROOT`       | auto-detected  | Override project directory                        |

**Examples:**

```bash
# Run on PCam (default 50K train subsample)
sbatch --export=ALL,NETID=ab1234,DATASET=pcam scripts/exp4_tabpfn_baseline.sbatch

# Run on CAMELYON17 with larger train subsample
sbatch --export=ALL,NETID=ab1234,DATASET=camelyon17,MAX_TRAIN_SAMPLES=100000 \
       scripts/exp4_tabpfn_baseline.sbatch
```

**Outputs:** `data/exp4_<dataset>_results.json`, `data/exp4_<dataset>_calibration.png`,
`data/exp4_<dataset>_roc.png`

---

## Recommended submission order

Run all three datasets through the full pipeline:

```bash
# Step 1: Extract embeddings + train NN for each dataset
sbatch --export=ALL,NETID=ab1234,DATASET=pcam       scripts/exp3_nn_baseline.sbatch
sbatch --export=ALL,NETID=ab1234,DATASET=camelyon17  scripts/exp3_nn_baseline.sbatch
sbatch --export=ALL,NETID=ab1234,DATASET=embed       scripts/exp3_nn_baseline.sbatch

# Step 2: After Exp3 jobs complete, run TabPFN on the same embeddings
sbatch --export=ALL,NETID=ab1234,DATASET=pcam        scripts/exp4_tabpfn_baseline.sbatch
sbatch --export=ALL,NETID=ab1234,DATASET=camelyon17  scripts/exp4_tabpfn_baseline.sbatch
sbatch --export=ALL,NETID=ab1234,DATASET=embed       scripts/exp4_tabpfn_baseline.sbatch
```

Use `squeue -u $USER` to monitor job status and check `slurm_logs/` for output.

---

## Monitoring and troubleshooting

```bash
# Check job queue
squeue -u $USER

# View job output in real time
tail -f slurm_logs/exp3_nn_baseline_<JOBID>.out

# Check GPU utilization on a running node
srun --jobid=<JOBID> nvidia-smi
```

**Common issues:**

| Symptom                                  | Cause                                      | Fix                                                  |
| ---------------------------------------- | ------------------------------------------ | ---------------------------------------------------- |
| `No module named ...`                    | `PYTHONHOME`/`PYTHONPATH` not unset        | Already handled in sbatch templates                  |
| `Conda env not found`                    | Env not created yet                        | Run `NETID=... bash scripts/setup_env.sh` first      |
| `CUDA out of memory`                     | Batch size too large for GPU               | Reduce `--batch-size` in the Python script           |
| `Permission denied` on `/home`           | Cache writing to home dir                  | Verify `set_scratch_env.sh` is sourced               |
| `TabPFN import failed`                   | `tabpfn` not installed                     | Run `pip install tabpfn` in the conda env            |

---

## File inventory

| File                                  | Purpose                                              |
| ------------------------------------- | ---------------------------------------------------- |
| `set_scratch_env.sh`                  | Redirects all caches to `/scratch/<NETID>`           |
| `setup_env.sh`                        | Creates conda env and installs dependencies          |
| `download_datasets.py`               | Downloads PCam, CAMELYON17-WILDS, EMBED              |
| `exp0_algorithm_verification.sbatch`  | SLURM template for RPCholesky benchmark (CPU)        |
| `exp3_nn_baseline.sbatch`             | SLURM template for NN baseline (GPU, H200)           |
| `exp4_tabpfn_baseline.sbatch`         | SLURM template for TabPFN baseline (GPU, H200)       |
