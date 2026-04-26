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

### Experiment 0: RPCholesky Algorithm Verification

CPU-only job. No GPU required.

```bash
sbatch --export=ALL,NETID=ab1234 scripts/exp0_algorithm_verification.sbatch
```

**Outputs:** `data/exp0_results.mat`, `data/exp0_*.png`

---

### Experiment 3: Deterministic Neural Network Baseline

Two-step GPU job: (1) extract frozen DenseNet-121 or DINOv2 embeddings, then
(2) train a neural network classifier head on the embeddings. The default head
is now `residual_mlp`, a stronger residual MLP with LayerNorm/GELU/dropout; set
`MODEL_ARCH=mlp` to reproduce the old 2-layer MLP baseline.

```bash
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=pcam \
       scripts/exp3_nn_baseline.sbatch
```

| Variable       | Default        | Description                                      |
| -------------- | -------------- | ------------------------------------------------ |
| `NETID`        | *(required)*   | Your cluster NetID                               |
| `DATASET`      | `pcam`         | `pcam`, `camelyon17`, or `embed`                 |
| `ENCODER`      | `densenet121`  | `densenet121` or `dinov2_vitl14`                 |
| `SKIP_EMBED`   | `0`            | Set to `1` to skip embedding extraction          |
| `EMBEDDING_DIR`| `data/embeddings` | Embedding root (project format or partner HG layout) |
| `MODEL_ARCH`   | `residual_mlp` | `residual_mlp` or `mlp`                          |
| `HIDDEN_DIM`   | `512`          | Classifier hidden width                          |
| `NUM_LAYERS`   | `3`            | Residual blocks for `residual_mlp`               |
| `DROPOUT`      | `0.3`          | Classifier dropout                               |
| `LR`           | `1e-3`         | AdamW learning rate                              |
| `WEIGHT_DECAY` | `1e-4`         | AdamW weight decay                               |
| `EPOCHS`       | `50`           | Max classifier epochs                            |
| `PATIENCE`     | `5`            | Early-stopping patience on validation AUROC      |
| `BATCH_SIZE`   | `512`          | Embedding batch size                             |
| `CONDA_ENV`    | auto-detected  | Override conda env path                          |
| `PROJECT_ROOT` | auto-detected  | Override project directory                       |

**Examples:**

```bash
# Run on PCam with default encoder
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=pcam \
       scripts/exp3_nn_baseline.sbatch

# Run on CAMELYON17 with DINOv2 encoder
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=camelyon17,ENCODER=dinov2_vitl14 \
       scripts/exp3_nn_baseline.sbatch

# Reuse existing embeddings (skip extraction step)
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=pcam,SKIP_EMBED=1 \
       scripts/exp3_nn_baseline.sbatch

# Reproduce the old 2-layer MLP baseline
sbatch --account=torch_pr_xxx_yyy \
       --export=ALL,NETID=ab1234,DATASET=pcam,SKIP_EMBED=1,MODEL_ARCH=mlp,HIDDEN_DIM=256 \
       scripts/exp3_nn_baseline.sbatch

# Use partner embeddings (HG layout) for Exp3; skip extraction
sbatch --account=torch_pr_xxx_yyy \
       --export=ALL,NETID=ab1234,DATASET=pcam,SKIP_EMBED=1,EMBEDDING_DIR=/scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets \
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
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=pcam \
       scripts/exp4_tabpfn_baseline.sbatch
```

| Variable             | Default        | Description                                       |
| -------------------- | -------------- | ------------------------------------------------- |
| `NETID`              | *(required)*   | Your cluster NetID                                |
| `DATASET`            | `pcam`         | `pcam`, `camelyon17`, or `embed`                  |
| `MAX_TRAIN_SAMPLES`  | `50000`        | Subsample training set to this size for TabPFN    |
| `EMBEDDING_DIR`      | `data/embeddings` | Embedding root (project format or partner HG layout) |
| `TABPFN_TOKEN`       | unset          | Prior Labs API token for headless cluster auth    |
| `TABPFN_TOKEN_FILE`  | unset          | Path to file containing token; safer than inline token |
| `CONDA_ENV`          | auto-detected  | Override conda env path                           |
| `PROJECT_ROOT`       | auto-detected  | Override project directory                        |

**Examples:**

```bash
# Run on PCam (default 50K train subsample)
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=pcam \
       scripts/exp4_tabpfn_baseline.sbatch

# Run on CAMELYON17 with larger train subsample
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=camelyon17,MAX_TRAIN_SAMPLES=100000 \
       scripts/exp4_tabpfn_baseline.sbatch

# Use partner embeddings (HG layout) for Exp4
sbatch --account=torch_pr_xxx_yyy \
       --export=ALL,NETID=ab1234,DATASET=camelyon17,EMBEDDING_DIR=/scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets \
       scripts/exp4_tabpfn_baseline.sbatch

# Headless TabPFN auth using a token file
printf '%s\n' '<your-prior-labs-token>' > /scratch/ab1234/tabpfn_token.txt
chmod 600 /scratch/ab1234/tabpfn_token.txt
sbatch --account=torch_pr_xxx_yyy \
       --export=ALL,NETID=ab1234,DATASET=pcam,TABPFN_TOKEN_FILE=/scratch/ab1234/tabpfn_token.txt \
       scripts/exp4_tabpfn_baseline.sbatch
```

**Outputs:** `data/exp4_<dataset>_results.json`, `data/exp4_<dataset>_calibration.png`,
`data/exp4_<dataset>_roc.png`

---

## Recommended submission order

Run all three datasets through the full pipeline:

```bash
# Step 1: Extract embeddings + train NN for each dataset
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=pcam       scripts/exp3_nn_baseline.sbatch
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=camelyon17  scripts/exp3_nn_baseline.sbatch
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=embed       scripts/exp3_nn_baseline.sbatch

# Step 2: After Exp3 jobs complete, run TabPFN on the same embeddings
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=pcam        scripts/exp4_tabpfn_baseline.sbatch
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=camelyon17  scripts/exp4_tabpfn_baseline.sbatch
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=embed       scripts/exp4_tabpfn_baseline.sbatch
```

Use `squeue -u $USER` to monitor job status and check `slurm_logs/` for output.

---

## Embedding formats supported by Exp3/Exp4

Both `exp3_nn_baseline.py` and `exp4_tabpfn_baseline.py` can load either format:

1) **Project-native format** (generated by `extract_embeddings.py`)

```text
<EMBEDDING_DIR>/pcam_train_embeddings.npy
<EMBEDDING_DIR>/pcam_train_labels.npy
<EMBEDDING_DIR>/pcam_val_embeddings.npy
<EMBEDDING_DIR>/pcam_val_labels.npy
<EMBEDDING_DIR>/pcam_test_embeddings.npy
<EMBEDDING_DIR>/pcam_test_labels.npy
```

2) **Partner HG layout**

```text
<EMBEDDING_DIR>/pcam-hg/train/embeddings/projected_512.npy
<EMBEDDING_DIR>/pcam-hg/train/embeddings/y_embeddings.npy
<EMBEDDING_DIR>/pcam-hg/valid/embeddings/projected_512.npy
<EMBEDDING_DIR>/pcam-hg/test/embeddings/projected_512.npy
```

Notes:
- For HG layout, labels are loaded from `y_embeddings.npy` if present, otherwise
  from `<split>/labels.csv`.
- `val` in code maps to `valid` in the HG directory naming.
- For Exp3 with partner embeddings, set `SKIP_EMBED=1` so the extraction stage is skipped.

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
| `exp3_nn_baseline.sbatch`             | SLURM template for NN baseline (1 GPU)               |
| `exp4_tabpfn_baseline.sbatch`         | SLURM template for TabPFN baseline (1 GPU)           |
