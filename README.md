# Scalable Gaussian Process Classification via RPCholesky

This project investigates whether **Randomly Pivoted Cholesky (RPCholesky)** low-rank
approximations can make **Gaussian Process Classification (GPC)** practical for
large-scale medical imaging tasks, while preserving the calibrated uncertainty
estimates that GPs are valued for.

Standard GPC requires inverting an \(N \times N\) kernel matrix, which is
\(O(N^3)\) and infeasible for datasets with hundreds of thousands of samples.
RPCholesky (Chen et al., 2023) provides a rank-\(k\) Nystr\"{o}m approximation
that reduces this to \(O(Nk^2)\), but it is an open question whether the
approximation degrades calibration or predictive performance on real clinical data.

We benchmark RPCholesky-accelerated GPC against deterministic baselines on three
histopathology / medical imaging datasets:

- **PatchCamelyon (PCam)** -- 327K lymph node patches, binary metastasis detection
- **CAMELYON17-WILDS** -- 455K patches from 5 hospitals, with distribution shift
- **EMBED** -- mammography screening dataset (access-gated)

## Experiments

| # | Experiment | Question |
|---|-----------|----------|
| 0 | RPCholesky algorithm verification | Do Block and Accelerated RPCholesky match the approximation quality of Basic RPCholesky while being faster? |
| 1 | MCMC-based GPC | Can RPCholesky + MCMC scale GPC to large N while maintaining calibration? |
| 3 | Deterministic NN baseline | How does a simple MLP on frozen DenseNet-121 embeddings perform on predictive metrics and calibration? |
| 4 | TabPFN baseline | How does a tabular foundation model (TabPFN) compare on the same frozen embeddings? |

Experiments 3 and 4 serve as **baselines**: if the GP achieves comparable AUROC
but better-calibrated probabilities (lower ECE, lower Brier score) and fewer
false negatives, that supports the thesis that Bayesian uncertainty from
RPCholesky-GPC is worth the extra compute.

## Environment setup (cluster)

On this cluster, `/home` has a very small disk quota. All conda environments,
packages, and caches are stored under `/scratch/<NETID>` instead.

### Step 1: Set scratch environment variables

```bash
NETID=ab1234 source scripts/set_scratch_env.sh
```

This creates the directory structure under `/scratch/<NETID>` and exports
`CONDA_ENVS_DIRS`, `CONDA_PKGS_DIRS`, `PIP_CACHE_DIR`, `HF_HOME`, `TMPDIR`,
etc. so nothing lands in `/home`.

### Step 2: Create the conda environment

```bash
NETID=ab1234 bash scripts/setup_env.sh
```

This creates a conda env at `/scratch/<NETID>/conda-envs/gpc` with Python 3.12
and installs all base dependencies (`numpy`, `scipy`, `scikit-learn`, `emcee`,
`matplotlib`, `tqdm`, `gdown`).

To also install dataset download dependencies (WILDS + PyTorch):

```bash
NETID=ab1234 bash scripts/setup_env.sh --with-datasets
```

### Step 3: Submit a SLURM job

```bash
sbatch --export=ALL,NETID=ab1234 scripts/exp0_algorithm_verification.sbatch
```

The sbatch template automatically sources `set_scratch_env.sh`, activates the
conda env, and handles the critical `unset PYTHONHOME PYTHONPATH` step required
on this cluster.

## Experiment 0: RPCholesky scaling benchmark

### Objective
Evaluate and compare three Nyström/Cholesky approximation methods:
- **Basic RPCholesky** (sequential randomized pivots)
- **Block RPCholesky** (block pivots, block size `b=120`)
- **Accelerated RPCholesky** (rejection-based accelerated variant, block size `b=120`)

The benchmark studies:
1. Approximation error vs rank `k` (fixed `N`)
2. Runtime vs rank `k` (fixed `N`)
3. Runtime vs `N` (with rank ratio held approximately constant)

---

### Data / kernel construction
We use a synthetic Gaussian kernel matrix:
- Sample points `X ~ Uniform([0,1]^d)` with `d=10`
- Build `A = KernelMatrix(X, kernel="gaussian", bandwidth=1.0)`

We intentionally avoid dataset-specific toy shapes (`smile`, `expspiral`, `outliers`) to get a neutral scaling benchmark.

---

### Sweep configuration
- **Trials:** `5`
- **Fixed-N sweeps:** `N = 20000`
- **k values:** `k = [10, 100, 200, 300, 400, 500, 600]`
- **N sweep values:** `N in [1000, 5000, 10000, 50000]`
- **N-sweep rank rule:** `k = max(50, N // 50)` (about 2% rank ratio)
- **Basic skip rule:** Basic RPCholesky is skipped when `N > 10000`

Error metric:
\[
\text{relative trace error} = \frac{\mathrm{trace}(A) - \mathrm{trace}(\hat A)}{\mathrm{trace}(A)}
\]

---

### How to run

```bash
python experiments/exp0_algorithm_verification.py
```

This writes:
- `data/exp0_results.mat`
- `data/exp0_error_vs_k.png`
- `data/exp0_time_vs_k.png`
- `data/exp0_time_vs_N.png`

## Verify Experiment 0 outputs

Use this quick checklist against the generated plots:
- `exp0_error_vs_k.png`: relative trace error should decrease as `k` increases.
- `exp0_time_vs_k.png`: runtime should increase with `k`; Basic should be much slower than Block/Accel.
- `exp0_time_vs_N.png`: runtime should increase with `N`; Basic is skipped for `N > 10000`.

## Download datasets for later experiments

Use:
```bash
python scripts/download_datasets.py --datasets all --root datasets
```

You can also fetch datasets individually:
```bash
python scripts/download_datasets.py --datasets pcam --root datasets
python scripts/download_datasets.py --datasets camelyon17 --root datasets
python scripts/download_datasets.py --datasets embed --root datasets
```

### Notes by dataset
- **PCam**: downloaded from HuggingFace mirror ([1aurent/PatchCamelyon](https://huggingface.co/datasets/1aurent/PatchCamelyon)).
- **CAMELYON17-WILDS**: downloaded from HuggingFace mirror ([wltjr1007/Camelyon17-WILDS](https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS)). The original CodaLab source used by the `wilds` library has been broken since June 2025.
- **EMBED**: requires approval first. Submit access request:
  - [Access request form](https://forms.gle/6YVFKTz7ucEJKEWw8)
  - [Documentation](https://docs.hitilab.com/)
  - [Data use agreement](https://github.com/Emory-HITI/EMBED_Open_Data/blob/main/EMBED_license.md)

After EMBED approval, sync from your approved S3 path:
```bash
python scripts/download_datasets.py \
  --datasets embed \
  --root datasets \
  --embed-s3-uri s3://<approved-bucket-or-prefix>
```

## Experiment 3: Deterministic Neural Network Baseline

### Overview

Trains a 2-layer MLP classifier on frozen DenseNet-121 embeddings, evaluating
predictive performance (AUROC, AUPRC), calibration (ECE, Brier score),
sensitivity, and false-negative rate on PCam, CAMELYON17-WILDS, and EMBED.

### Pipeline

1. **Extract embeddings** -- pass raw images through frozen DenseNet-121 and
   save 1024-dim feature vectors as `.npy` files.
2. **Train + evaluate** -- train the MLP head on the embeddings, then compute
   all metrics on the held-out test split.

### Running locally

```bash
# Step 1: extract embeddings (needs GPU for reasonable speed)
python experiments/extract_embeddings.py \
    --dataset pcam --data-root datasets --device cuda

# Step 2: train & evaluate
python experiments/exp3_nn_baseline.py \
    --dataset pcam --embedding-dir data/embeddings --device cuda
```

Replace `pcam` with `camelyon17` or `embed` as needed.

### Running on the cluster (SLURM)

```bash
sbatch --export=ALL,NETID=ab1234,DATASET=pcam \
       scripts/exp3_nn_baseline.sbatch
```

Optional overrides:

| Variable      | Default       | Description                                |
| ------------- | ------------- | ------------------------------------------ |
| `DATASET`     | `pcam`        | `pcam`, `camelyon17`, or `embed`           |
| `ENCODER`     | `densenet121` | `densenet121` or `dinov2_vitl14`           |
| `SKIP_EMBED`  | `0`           | Set to `1` to reuse existing embeddings    |

### Outputs

All outputs are written to `data/`:

- `exp3_<dataset>_results.json` -- full metric dictionary
- `exp3_<dataset>_calibration.png` -- reliability diagram
- `exp3_<dataset>_roc.png` -- ROC curve
- `exp3_camelyon17_per_hospital.json` -- per-hospital breakdown (CAMELYON17 only)

## Experiment 4: TabPFN Tabular Model Baseline

### Overview

Uses [TabPFN](https://github.com/PriorLabs/TabPFN) (Hollmann et al., 2025),
a pre-trained transformer foundation model for tabular data, as a classifier
on the same frozen embeddings used in Experiment 3. TabPFN performs in-context
learning in a single forward pass -- no gradient-based training loop is needed.

TabPFN works best on datasets with up to ~50k samples. For larger training sets
the script automatically subsamples (stratified) to `--max-train-samples`.

### Prerequisites

Embeddings must already exist in `data/embeddings/` (produced by
`experiments/extract_embeddings.py` from Experiment 3).

### Running locally

```bash
python experiments/exp4_tabpfn_baseline.py \
    --dataset pcam --embedding-dir data/embeddings --device cuda
```

Replace `pcam` with `camelyon17` or `embed` as needed.

### Running on the cluster (SLURM)

```bash
sbatch --export=ALL,NETID=ab1234,DATASET=pcam \
       scripts/exp4_tabpfn_baseline.sbatch
```

Optional overrides:

| Variable              | Default | Description                                       |
| --------------------- | ------- | ------------------------------------------------- |
| `DATASET`             | `pcam`  | `pcam`, `camelyon17`, or `embed`                  |
| `MAX_TRAIN_SAMPLES`   | `50000` | Subsample training set to this size for TabPFN    |

### Outputs

All outputs are written to `data/`:

- `exp4_<dataset>_results.json` -- full metric dictionary
- `exp4_<dataset>_calibration.png` -- reliability diagram
- `exp4_<dataset>_roc.png` -- ROC curve
- `exp4_camelyon17_per_hospital.json` -- per-hospital breakdown (CAMELYON17 only)
