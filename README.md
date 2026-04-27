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

If PCam is already downloaded and you just want usable split folders with
decompressed HDF5 files, run:
```bash
python scripts/download_datasets.py \
  --datasets pcam \
  --root /scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets \
  --pcam-source existing \
  --prepare-pcam
```

That creates:
- `datasets/pcam/train/{x.h5,y.h5,meta.csv}`
- `datasets/pcam/valid/{x.h5,y.h5,meta.csv}`
- `datasets/pcam/test/{x.h5,y.h5,meta.csv}`

If you want actual image files under `train/images`, `valid/images`, and
`test/images`, run:
```bash
python scripts/download_datasets.py \
  --datasets pcam \
  --root /scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets \
  --pcam-source existing \
  --prepare-pcam \
  --export-pcam-images
```

That additionally creates:
- `datasets/pcam/train/images/*.png`
- `datasets/pcam/valid/images/*.png`
- `datasets/pcam/test/images/*.png`
- `datasets/pcam/{train,valid,test}/labels.csv`

There is also a shell wrapper for the same operation:
```bash
bash scripts/export_pcam_images.sh \
  /scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets
```

If you want a separate Hugging Face-based export that creates a new
`pcam-hg/` directory with images named `1.png`, `2.png`, ... for each split,
run:
```bash
bash scripts/create_pcam_hg.sh \
  /scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets/pcam-hg
```

This export is resumable: if it gets interrupted, rerunning the same command
continues from the next image recorded in each split's `labels.csv`.

If you want PCam directly from Hugging Face instead of the original archives, run:
```bash
python scripts/download_datasets.py \
  --datasets pcam \
  --root /scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets \
  --pcam-source hf
```

If you want a separate Hugging Face-based export for CAMELYON17-WILDS that
creates a new `camelyon17-hg/` directory with images named `1.png`, `2.png`,
... for each split, run:
```bash
bash scripts/create_camelyon17_hg.sh \
  /scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets/camelyon17-hg
```

This export uses Hugging Face's `train`, `validation`, and `test` splits and
writes them locally as `train/`, `valid/`, and `test/`. It is also resumable:
if interrupted, rerunning the same command continues from the next image
recorded in each split's `labels.csv`.

### Notes by dataset
- **PCam**: downloaded from official PatchCamelyon Google Drive files and MD5-verified.
  - You can also use `--pcam-source hf` to download the Hugging Face parquet
    shards from `1aurent/PatchCamelyon`.
  - You can use `--pcam-source existing --prepare-pcam` to reorganize already
    downloaded `*.h5.gz` files into `train/`, `valid/`, and `test/`.
  - You can use `--export-pcam-images` after preparation to materialize
    individual image files plus a `labels.csv` manifest for each split.
- **CAMELYON17-WILDS**: downloaded through `wilds.get_dataset(..., download=True)`.
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

## Experiment 3: U-Net Binary Classification

Experiment 3 adds a separate U-Net based classifier for pathology patches.
The architecture lives in:

- `src/models/unet_classifier.py`

The training entry point lives in:

- `experiments/exp3_unet_classifier.py`

This expects the Hugging Face image exports:

- `datasets/pcam-hg/{train,valid,test}`
- `datasets/camelyon17-hg/{train,valid,test}`

with each split containing `images/` and `labels.csv`.

### How to run locally

Make sure the conda env has PyTorch installed. On the cluster, that is the same
env created with:

```bash
NETID=ab1234 bash scripts/setup_env.sh --with-datasets
```

Train on PCam:

```bash
python experiments/exp3_unet_classifier.py \
  --dataset pcam \
  --data-root /scratch/ab1234/gpc-mcmc-cholesky-factorization/datasets \
  --output-root /scratch/ab1234/gpc-mcmc-cholesky-factorization/data/exp3_unet \
  --epochs 5 \
  --batch-size 32
```

Train on PCam with W&B logging:

```bash
export WANDB_API_KEY=<your_api_key>

python experiments/exp3_unet_classifier.py \
  --dataset pcam \
  --data-root /scratch/ab1234/gpc-mcmc-cholesky-factorization/datasets \
  --output-root /scratch/ab1234/gpc-mcmc-cholesky-factorization/data/exp3_unet \
  --epochs 5 \
  --batch-size 32 \
  --wandb \
  --wandb-project ML-Final_project \
  --wandb-entity a-salt
```

Train on CAMELYON17:

```bash
python experiments/exp3_unet_classifier.py \
  --dataset camelyon17 \
  --data-root /scratch/ab1234/gpc-mcmc-cholesky-factorization/datasets \
  --output-root /scratch/ab1234/gpc-mcmc-cholesky-factorization/data/exp3_unet \
  --epochs 5 \
  --batch-size 32
```

Outputs are saved under:

- `data/exp3_unet/<dataset>/best_model.pt`
- `data/exp3_unet/<dataset>/history.json`
- `data/exp3_unet/<dataset>/test_metrics.json`
- `data/exp3_unet/<dataset>/test_predictions.csv`

### Run with SLURM

Submit the provided GPU job script with:

```bash
sbatch --export=ALL,NETID=ab1234,DATASET=pcam,EPOCHS=5,BATCH_SIZE=32 \
  scripts/exp3_unet_training.sbatch
```

To switch datasets, set `DATASET=camelyon17`.

To enable W&B in SLURM, export the API key and set `USE_WANDB=1`:

```bash
export WANDB_API_KEY=<your_api_key>

sbatch --export=ALL,NETID=ab1234,DATASET=pcam,EPOCHS=5,BATCH_SIZE=32,USE_WANDB=1,WANDB_PROJECT=ML-Final_project,WANDB_ENTITY=a-salt \
  scripts/exp3_unet_training.sbatch
```

## Phikon Embeddings

If you want per-image foundation-model embeddings for the exported
`pcam-hg` and `camelyon17-hg` splits, use:

Make sure the active env has `torch`, `torchvision`, and `transformers`
installed together. On the cluster, the same `--with-datasets` env plus
`pip install -r requirements.txt` is enough.

```bash
python scripts/create_phikon_embeddings.py \
  --dataset pcam \
  --data-root /scratch/ab1234/gpc-mcmc-cholesky-factorization/datasets \
  --device cuda
```

If you want a smaller projected embedding size such as `512`, add:

```bash
python scripts/create_phikon_embeddings.py \
  --dataset pcam \
  --data-root /scratch/ab1234/gpc-mcmc-cholesky-factorization/datasets \
  --device cuda \
  --project-dim 512
```

For CAMELYON17:

```bash
python scripts/create_phikon_embeddings.py \
  --dataset camelyon17 \
  --data-root /scratch/ab1234/gpc-mcmc-cholesky-factorization/datasets \
  --device cuda
```

To run both datasets:

```bash
python scripts/create_phikon_embeddings.py \
  --dataset all \
  --data-root /scratch/ab1234/gpc-mcmc-cholesky-factorization/datasets \
  --device cuda
```

This adds an `embeddings/` directory inside each split:

- `train/images/`
- `train/labels.csv`
- `train/embeddings/embeddings.npy`
- `train/embeddings/metadata.json`
- `train/embeddings/progress.json`

and likewise for `valid/` and `test/`.

If `--project-dim 512` is used, each split also gets:

- `train/embeddings/projected_512.npy`
- `train/embeddings/projected_512_metadata.json`
- `train/embeddings/projected_512_progress.json`

and the dataset root gets:

- `embedding_projection/pca_512.npz`
- `embedding_projection/pca_512_metadata.json`

Alignment rule:

- row `i` of `train/embeddings/embeddings.npy`
- row `i` of `train/labels.csv` excluding the header
- the image path from that same `labels.csv` row

all refer to the same example.

The extractor is resumable. If interrupted, rerunning the same command resumes
from the last saved row recorded in `progress.json`.

By default it also prints one line per saved image or projected row. Use
`--no-log-each-image` if you want quieter output.

By default the script checkpoints every `100` rows. That means it flushes the
current `.npy` file to disk and updates `progress.json` every 100 rows, which is
faster than checkpointing every single row. If a job is interrupted, at most
the last 100 rows may need to be recomputed. You can change this with
`--checkpoint-every N`.

The optional size reduction is a linear PCA projection fit on the train split
and then reused for valid and test, so all splits remain in the same projected
feature space.

## Label Arrays

If you want an aligned label array for each exported split, use:

```bash
python scripts/create_label_arrays.py \
  --dataset camelyon17 \
  --data-root /scratch/ab1234/gpc-mcmc-cholesky-factorization/datasets
```

This adds to each split:

- `train/embeddings/y_embeddings.npy`
- `train/embeddings/y_embeddings_metadata.json`

and likewise for `valid/` and `test/`.

`y_embeddings.npy` has shape `(n, 1)`, where row `i` matches row `i` of
`labels.csv` excluding the header and therefore the image path from that same
row.
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
