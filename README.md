# my_cholesky_project

Experiment code for benchmarking RPCholesky variants for scalable kernel matrix approximation.

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
