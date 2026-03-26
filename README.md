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

### Notes by dataset
- **PCam**: downloaded from official PatchCamelyon Google Drive files and MD5-verified.
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
