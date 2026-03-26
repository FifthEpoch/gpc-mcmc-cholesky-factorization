# my_cholesky_project

Experiment code for benchmarking RPCholesky variants for scalable kernel matrix approximation.

## Environment setup

### Quick start (base environment)
```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

### Include dataset tooling dependencies
```bash
bash scripts/setup_env.sh --with-datasets
source .venv/bin/activate
```

This installs a local virtual environment and dependencies required for:
- Experiment 0 benchmarking
- Dataset download helper script for later experiments

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
