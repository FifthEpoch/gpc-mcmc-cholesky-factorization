# my_cholesky_project

Experiment code for benchmarking RPCholesky variants for scalable kernel matrix approximation.

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
