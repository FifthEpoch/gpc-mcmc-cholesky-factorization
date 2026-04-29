"""
Real-data low-rank GP classification predictive sampling using:
1) precomputed RP-Cholesky factor F for K_XX,
2) HMC in the non-centered training coordinates f = F nu,
3) pivot points from RP-Cholesky as inducing points for prediction,
4) per-point marginal predictive sampling for classification metrics,
5) optional parallel CPU or torch GPU predictive sampling.

Example:
    python experiments/exp2_rose_exp1_metrics_parallel.py \
        --factor-dir data/rpchol_test \
        --test-embeddings datasets/pcam_test_embedding.npy \
        --test-labels datasets/pcam_test_label.npy \
        --k 100 \
        --predictive-backend auto \
        --prediction-n-jobs 8
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import json
import os
import sys
import time
from pathlib import Path

import emcee
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import expit

# Allow direct script execution without package install.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from my_cholesky.kernels import GaussianKernel_mtx
from predictive_metrics import (
    evaluate_binary_probabilistic_predictions,
    print_metric_table,
    summarize_predictive_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-data low-rank HMC GP classification prediction."
    )
    parser.add_argument(
        "--factor-dir",
        type=str,
        required=True,
        help="Directory containing factor_k{k}.npy and labels.npy.",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to training embeddings .npy. If omitted, loads from factor_dir/embeddings.npy.",
    )
    parser.add_argument(
        "--test-embeddings",
        type=str,
        required=True,
        help="Test embeddings .npy file of shape (n_test, d).",
    )
    parser.add_argument(
        "--test-labels",
        type=str,
        required=True,
        help="Test labels .npy file of shape (n_test,).",
    )
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help=(
            "Gaussian kernel bandwidth. If omitted, loads bandwidth_actual from "
            "factor_dir/summary.json (the value used when the factor was built)."
        ),
    )
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-warmup", type=int, default=200)
    parser.add_argument("--n-leapfrog", type=int, default=12)
    parser.add_argument(
        "--prediction-batch-size",
        type=int,
        default=512,
        help="Number of posterior samples to process per batched predictive solve.",
    )
    parser.add_argument(
        "--predictive-backend",
        choices=("auto", "cpu", "cpu-parallel", "gpu"),
        default="auto",
        help=(
            "Backend for predictive sampling. 'auto' uses GPU if available, "
            "otherwise CPU parallelism when --prediction-n-jobs > 1."
        ),
    )
    parser.add_argument(
        "--prediction-n-jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help=(
            "Number of CPU worker threads for --predictive-backend cpu-parallel. "
            "Set BLAS thread env vars to 1 on HPC to avoid oversubscription."
        ),
    )
    parser.add_argument(
        "--gpu-device",
        type=str,
        default="auto",
        help="Torch device for GPU predictive sampling: auto, cuda, cuda:0, mps, or cpu.",
    )
    parser.add_argument("--hmc-step-constant", type=float, default=0.05)
    parser.add_argument("--hmc-target-accept", type=float, default=0.8)
    parser.add_argument(
        "--no-adapt-step-size",
        action="store_false",
        dest="adapt_step_size",
        help="Disable HMC dual averaging step-size adaptation during warmup.",
    )
    parser.add_argument(
        "--subsample-test",
        type=int,
        default=None,
        help="Randomly subsample test data to this size for quick tests.",
    )
    parser.add_argument(
        "--stratified-subsample-test",
        action="store_true",
        help=(
            "Use the same stratified test subsampling policy as exp1 full HMC "
            "(seed + 1)."
        ),
    )
    parser.add_argument(
        "--standardize-from-factor",
        action="store_true",
        help="Apply factor_dir/train_mean.npy and train_std.npy to test embeddings.",
    )
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def log_posterior(nu, factor, y):
    """
    Approximate low-rank posterior with f = factor @ nu and nu ~ N(0, I).
    """
    f = factor @ nu
    p = expit(f)
    return log_posterior_from_cached_forward(nu, p, y)


def log_posterior_from_cached_forward(nu, p, y):
    eps = 1e-10
    log_lik = np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    log_prior = -0.5 * np.dot(nu, nu)
    return float(log_lik + log_prior)


def grad_log_posterior(nu, factor, y):
    grad, _ = grad_log_posterior_with_cache(nu, factor, y)
    return grad


def grad_log_posterior_with_cache(nu, factor, y):
    f = factor @ nu
    p = expit(f)
    return factor.T @ (y - p) - nu, p


def compute_tau_emcee(chain):
    """Estimate integrated autocorrelation time with emcee's implementation."""
    try:
        tau = emcee.autocorr.integrated_time(chain, quiet=True)
    except Exception as err:
        print(f"WARNING: emcee autocorr.integrated_time failed: {err}")
        return float("nan")

    tau = np.asarray(tau, dtype=float).reshape(-1)
    if tau.size == 0:
        return float("nan")
    return float(np.nanmean(tau))


def sample_predictive_probabilities_pivots(
    F,
    X_train,
    X_test,
    pivot_indices,
    K_pp,
    nu_samples,
    bandwidth,
    batch_size=512,
    seed=0,
):
    """
    Option B predictive sampler using RPCholesky pivots as inducing points.

    This computes per-test-point marginal predictive distributions without
    building n_test x n_train or n_test x n_test kernel matrices. The joint
    predictive covariance is not sampled; add a pivot-based joint sampler later
    if a downstream diagnostic needs correlations across test points.
    """
    rng = np.random.default_rng(seed)

    pivot_indices = np.asarray(pivot_indices, dtype=np.int64)
    X_pivots = X_train[pivot_indices].astype(np.float32, copy=False)
    F_pivots = F[pivot_indices, :].astype(np.float32, copy=False)

    K_pp = np.asarray(K_pp, dtype=np.float64)
    # Numerical jitter for Cholesky stability. This is NOT a model nugget --
    # the model is K_PP exactly. The jitter only protects the float64 Cholesky
    # from breakdown when K_PP is borderline-singular due to highly correlated pivots.
    k_size = K_pp.shape[0]
    jitter = 1e-10 * np.trace(K_pp) / k_size
    K_pp_jittered = K_pp + jitter * np.eye(k_size)
    L_pp = np.linalg.cholesky(K_pp_jittered)

    K_test_pivots = GaussianKernel_mtx(X_test, X_pivots, bandwidth=bandwidth).astype(
        np.float32,
        copy=False,
    )

    K_tp_t = K_test_pivots.T.astype(np.float64, copy=False)
    V = solve_triangular(L_pp, K_tp_t, lower=True, check_finite=False)
    V = solve_triangular(L_pp.T, V, lower=False, check_finite=False)
    var_f64 = 1.0 - np.sum(
        K_test_pivots.astype(np.float64, copy=False)
        * V.T.astype(np.float64, copy=False),
        axis=1,
    )
    var = np.clip(var_f64, 1e-8, None).astype(np.float32, copy=False)
    std = np.sqrt(var, dtype=np.float32)

    n_samples = nu_samples.shape[0]
    n_test = X_test.shape[0]
    latent_samples = np.zeros((n_samples, n_test), dtype=np.float32)
    p_samples = np.zeros((n_samples, n_test), dtype=np.float32)

    batch_size = max(1, int(batch_size))
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        nu_batch = nu_samples[start:end].T
        f_pivots = F_pivots @ nu_batch
        rhs = f_pivots.astype(np.float64, copy=False)
        alpha = solve_triangular(L_pp, rhs, lower=True, check_finite=False)
        alpha = solve_triangular(L_pp.T, alpha, lower=False, check_finite=False)
        alpha = alpha.astype(np.float32, copy=False)

        mean = K_test_pivots @ alpha
        xi = rng.standard_normal(mean.shape).astype(np.float32, copy=False)
        f_star = mean + std[:, None] * xi
        latent_samples[start:end] = f_star.T
        p_samples[start:end] = expit(f_star).T.astype(np.float32, copy=False)

    return {
        "p_samples": p_samples,
        "latent_samples": latent_samples,
        "predictive_cov_approx_factor": None,
        "backend_used": "cpu",
    }


def _prepare_pivot_predictive_terms(
    F,
    X_train,
    X_test,
    pivot_indices,
    K_pp,
    bandwidth,
):
    pivot_indices = np.asarray(pivot_indices, dtype=np.int64)
    X_pivots = X_train[pivot_indices].astype(np.float32, copy=False)
    F_pivots = F[pivot_indices, :].astype(np.float32, copy=False)

    K_pp = np.asarray(K_pp, dtype=np.float64)
    k_size = K_pp.shape[0]
    jitter = 1e-10 * np.trace(K_pp) / k_size
    K_pp_jittered = K_pp + jitter * np.eye(k_size)
    L_pp = np.linalg.cholesky(K_pp_jittered)

    K_test_pivots = GaussianKernel_mtx(X_test, X_pivots, bandwidth=bandwidth).astype(
        np.float32,
        copy=False,
    )

    K_tp_t = K_test_pivots.T.astype(np.float64, copy=False)
    V = solve_triangular(L_pp, K_tp_t, lower=True, check_finite=False)
    V = solve_triangular(L_pp.T, V, lower=False, check_finite=False)
    var_f64 = 1.0 - np.sum(
        K_test_pivots.astype(np.float64, copy=False)
        * V.T.astype(np.float64, copy=False),
        axis=1,
    )
    var = np.clip(var_f64, 1e-8, None).astype(np.float32, copy=False)
    std = np.sqrt(var, dtype=np.float32)

    return F_pivots, K_test_pivots, L_pp, std


def _process_predictive_batch_cpu(
    start,
    end,
    nu_samples,
    F_pivots,
    K_test_pivots,
    L_pp,
    std,
    seed,
):
    rng = np.random.default_rng(seed + start)
    nu_batch = nu_samples[start:end].T
    f_pivots = F_pivots @ nu_batch
    rhs = f_pivots.astype(np.float64, copy=False)
    alpha = solve_triangular(L_pp, rhs, lower=True, check_finite=False)
    alpha = solve_triangular(L_pp.T, alpha, lower=False, check_finite=False)
    alpha = alpha.astype(np.float32, copy=False)

    mean = K_test_pivots @ alpha
    xi = rng.standard_normal(mean.shape).astype(np.float32, copy=False)
    f_star = mean + std[:, None] * xi
    latent_batch = f_star.T
    p_batch = expit(f_star).T.astype(np.float32, copy=False)
    return start, end, latent_batch, p_batch


def sample_predictive_probabilities_pivots_parallel_cpu(
    F,
    X_train,
    X_test,
    pivot_indices,
    K_pp,
    nu_samples,
    bandwidth,
    batch_size=512,
    n_jobs=4,
    seed=0,
):
    """
    CPU-parallel version of the pivot predictive sampler.

    Each posterior-sample batch is independent after K_test_pivots, L_pp, and
    std are computed, so a thread pool can evaluate batches concurrently. BLAS
    calls release the GIL; on clusters, use OMP/MKL/OPENBLAS_NUM_THREADS=1 when
    n_jobs > 1 to avoid nested thread oversubscription.
    """
    F_pivots, K_test_pivots, L_pp, std = _prepare_pivot_predictive_terms(
        F=F,
        X_train=X_train,
        X_test=X_test,
        pivot_indices=pivot_indices,
        K_pp=K_pp,
        bandwidth=bandwidth,
    )

    n_samples = nu_samples.shape[0]
    n_test = X_test.shape[0]
    latent_samples = np.zeros((n_samples, n_test), dtype=np.float32)
    p_samples = np.zeros((n_samples, n_test), dtype=np.float32)

    batch_size = max(1, int(batch_size))
    n_jobs = max(1, int(n_jobs))
    batch_ranges = [
        (start, min(start + batch_size, n_samples))
        for start in range(0, n_samples, batch_size)
    ]

    if n_jobs == 1 or len(batch_ranges) == 1:
        for start, end in batch_ranges:
            _, _, latent_batch, p_batch = _process_predictive_batch_cpu(
                start,
                end,
                nu_samples,
                F_pivots,
                K_test_pivots,
                L_pp,
                std,
                seed,
            )
            latent_samples[start:end] = latent_batch
            p_samples[start:end] = p_batch
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _process_predictive_batch_cpu,
                    start,
                    end,
                    nu_samples,
                    F_pivots,
                    K_test_pivots,
                    L_pp,
                    std,
                    seed,
                )
                for start, end in batch_ranges
            ]
            for future in as_completed(futures):
                start, end, latent_batch, p_batch = future.result()
                latent_samples[start:end] = latent_batch
                p_samples[start:end] = p_batch

    return {
        "p_samples": p_samples,
        "latent_samples": latent_samples,
        "predictive_cov_approx_factor": None,
        "backend_used": "cpu-parallel",
    }


def _resolve_torch_device(torch, device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _torch_gaussian_kernel_mtx(torch, X, Y, bandwidth):
    x_norm = (X * X).sum(dim=1, keepdim=True)
    y_norm = (Y * Y).sum(dim=1, keepdim=True).T
    sqdist = torch.clamp(x_norm + y_norm - 2.0 * (X @ Y.T), min=0.0)
    return torch.exp(-0.5 * sqdist / (bandwidth * bandwidth))


def sample_predictive_probabilities_pivots_gpu(
    F,
    X_train,
    X_test,
    pivot_indices,
    K_pp,
    nu_samples,
    bandwidth,
    batch_size=512,
    seed=0,
    device_arg="auto",
):
    """
    Torch GPU predictive sampler.

    This accelerates the same independent posterior-sample batches on CUDA/MPS
    when available. If device_arg resolves to CPU, it still runs through torch
    and is mainly useful for debugging the backend.
    """
    try:
        torch = importlib.import_module("torch")
    except ImportError as err:
        raise RuntimeError("GPU predictive backend requires PyTorch.") from err

    device = _resolve_torch_device(torch, device_arg)
    dtype = torch.float32

    pivot_indices = np.asarray(pivot_indices, dtype=np.int64)
    X_pivots_np = X_train[pivot_indices].astype(np.float32, copy=False)
    F_pivots_np = F[pivot_indices, :].astype(np.float32, copy=False)

    X_test_t = torch.as_tensor(X_test, dtype=dtype, device=device)
    X_pivots_t = torch.as_tensor(X_pivots_np, dtype=dtype, device=device)
    F_pivots_t = torch.as_tensor(F_pivots_np, dtype=dtype, device=device)
    K_pp_t = torch.as_tensor(K_pp, dtype=dtype, device=device)
    nu_samples_t = torch.as_tensor(nu_samples, dtype=dtype, device=device)

    k_size = K_pp_t.shape[0]
    jitter = 1e-10 * torch.trace(K_pp_t) / k_size
    K_pp_jittered = K_pp_t + jitter * torch.eye(k_size, dtype=dtype, device=device)
    L_pp = torch.linalg.cholesky(K_pp_jittered)

    K_test_pivots = _torch_gaussian_kernel_mtx(
        torch,
        X_test_t,
        X_pivots_t,
        float(bandwidth),
    )
    V = torch.linalg.solve_triangular(L_pp, K_test_pivots.T, upper=False)
    V = torch.linalg.solve_triangular(L_pp.T, V, upper=True)
    var = torch.clamp(1.0 - torch.sum(K_test_pivots * V.T, dim=1), min=1e-8)
    std = torch.sqrt(var)

    n_samples = nu_samples.shape[0]
    n_test = X_test.shape[0]
    latent_samples = np.zeros((n_samples, n_test), dtype=np.float32)
    p_samples = np.zeros((n_samples, n_test), dtype=np.float32)

    batch_size = max(1, int(batch_size))
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    except RuntimeError:
        generator = None
        torch.manual_seed(int(seed))

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            nu_batch = nu_samples_t[start:end].T
            f_pivots = F_pivots_t @ nu_batch
            alpha = torch.linalg.solve_triangular(L_pp, f_pivots, upper=False)
            alpha = torch.linalg.solve_triangular(L_pp.T, alpha, upper=True)
            mean = K_test_pivots @ alpha
            randn_kwargs = {"dtype": dtype, "device": device}
            if generator is not None:
                randn_kwargs["generator"] = generator
            xi = torch.randn(mean.shape, **randn_kwargs)
            f_star = mean + std[:, None] * xi
            latent_batch = f_star.T.detach().cpu().numpy().astype(np.float32, copy=False)
            p_batch = torch.sigmoid(f_star).T.detach().cpu().numpy().astype(
                np.float32,
                copy=False,
            )
            latent_samples[start:end] = latent_batch
            p_samples[start:end] = p_batch

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return {
        "p_samples": p_samples,
        "latent_samples": latent_samples,
        "predictive_cov_approx_factor": None,
        "backend_used": f"gpu:{device}",
    }


def choose_predictive_backend(args):
    if args.predictive_backend != "auto":
        return args.predictive_backend
    try:
        torch = importlib.import_module("torch")
        device = _resolve_torch_device(torch, args.gpu_device)
        if device.type in ("cuda", "mps"):
            return "gpu"
    except Exception:
        pass
    if args.prediction_n_jobs > 1:
        return "cpu-parallel"
    return "cpu"


def run_hmc(
    factor,
    y,
    n_samples,
    n_warmup,
    seed,
    initial_step_size,
    n_leapfrog,
    target_accept=0.8,
    adapt_step_size=True,
):
    rng = np.random.default_rng(seed)
    if n_leapfrog < 1:
        raise ValueError("n_leapfrog must be at least 1")

    dim = factor.shape[1]
    total_steps = n_warmup + n_samples

    nu = np.zeros(dim, dtype=np.float64)
    logp = log_posterior(nu, factor, y)

    nu_samples = np.zeros((n_samples, dim), dtype=np.float64)
    logp_trace = np.zeros((n_samples,), dtype=np.float64)
    step_times = np.zeros(total_steps, dtype=np.float64)
    accepts = np.zeros(total_steps, dtype=bool)
    post_idx = 0

    step_size = initial_step_size
    log_step = np.log(initial_step_size)
    mu = np.log(10 * initial_step_size)
    log_step_bar = 0.0
    H_bar = 0.0
    gamma, t0_dual, kappa = 0.05, 10, 0.75

    for i in range(total_steps):
        t0 = time.perf_counter()
        if adapt_step_size and n_warmup > 0:
            if i < n_warmup:
                step_size = np.exp(log_step)
            elif i == n_warmup:
                step_size = np.exp(log_step_bar)

        current_nu = nu.copy()
        current_logp = logp
        momentum = rng.standard_normal(dim)

        grad, _ = grad_log_posterior_with_cache(current_nu, factor, y)
        proposal_nu = current_nu.copy()
        proposal_p = momentum + 0.5 * step_size * grad

        for leapfrog_idx in range(n_leapfrog):
            proposal_nu = proposal_nu + step_size * proposal_p
            grad, proposal_prob = grad_log_posterior_with_cache(
                proposal_nu,
                factor,
                y,
            )
            if leapfrog_idx != n_leapfrog - 1:
                proposal_p = proposal_p + step_size * grad

        proposal_p = proposal_p + 0.5 * step_size * grad
        proposal_p = -proposal_p

        proposal_logp = log_posterior_from_cached_forward(
            proposal_nu,
            proposal_prob,
            y,
        )
        current_h = -current_logp + 0.5 * np.dot(momentum, momentum)
        proposal_h = -proposal_logp + 0.5 * np.dot(proposal_p, proposal_p)
        log_accept = current_h - proposal_h
        accept_prob = 1.0 if log_accept >= 0.0 else float(np.exp(log_accept))

        if np.log(rng.random()) < log_accept:
            nu = proposal_nu
            logp = proposal_logp
            accepts[i] = True

        if adapt_step_size and i < n_warmup:
            eta = 1.0 / (i + 1 + t0_dual)
            H_bar = (1 - eta) * H_bar + eta * (target_accept - accept_prob)
            log_step = mu - np.sqrt(i + 1) / gamma * H_bar
            eta_bar = (i + 1) ** (-kappa)
            log_step_bar = eta_bar * log_step + (1 - eta_bar) * log_step_bar

        step_times[i] = time.perf_counter() - t0
        if i >= n_warmup:
            nu_samples[post_idx] = nu
            logp_trace[post_idx] = logp
            post_idx += 1

    post = slice(n_warmup, total_steps)
    warmup = slice(0, n_warmup)
    warmup_time = float(np.sum(step_times[warmup]))
    sampling_time = float(np.sum(step_times[post]))
    return {
        "nu_samples": nu_samples,
        "logp_trace": logp_trace,
        "accept_rate": float(np.mean(accepts[post])),
        "step_size": float(step_size),
        "per_step_time": float(np.mean(step_times[post])),
        # Standard timing convention: total_mcmc_time is post-warmup sampling only.
        "warmup_time": warmup_time,
        "sampling_time": sampling_time,
        "total_mcmc_time": sampling_time,
        "total_sampler_time": warmup_time + sampling_time,
    }


def _subsample_rows(rng, size, *arrays):
    n_rows = arrays[0].shape[0]
    if size is None or size >= n_rows:
        return arrays
    idx = np.sort(rng.choice(n_rows, size=size, replace=False))
    return tuple(arr[idx] for arr in arrays)


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_n: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same stratified random subsample policy as exp1 full-HMC embeddings."""
    if max_n is None or len(y) <= max_n:
        idx = np.arange(len(y))
        return X, y, idx

    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    frac_pos = len(pos) / len(y)
    n_pos = max(1, int(round(max_n * frac_pos)))
    n_neg = max_n - n_pos

    idx = np.concatenate(
        [
            rng.choice(pos, size=min(n_pos, len(pos)), replace=False),
            rng.choice(neg, size=min(n_neg, len(neg)), replace=False),
        ]
    )
    rng.shuffle(idx)
    return X[idx], y[idx], idx


def main():
    wall_t0 = time.perf_counter()
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_load_start = time.perf_counter()
    factor_dir = Path(args.factor_dir)
    if args.bandwidth is None:
        summary_path = factor_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"No --bandwidth given and {summary_path} not found. "
                "Either pass --bandwidth explicitly, or re-run "
                "exp_rpcholesky_embeddings.py to regenerate the factor directory "
                "with summary.json included."
            )
        with open(summary_path, "r") as f:
            summary = json.load(f)
        if "bandwidth_actual" not in summary:
            raise KeyError(
                f"{summary_path} does not contain 'bandwidth_actual'. "
                "Pass --bandwidth explicitly."
            )
        bandwidth = float(summary["bandwidth_actual"])
        print(f"Loaded bandwidth from {summary_path}: {bandwidth:.6f}")
    else:
        bandwidth = float(args.bandwidth)
        print(f"Using --bandwidth: {bandwidth:.6f}")

    if args.embeddings is None:
        emb_path = factor_dir / "embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(
                f"No --embeddings given and {emb_path} not found. "
                "Re-run exp_rpcholesky_embeddings.py to regenerate the factor directory "
                "with embeddings.npy included, or pass --embeddings explicitly."
            )
    else:
        emb_path = Path(args.embeddings)
    X_train = np.load(emb_path).astype(np.float32, copy=False)
    print(f"Loaded training embeddings from: {emb_path}  shape={X_train.shape}")
    y_train = np.load(factor_dir / "labels.npy").astype(np.float32, copy=False).squeeze()
    X_test = np.load(args.test_embeddings).astype(np.float32, copy=False)
    y_test = np.load(args.test_labels).astype(np.float32, copy=False).squeeze()

    if args.standardize_from_factor:
        mean_path = factor_dir / "train_mean.npy"
        std_path = factor_dir / "train_std.npy"
        if not mean_path.exists() or not std_path.exists():
            raise FileNotFoundError(
                "--standardize-from-factor requires train_mean.npy and "
                f"train_std.npy under {factor_dir}."
            )
        train_mean = np.load(mean_path).astype(np.float32, copy=False)
        train_std = np.load(std_path).astype(np.float32, copy=False)
        X_test = ((X_test - train_mean) / train_std).astype(np.float32, copy=False)
        print(f"Standardized test embeddings using {mean_path} and {std_path}")

    F = np.load(factor_dir / f"factor_k{args.k}.npy").astype(np.float64, copy=False)
    pivots_path = factor_dir / f"pivots_k{args.k}.npy"
    if not pivots_path.exists():
        raise FileNotFoundError(
            f"{pivots_path} not found. Re-run exp_rpcholesky_embeddings.py "
            "to regenerate the factor directory with pivot indices included."
        )
    pivot_indices = np.load(pivots_path).astype(np.int64, copy=False)
    kpp_path = factor_dir / f"kernel_submatrix_k{args.k}.npy"
    if not kpp_path.exists():
        raise FileNotFoundError(
            f"{kpp_path} not found. Re-run exp_rpcholesky_embeddings.py "
            "to regenerate the factor directory with the pivot kernel submatrix included."
        )
    K_pivots = np.load(kpp_path).astype(np.float64, copy=False)
    data_loading_time = time.perf_counter() - data_load_start

    assert X_train.shape[0] == F.shape[0] == y_train.shape[0], (
        f"Misaligned training arrays: X_train={X_train.shape[0]}, "
        f"F={F.shape[0]}, y_train={y_train.shape[0]}. "
        f"All three must come from the same factor directory."
    )
    assert pivot_indices.shape[0] == F.shape[1], (
        f"Misaligned pivot indices: pivots={pivot_indices.shape[0]}, "
        f"F rank={F.shape[1]}."
    )
    assert K_pivots.shape == (F.shape[1], F.shape[1]), (
        f"Misaligned pivot kernel submatrix: K_pivots={K_pivots.shape}, "
        f"expected {(F.shape[1], F.shape[1])}."
    )
    F_P_check = F[pivot_indices, :]
    recon = F_P_check @ F_P_check.T
    abs_err = float(np.max(np.abs(recon - K_pivots)))
    rel_err = abs_err / float(np.max(np.abs(K_pivots)) + 1e-30)
    print(f"Pivot consistency check: |F_P F_P^T - K_PP| max={abs_err:.2e}, rel={rel_err:.2e}")
    if rel_err > 1e-3:
        print(
            "  WARNING: F[P,:] F[P,:]^T does not closely match K_PP. "
            "The pivots-as-inducing-points prediction path assumes equality. "
            "If your RPCholesky variant doesn't preserve this property exactly, "
            "predictions may be inconsistent with the HMC training model."
        )

    test_idx = np.arange(X_test.shape[0])
    if args.subsample_test is not None:
        if args.stratified_subsample_test:
            X_test, y_test, test_idx = stratified_subsample(
                X_test,
                y_test,
                args.subsample_test,
                args.seed + 1,
            )
        else:
            rng = np.random.default_rng(args.seed)
            test_idx = np.sort(
                rng.choice(X_test.shape[0], size=args.subsample_test, replace=False)
            )
            X_test, y_test = X_test[test_idx], y_test[test_idx]

    assert X_test.shape[0] == y_test.shape[0]

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    actual_rank = F.shape[1]
    F_hmc = F.astype(np.float64, copy=False)
    print(f"HMC factor dtype: {F_hmc.dtype}, prediction factor dtype: {F.dtype}")

    hmc_step = args.hmc_step_constant / (args.k ** 0.25)
    print(
        f"Data: X_train={X_train.shape}, F={F.shape}, X_test={X_test.shape}, "
        f"y_test={y_test.shape}"
    )
    print(
        f"HMC: k={args.k}, actual_rank={actual_rank}, initial_step={hmc_step:.6f}, "
        f"target_accept={args.hmc_target_accept:.3f}, adapt={args.adapt_step_size}"
    )

    hmc_stats = run_hmc(
        factor=F_hmc,
        y=y_train,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        seed=args.seed,
        initial_step_size=hmc_step,
        n_leapfrog=args.n_leapfrog,
        target_accept=args.hmc_target_accept,
        adapt_step_size=args.adapt_step_size,
    )

    nu_samples = hmc_stats["nu_samples"]
    tau_nu = compute_tau_emcee(nu_samples)
    tau_logp = compute_tau_emcee(hmc_stats["logp_trace"])
    print(f"HMC warmup time: {hmc_stats['warmup_time']:.2f}s")
    print(f"HMC sampling time: {hmc_stats['sampling_time']:.2f}s")
    print(f"HMC acceptance rate: {hmc_stats['accept_rate']:.3f}")
    print(f"HMC final step size: {hmc_stats['step_size']:.6f}")
    print(f"HMC tau (nu mean): {tau_nu:.2f}")
    print(f"HMC tau (logp): {tau_logp:.2f}")

    predictive_backend = choose_predictive_backend(args)
    print(f"Sampling test predictive distribution with backend: {predictive_backend}")
    if predictive_backend == "cpu-parallel" and args.prediction_n_jobs > 1:
        print(
            "CPU parallel predictive sampling uses outer batch threads. "
            "For HPC runs, consider OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, "
            "and OPENBLAS_NUM_THREADS=1."
        )
    t_pred_start = time.perf_counter()
    if predictive_backend == "gpu":
        try:
            pred_test = sample_predictive_probabilities_pivots_gpu(
                F=F,
                X_train=X_train,
                X_test=X_test,
                pivot_indices=pivot_indices,
                K_pp=K_pivots,
                nu_samples=nu_samples,
                bandwidth=bandwidth,
                batch_size=args.prediction_batch_size,
                seed=999,
                device_arg=args.gpu_device,
            )
        except Exception as err:
            if args.predictive_backend == "gpu":
                raise
            print(f"GPU predictive backend failed ({err}); falling back to CPU parallel.")
            pred_test = sample_predictive_probabilities_pivots_parallel_cpu(
                F=F,
                X_train=X_train,
                X_test=X_test,
                pivot_indices=pivot_indices,
                K_pp=K_pivots,
                nu_samples=nu_samples,
                bandwidth=bandwidth,
                batch_size=args.prediction_batch_size,
                n_jobs=args.prediction_n_jobs,
                seed=999,
            )
    elif predictive_backend == "cpu-parallel":
        pred_test = sample_predictive_probabilities_pivots_parallel_cpu(
            F=F,
            X_train=X_train,
            X_test=X_test,
            pivot_indices=pivot_indices,
            K_pp=K_pivots,
            nu_samples=nu_samples,
            bandwidth=bandwidth,
            batch_size=args.prediction_batch_size,
            n_jobs=args.prediction_n_jobs,
            seed=999,
        )
    else:
        pred_test = sample_predictive_probabilities_pivots(
            F=F,
            X_train=X_train,
            X_test=X_test,
            pivot_indices=pivot_indices,
            K_pp=K_pivots,
            nu_samples=nu_samples,
            bandwidth=bandwidth,
            batch_size=args.prediction_batch_size,
            seed=999,
        )
    inference_time = time.perf_counter() - t_pred_start
    print(f"Predictive sampling time: {inference_time:.2f}s")
    print(f"Predictive backend used: {pred_test.get('backend_used', predictive_backend)}")

    evaluation_start = time.perf_counter()
    pred_summary = summarize_predictive_distribution(
        p_samples=pred_test["p_samples"],
        latent_samples=pred_test["latent_samples"],
    )
    predictive_prob = pred_summary["prob_mean"]
    predictive_var = pred_summary["prob_variance"]
    predictive_latent_mean = pred_summary["latent_mean"]
    predictive_latent_var = pred_summary["latent_variance"]

    test_metrics = evaluate_binary_probabilistic_predictions(
        y_true=y_test,
        p_pred=predictive_prob,
        threshold=0.5,
        n_bins=15,
    )
    test_metrics["posterior_expected_log_loss"] = float(
        -test_metrics["mean_predictive_log_likelihood"]
    )
    test_metrics["posterior_log_loss_mean"] = float(
        test_metrics["posterior_expected_log_loss"]
    )
    test_metrics["posterior_total_log_loss"] = float(-test_metrics["elpd"])
    test_metrics["kernel_time_sec"] = 0.0
    test_metrics["cholesky_time_sec"] = 0.0
    test_metrics["train_time_sec"] = round(float(hmc_stats["sampling_time"]), 3)
    test_metrics["prediction_time_sec"] = round(float(inference_time), 3)
    test_metrics["predictive_backend"] = str(pred_test.get("backend_used", predictive_backend))
    test_metrics["prediction_n_jobs"] = int(args.prediction_n_jobs)
    test_metrics["n_train"] = int(n_train)
    test_metrics["n_test"] = int(n_test)
    test_metrics["feature_dim"] = int(X_train.shape[1])
    test_metrics["kernel_bandwidth"] = float(bandwidth)
    test_metrics["n_samples"] = int(args.n_samples)
    test_metrics["n_warmup"] = int(args.n_warmup)
    test_metrics["hmc_step_size"] = float(hmc_stats["step_size"])
    test_metrics["n_leapfrog"] = int(args.n_leapfrog)
    test_metrics["n_conditional_draws"] = 1
    test_metrics["accept_rate"] = float(hmc_stats["accept_rate"])
    test_metrics["tau_nu"] = float(tau_nu)
    test_metrics["tau_logp"] = float(tau_logp)
    test_metrics["prob_min"] = float(np.min(predictive_prob))
    test_metrics["prob_mean"] = float(np.mean(predictive_prob))
    test_metrics["prob_max"] = float(np.max(predictive_prob))

    print_metric_table(test_metrics, title="Exp2 low-rank HMC GP test metrics")
    evaluation_time = time.perf_counter() - evaluation_start

    total_wall = time.perf_counter() - wall_t0

    prefix = f"exp2_rose_exp1_metrics_parallel_k{args.k}"
    metrics_path = output_dir / f"{prefix}_metrics.json"
    output_path = output_dir / f"{prefix}_results.npz"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "factor_dir": args.factor_dir,
                "embeddings": str(emb_path),
                "test_embeddings": args.test_embeddings,
                "test_labels": args.test_labels,
                "k": args.k,
                "kernel_bandwidth": bandwidth,
                "n_samples": args.n_samples,
                "n_warmup": args.n_warmup,
                "hmc_step_size": hmc_stats["step_size"],
                "n_leapfrog": args.n_leapfrog,
                "n_conditional_draws": 1,
                "prediction_batch_size": args.prediction_batch_size,
                "predictive_backend": args.predictive_backend,
                "predictive_backend_used": pred_test.get("backend_used", predictive_backend),
                "prediction_n_jobs": args.prediction_n_jobs,
                "gpu_device": args.gpu_device,
                "stratified_subsample_test": args.stratified_subsample_test,
                "standardize_from_factor": args.standardize_from_factor,
                "seed": args.seed,
                "metrics": test_metrics,
            },
            f,
            indent=2,
        )

    npz_payload = {
        "predictive_prob": predictive_prob,
        "predictive_std": pred_summary["prob_std"],
        "predictive_var": predictive_var,
        "predictive_latent_mean": predictive_latent_mean,
        "predictive_latent_std": pred_summary["latent_std"],
        "predictive_latent_var": predictive_latent_var,
        "predictive_prob_test": predictive_prob,
        "predictive_prob_var_test": predictive_var,
        "predictive_latent_mean_test": predictive_latent_mean,
        "predictive_latent_var_test": predictive_latent_var,
        "prob_q05": pred_summary["prob_q05"],
        "prob_q50": pred_summary["prob_q50"],
        "prob_q95": pred_summary["prob_q95"],
        "latent_q05": pred_summary["latent_q05"],
        "latent_q50": pred_summary["latent_q50"],
        "latent_q95": pred_summary["latent_q95"],
        "y_test": y_test,
        "test_indices": test_idx,
        "accept_rate": hmc_stats["accept_rate"],
        "tau_nu": tau_nu,
        "tau_logp": tau_logp,
        "final_step_size": hmc_stats["step_size"],
        "nu_samples": hmc_stats["nu_samples"],
        "k": args.k,
        "bandwidth": bandwidth,
        "predictive_backend": np.array(str(pred_test.get("backend_used", predictive_backend))),
        "prediction_n_jobs": args.prediction_n_jobs,
        "pivot_indices": pivot_indices,
        "n_train": n_train,
        "n_test": n_test,
        "timing_scope": (
            "precomputed_factor_load, hmc_warmup, hmc_sampling, "
            "predictive_sampling, evaluation"
        ),
        "data_loading_time_sec": data_loading_time,
        "factor_time_sec": 0.0,
        "warmup_time_sec": hmc_stats["warmup_time"],
        "sampling_time_sec": hmc_stats["sampling_time"],
        "per_step_time_sec": hmc_stats["per_step_time"],
        "total_mcmc_time_sec": hmc_stats["total_mcmc_time"],
        "inference_time_sec": inference_time,
        "evaluation_time_sec": evaluation_time,
        "total_pipeline_time_sec": total_wall,
        "total_model_compute_time_sec": (
            hmc_stats["total_sampler_time"] + inference_time + evaluation_time
        ),
    }
    npz_payload.update(
        {key: value for key, value in test_metrics.items() if key not in npz_payload}
    )
    np.savez(output_path, **npz_payload)

    print(f"Total wall-clock time: {total_wall:.2f}s")
    print("Saved:")
    print(f"- {metrics_path}")
    print(f"- {output_path}")


if __name__ == "__main__":
    main()
