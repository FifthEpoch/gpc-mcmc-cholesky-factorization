#!/usr/bin/env python3
"""
Verify Option A vs Option B predictive GP computations for accelerated RPCholesky.

This is a small-N diagnostic that compares:
- Exact dense Cholesky predictive inference
- Option A: RPCholesky full-factor predictive inference
- Option B: RPCholesky pivot-only Nyström predictive inference

Example:
    python experiments/exp_verify_option_a_vs_b.py \
        --embeddings data/embeddings.npy \
        --labels data/labels.npy \
        --n-train 1500 \
        --n-test 500 \
        --k 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import expit, log_expit
from scipy.stats import norm, wasserstein_distance


# Allow direct script execution without package install.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from my_cholesky.arpcholesky import arpcholesky  # noqa: E402
from my_cholesky.matrix import KernelMatrix  # noqa: E402


JITTER = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify exact, full-factor, and pivot-only RPCholesky predictions."
    )
    parser.add_argument("--embeddings", type=str, required=True, help="Path to (N, d) .npy.")
    parser.add_argument("--labels", type=str, required=True, help="Path to (N,) .npy.")
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-warmup", type=int, default=500)
    parser.add_argument("--hmc-step", type=float, default=0.05)
    parser.add_argument("--hmc-leapfrog", type=int, default=12)
    parser.add_argument("--kernel", type=str, default="gaussian", choices=["gaussian", "matern", "laplace"])
    parser.add_argument(
        "--bandwidth",
        type=str,
        default="approx_median",
        help='Kernel bandwidth: a float, "median", or "approx_median".',
    )
    parser.add_argument("--likelihood", type=str, default="probit", choices=["probit", "sigmoid"])
    parser.add_argument("--output-dir", type=str, default="data/exp_verify_ab")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        args.bandwidth = float(args.bandwidth)
    except ValueError:
        pass
    return args


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def load_and_split(
    embeddings_path: Path,
    labels_path: Path,
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.load(embeddings_path).astype(np.float64, copy=False)
    y = np.load(labels_path)
    if y.ndim > 1:
        y = y.squeeze()
    y = y.astype(np.float64, copy=False)

    if X.ndim != 2:
        raise ValueError(f"embeddings must have shape (N, d), got {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"labels must have shape (N,) matching embeddings, got {y.shape}")
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive")
    if n_train + n_test > X.shape[0]:
        raise ValueError(
            f"requested n_train+n_test={n_train + n_test}, but only {X.shape[0]} rows are available"
        )

    rng = np.random.default_rng(seed)
    selected = rng.choice(X.shape[0], size=n_train + n_test, replace=False)
    train_indices = selected[:n_train]
    test_indices = selected[n_train:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    return X_train, y_train, X_test, y_test, train_indices, test_indices


def resolve_bandwidth(X_train: np.ndarray, kernel: str, bandwidth: str | float, seed: int) -> float:
    """Resolve string bandwidth specs once so all paths use identical kernels."""
    np_state = np.random.get_state()
    try:
        np.random.seed(seed)
        A = KernelMatrix(X_train, kernel=kernel, bandwidth=bandwidth)
    finally:
        np.random.set_state(np_state)
    return float(A.bandwidth)


def kernel_matrix_between(
    X_left: np.ndarray,
    X_right: np.ndarray,
    kernel: str,
    bandwidth: float,
) -> np.ndarray:
    _, _, kernel_mtx = KernelMatrix.kernel_from_input(kernel, bandwidth=bandwidth)
    return np.asarray(kernel_mtx(X_left, X_right), dtype=np.float64)


def build_exact_kernel_and_chol(
    X_train: np.ndarray,
    kernel: str,
    bandwidth: float,
    jitter: float = JITTER,
) -> tuple[np.ndarray, np.ndarray]:
    t0 = time.perf_counter()
    K_train = kernel_matrix_between(X_train, X_train, kernel=kernel, bandwidth=bandwidth)
    K_train = (K_train + K_train.T) / 2.0
    K_train.flat[:: K_train.shape[0] + 1] += jitter
    L_exact = np.linalg.cholesky(K_train)
    print(f"Exact dense K and Cholesky built in {time.perf_counter() - t0:.2f}s")
    return K_train, L_exact


def build_rpchol_factor(
    X_train: np.ndarray,
    K_train: np.ndarray,
    kernel: str,
    bandwidth: float,
    k: int,
    block_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    A = KernelMatrix(X_train, kernel=kernel, bandwidth=bandwidth)
    t0 = time.perf_counter()
    lra = arpcholesky(A, k=k, b=block_size, seed=seed)
    elapsed = time.perf_counter() - t0

    F = np.asarray(lra.get_left_factor(), dtype=np.float64)
    pivots = np.asarray(lra.get_indices(), dtype=np.int64)
    k_actual = F.shape[1]

    approx = F @ F.T
    frob_rel_error = float(np.linalg.norm(approx - K_train, ord="fro") / np.linalg.norm(K_train, ord="fro"))
    trace_K = float(np.trace(K_train))
    trace_FFt = float(np.sum(F * F))
    rel_trace_error = float((trace_K - trace_FFt) / trace_K)

    stats = {
        "k_requested": int(k),
        "k_actual": int(k_actual),
        "block_size": int(block_size),
        "elapsed_sec": float(elapsed),
        "kernel_queries": int(A.num_queries()),
        "frob_rel_error": frob_rel_error,
        "trace_K": trace_K,
        "trace_FFt": trace_FFt,
        "rel_trace_error": rel_trace_error,
    }
    print(
        f"RPCholesky built in {elapsed:.2f}s: requested k={k}, actual k={k_actual}, "
        f"Frobenius rel error={frob_rel_error:.4e}, trace error={rel_trace_error:.4e}"
    )
    return F, pivots, stats


def log_posterior(
    nu: np.ndarray,
    factor: np.ndarray,
    y: np.ndarray,
    likelihood: str,
) -> float:
    f = factor @ nu
    if likelihood == "sigmoid":
        log_lik = np.sum(y * log_expit(f) + (1.0 - y) * log_expit(-f))
    elif likelihood == "probit":
        log_lik = np.sum(y * norm.logcdf(f) + (1.0 - y) * norm.logcdf(-f))
    else:
        raise ValueError(f"Unknown likelihood: {likelihood}")
    log_prior = -0.5 * np.dot(nu, nu)
    return float(log_lik + log_prior)


def grad_log_posterior(
    nu: np.ndarray,
    factor: np.ndarray,
    y: np.ndarray,
    likelihood: str,
) -> np.ndarray:
    f = factor @ nu
    if likelihood == "sigmoid":
        grad_f = y - expit(f)
    elif likelihood == "probit":
        with np.errstate(over="ignore", invalid="ignore"):
            imr_pos = np.exp(norm.logpdf(f) - norm.logcdf(f))
            imr_neg = np.exp(norm.logpdf(-f) - norm.logcdf(-f))
        imr_pos = np.nan_to_num(imr_pos, nan=0.0, posinf=np.finfo(float).max)
        imr_neg = np.nan_to_num(imr_neg, nan=0.0, posinf=np.finfo(float).max)
        grad_f = y * imr_pos - (1.0 - y) * imr_neg
    else:
        raise ValueError(f"Unknown likelihood: {likelihood}")
    return factor.T @ grad_f - nu


def run_hmc(
    factor: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    n_warmup: int,
    seed: int,
    step_size: float,
    n_leapfrog: int,
    likelihood: str,
    label: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    dim = factor.shape[1]
    total_steps = n_warmup + n_samples
    nu = np.zeros(dim, dtype=np.float64)
    logp = log_posterior(nu, factor, y, likelihood)

    accepts = np.zeros(total_steps, dtype=bool)
    logp_trace = np.zeros(total_steps, dtype=np.float64)
    nu_samples = np.zeros((n_samples, dim), dtype=np.float64)
    post_idx = 0

    t0 = time.perf_counter()
    for i in range(total_steps):
        current_nu = nu.copy()
        current_logp = logp
        momentum = rng.standard_normal(dim)

        grad = grad_log_posterior(current_nu, factor, y, likelihood)
        proposal_nu = current_nu.copy()
        proposal_p = momentum + 0.5 * step_size * grad

        for leapfrog_idx in range(n_leapfrog):
            proposal_nu = proposal_nu + step_size * proposal_p
            grad = grad_log_posterior(proposal_nu, factor, y, likelihood)
            if leapfrog_idx != n_leapfrog - 1:
                proposal_p = proposal_p + step_size * grad

        proposal_p = -(proposal_p + 0.5 * step_size * grad)
        proposal_logp = log_posterior(proposal_nu, factor, y, likelihood)

        current_h = -current_logp + 0.5 * np.dot(momentum, momentum)
        proposal_h = -proposal_logp + 0.5 * np.dot(proposal_p, proposal_p)
        log_accept = current_h - proposal_h
        accept_prob = 1.0 if log_accept >= 0.0 else float(np.exp(log_accept))

        if np.log(rng.random()) < log_accept:
            nu = proposal_nu
            logp = proposal_logp
            accepts[i] = True

        logp_trace[i] = logp
        if i >= n_warmup:
            nu_samples[post_idx, :] = nu
            post_idx += 1

    elapsed = time.perf_counter() - t0
    post_accept = float(np.mean(accepts[n_warmup:]))
    print(f"HMC {label}: dim={dim}, accept={post_accept:.3f}, elapsed={elapsed:.2f}s")
    if post_accept < 0.4 or post_accept > 0.95:
        print(f"  WARNING: HMC {label} acceptance {post_accept:.3f} outside [0.40, 0.95]")

    return {
        "nu_samples": nu_samples,
        "accept_rate": post_accept,
        "elapsed_sec": float(elapsed),
        "logp_trace": logp_trace,
        "step_size": float(step_size),
        "n_leapfrog": int(n_leapfrog),
    }


def link_probability(f: np.ndarray, likelihood: str) -> np.ndarray:
    if likelihood == "sigmoid":
        return expit(f)
    if likelihood == "probit":
        return norm.cdf(f)
    raise ValueError(f"Unknown likelihood: {likelihood}")


def predict_exact(
    L_exact: np.ndarray,
    K_train_test: np.ndarray,
    nu_samples: np.ndarray,
    likelihood: str,
) -> np.ndarray:
    alpha = solve_triangular(L_exact.T, nu_samples.T, lower=False, check_finite=False)
    f_test = K_train_test.T @ alpha
    return link_probability(f_test.T, likelihood)


def predict_option_a(
    F: np.ndarray,
    K_train_test: np.ndarray,
    nu_samples: np.ndarray,
    likelihood: str,
) -> np.ndarray:
    Q, R = np.linalg.qr(F, mode="reduced")
    z = solve_triangular(R.T, nu_samples.T, lower=True, check_finite=False)
    alpha = Q @ z
    f_test = K_train_test.T @ alpha
    return link_probability(f_test.T, likelihood)


def predict_option_b(
    X_train: np.ndarray,
    X_test: np.ndarray,
    F: np.ndarray,
    pivots: np.ndarray,
    nu_samples: np.ndarray,
    kernel: str,
    bandwidth: float,
    likelihood: str,
    jitter: float = JITTER,
) -> np.ndarray:
    X_pivots = X_train[pivots]
    K_pp = kernel_matrix_between(X_pivots, X_pivots, kernel=kernel, bandwidth=bandwidth)
    K_pp = (K_pp + K_pp.T) / 2.0
    K_pp.flat[:: K_pp.shape[0] + 1] += jitter
    L_pp = np.linalg.cholesky(K_pp)

    K_pivot_test = kernel_matrix_between(X_pivots, X_test, kernel=kernel, bandwidth=bandwidth)
    f_pivots = F[pivots, :] @ nu_samples.T
    tmp = solve_triangular(L_pp, f_pivots, lower=True, check_finite=False)
    weights = solve_triangular(L_pp.T, tmp, lower=False, check_finite=False)
    f_test = K_pivot_test.T @ weights
    return link_probability(f_test.T, likelihood)


def binary_metrics(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = np.clip(probs, 1e-12, 1.0 - 1e-12)
    nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    brier = float(np.mean((probs - y) ** 2))
    auroc = compute_auroc(y, probs)
    return {"nll": nll, "brier": brier, "auroc": auroc}


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=float)
    start = 0
    while start < scores.shape[0]:
        end = start + 1
        while end < scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end

    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def comparison_pair(
    name: str,
    probs_left: np.ndarray,
    probs_right: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float | str]:
    mean_left = np.mean(probs_left, axis=0)
    mean_right = np.mean(probs_right, axis=0)
    diff = np.abs(mean_left - mean_right)
    w1 = [
        wasserstein_distance(probs_left[:, j], probs_right[:, j])
        for j in range(probs_left.shape[1])
    ]
    corr = np.corrcoef(mean_left, mean_right)[0, 1]
    metrics_left = binary_metrics(mean_left, y_test)
    metrics_right = binary_metrics(mean_right, y_test)
    return {
        "comparison": name,
        "mean_abs_diff": float(np.mean(diff)),
        "max_abs_diff": float(np.max(diff)),
        "mean_wasserstein1": float(np.mean(w1)),
        "pearson_corr": float(corr),
        "delta_nll": float(metrics_right["nll"] - metrics_left["nll"]),
        "delta_brier": float(metrics_right["brier"] - metrics_left["brier"]),
        "delta_auroc": float(metrics_right["auroc"] - metrics_left["auroc"]),
        "left_nll": metrics_left["nll"],
        "right_nll": metrics_right["nll"],
        "left_brier": metrics_left["brier"],
        "right_brier": metrics_right["brier"],
        "left_auroc": metrics_left["auroc"],
        "right_auroc": metrics_right["auroc"],
    }


def compute_comparison_metrics(
    probs_exact: np.ndarray,
    probs_option_a: np.ndarray,
    probs_option_b: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    metrics = {
        "Exact vs Option A": comparison_pair("Exact vs Option A", probs_exact, probs_option_a, y_test),
        "Exact vs Option B": comparison_pair("Exact vs Option B", probs_exact, probs_option_b, y_test),
        "Option A vs Option B": comparison_pair("Option A vs Option B", probs_option_a, probs_option_b, y_test),
    }
    metrics["per_path"] = {
        "Exact": binary_metrics(np.mean(probs_exact, axis=0), y_test),
        "Option A": binary_metrics(np.mean(probs_option_a, axis=0), y_test),
        "Option B": binary_metrics(np.mean(probs_option_b, axis=0), y_test),
    }
    return metrics


def choose_diagnostic_points(mean_exact: np.ndarray, y_test: np.ndarray) -> list[tuple[str, int]]:
    pred = (mean_exact >= 0.5).astype(int)
    conf = np.abs(mean_exact - 0.5)
    correct = pred == y_test.astype(int)
    chosen: list[tuple[str, int]] = []

    if np.any(correct):
        idx = np.where(correct)[0][np.argmax(conf[correct])]
        chosen.append(("highest-confidence correct", int(idx)))
    if np.any(~correct):
        idx = np.where(~correct)[0][np.argmax(conf[~correct])]
        chosen.append(("highest-confidence incorrect", int(idx)))

    chosen.append(("most uncertain", int(np.argmin(conf))))

    if np.any(correct):
        correct_idx = np.where(correct)[0]
        idx = correct_idx[np.argsort(conf[correct_idx])[len(correct_idx) // 2]]
        chosen.append(("median-uncertainty correct", int(idx)))

    used = {idx for _, idx in chosen}
    fallback_order = np.argsort(conf)
    for idx in fallback_order:
        if len(chosen) >= 4:
            break
        if int(idx) not in used:
            chosen.append((f"fallback point {len(chosen) + 1}", int(idx)))
            used.add(int(idx))
    return chosen[:4]


def make_diagnostic_plots(
    output_dir: Path,
    probs_exact: np.ndarray,
    probs_option_a: np.ndarray,
    probs_option_b: np.ndarray,
    y_test: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    mean_exact = np.mean(probs_exact, axis=0)
    mean_a = np.mean(probs_option_a, axis=0)
    mean_b = np.mean(probs_option_b, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    panels = [
        (axes[0], mean_a, metrics["Exact vs Option A"]["mean_abs_diff"], "Option A"),
        (axes[1], mean_b, metrics["Exact vs Option B"]["mean_abs_diff"], "Option B"),
    ]
    for ax, yvals, mad, title in panels:
        ax.scatter(mean_exact, yvals, s=16, alpha=0.75)
        ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Exact vs {title} (MAD={mad:.3e})")
        ax.set_xlabel("Exact posterior mean probability")
        ax.set_ylabel(f"{title} posterior mean probability")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_exact_vs_options.png", dpi=170)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(np.abs(mean_exact - mean_a), bins=40, alpha=0.65, label="|Exact - Option A|")
    plt.hist(np.abs(mean_exact - mean_b), bins=40, alpha=0.65, label="|Exact - Option B|")
    plt.xlabel("Absolute pointwise difference in posterior mean probability")
    plt.ylabel("Count")
    plt.title("Pointwise predictive differences")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "hist_pointwise_differences.png", dpi=170)
    plt.close()

    selected = choose_diagnostic_points(mean_exact, y_test)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (label, idx) in zip(axes.ravel(), selected):
        ax.hist(probs_exact[:, idx], bins=35, alpha=0.45, density=True, label="Exact")
        ax.hist(probs_option_a[:, idx], bins=35, alpha=0.45, density=True, label="Option A")
        ax.hist(probs_option_b[:, idx], bins=35, alpha=0.45, density=True, label="Option B")
        ax.set_title(f"{label} (idx={idx}, y={int(y_test[idx])}, p_exact={mean_exact[idx]:.3f})")
        ax.set_xlabel("Predictive probability")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "marginal_predictive_histograms.png", dpi=170)
    plt.close()


def print_summary_table(metrics: dict[str, Any]) -> None:
    rows = [metrics["Exact vs Option A"], metrics["Exact vs Option B"], metrics["Option A vs Option B"]]
    fmt = "{:<22} {:>10} {:>10} {:>10} {:>10} {:>11} {:>11} {:>11}"
    print("\n" + "=" * 100)
    print(
        fmt.format(
            "Comparison",
            "MeanAbs",
            "MaxAbs",
            "W1",
            "Corr",
            "Delta NLL",
            "Delta Brier",
            "Delta AUROC",
        )
    )
    print("-" * 100)
    for row in rows:
        print(
            fmt.format(
                row["comparison"],
                f"{row['mean_abs_diff']:.3e}",
                f"{row['max_abs_diff']:.3e}",
                f"{row['mean_wasserstein1']:.3e}",
                f"{row['pearson_corr']:.5f}",
                f"{row['delta_nll']:.3e}",
                f"{row['delta_brier']:.3e}",
                f"{row['delta_auroc']:.3e}",
            )
        )
    print("=" * 100)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and splitting data...")
    X_train, y_train, X_test, y_test, train_indices, test_indices = load_and_split(
        Path(args.embeddings),
        Path(args.labels),
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
    )
    np.save(output_dir / "train_indices.npy", train_indices.astype(np.int64))
    np.save(output_dir / "test_indices.npy", test_indices.astype(np.int64))

    bandwidth_actual = resolve_bandwidth(X_train, args.kernel, args.bandwidth, args.seed)
    print(f"Kernel: {args.kernel}, bandwidth spec={args.bandwidth}, actual={bandwidth_actual:.6g}")

    print("Building exact dense baseline...")
    K_train, L_exact = build_exact_kernel_and_chol(
        X_train, kernel=args.kernel, bandwidth=bandwidth_actual, jitter=JITTER
    )

    print("Building accelerated RPCholesky factor...")
    F, pivots, rpchol_stats = build_rpchol_factor(
        X_train,
        K_train,
        kernel=args.kernel,
        bandwidth=bandwidth_actual,
        k=args.k,
        block_size=args.block_size,
        seed=args.seed,
    )
    k_actual = F.shape[1]
    np.save(output_dir / "pivot_indices.npy", pivots.astype(np.int64))

    print("Running exact-reference HMC...")
    exact_hmc = run_hmc(
        L_exact,
        y_train,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        seed=args.seed,
        step_size=args.hmc_step,
        n_leapfrog=args.hmc_leapfrog,
        likelihood=args.likelihood,
        label="Exact",
    )

    print("Running RPCholesky HMC...")
    rpchol_hmc = run_hmc(
        F,
        y_train,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        seed=args.seed,
        step_size=args.hmc_step,
        n_leapfrog=args.hmc_leapfrog,
        likelihood=args.likelihood,
        label="RPCholesky",
    )

    nu_exact = exact_hmc["nu_samples"]
    nu_rpchol = rpchol_hmc["nu_samples"]
    np.save(output_dir / "nu_samples_exact.npy", nu_exact)
    np.save(output_dir / "nu_samples_rpchol.npy", nu_rpchol)

    print("Computing train-test kernels and predictions...")
    K_train_test = kernel_matrix_between(X_train, X_test, kernel=args.kernel, bandwidth=bandwidth_actual)
    probs_exact = predict_exact(L_exact, K_train_test, nu_exact, args.likelihood)
    probs_option_a = predict_option_a(F, K_train_test, nu_rpchol, args.likelihood)
    probs_option_b = predict_option_b(
        X_train,
        X_test,
        F,
        pivots,
        nu_rpchol,
        kernel=args.kernel,
        bandwidth=bandwidth_actual,
        likelihood=args.likelihood,
        jitter=JITTER,
    )
    np.save(output_dir / "predictive_probs_exact.npy", probs_exact)
    np.save(output_dir / "predictive_probs_option_a.npy", probs_option_a)
    np.save(output_dir / "predictive_probs_option_b.npy", probs_option_b)

    print("Computing metrics and plots...")
    metrics = compute_comparison_metrics(probs_exact, probs_option_a, probs_option_b, y_test)
    with open(output_dir / "comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(metrics), f, indent=2)
    make_diagnostic_plots(output_dir, probs_exact, probs_option_a, probs_option_b, y_test, metrics)

    run_config = {
        "embeddings": str(args.embeddings),
        "labels": str(args.labels),
        "n_train": int(args.n_train),
        "n_test": int(args.n_test),
        "k_requested": int(args.k),
        "k_actual": int(k_actual),
        "block_size": int(args.block_size),
        "n_samples": int(args.n_samples),
        "n_warmup": int(args.n_warmup),
        "hmc_step": float(args.hmc_step),
        "hmc_leapfrog": int(args.hmc_leapfrog),
        "kernel": args.kernel,
        "bandwidth_spec": args.bandwidth,
        "bandwidth_actual": float(bandwidth_actual),
        "likelihood": args.likelihood,
        "seed": int(args.seed),
        "jitter": float(JITTER),
        "rpchol_stats": rpchol_stats,
        "hmc_exact": {
            "accept_rate": exact_hmc["accept_rate"],
            "elapsed_sec": exact_hmc["elapsed_sec"],
        },
        "hmc_rpchol": {
            "accept_rate": rpchol_hmc["accept_rate"],
            "elapsed_sec": rpchol_hmc["elapsed_sec"],
        },
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(run_config), f, indent=2)

    print_summary_table(metrics)
    print(f"\nSaved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
