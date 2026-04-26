"""
Experiment 1: Speed comparison of Dense Cholesky vs RPCholesky in
Random Walk Metropolis (RWM) for fake GP binary classification.

We compare:
    Method A (Dense):  f = L_dense @ nu,  L_dense = chol(K)
    Method B (RPChol): f = F @ nu,        F from arpcholesky(A, k, b=10)

across k in [20, 50, 100], reporting factor time, MCMC per-step time,
total MCMC time, acceptance rate, and Frobenius approximation error.
"""

from __future__ import annotations

import os
import sys
import time

import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit


# Allow direct script execution without package install.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from my_cholesky.arpcholesky import arpcholesky
from my_cholesky.matrix import KernelMatrix


def write_fake_blob_data(X: np.ndarray, y: np.ndarray, output_path: str) -> None:
    """Write only synthetic sample coordinates to a text file."""
    with open(output_path, "w", encoding="utf-8") as output_file:
        for x1, x2 in X:
            output_file.write(f"{x1:.10f}\t{x2:.10f}\n")


def write_chain_sample_header(output_file, dim: int) -> None:
    """Write the header for accepted post-warmup MCMC draws."""
    header = ["accepted_idx", "post_warmup_idx", "logp"] + [
        f"nu_{j}" for j in range(dim)
    ]
    output_file.write("\t".join(header) + "\n")


def append_chain_sample(
    output_file,
    accepted_idx: int,
    post_warmup_idx: int,
    logp: float,
    nu: np.ndarray,
) -> None:
    """Append one accepted post-warmup MCMC draw to the output file."""
    nu_values = "\t".join(f"{value:.10f}" for value in nu)
    output_file.write(
        f"{accepted_idx}\t{post_warmup_idx}\t{logp:.10f}\t{nu_values}\n"
    )
    output_file.flush()


def make_fake_blobs(seed: int = 42, output_path: str | None = None):
    """Generate N=2000 two-blob binary data in R^2."""
    rng = np.random.default_rng(seed)
    n_per_class = 1000
    cov = 0.5 * np.eye(2)
    x0 = rng.multivariate_normal(mean=[-1.0, 0.0], cov=cov, size=n_per_class)
    x1 = rng.multivariate_normal(mean=[1.0, 0.0], cov=cov, size=n_per_class)
    X = np.vstack([x0, x1])
    y = np.concatenate(
        [np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)]
    )
    if output_path is not None:
        write_fake_blob_data(X, y, output_path)
    return X, y


def log_posterior(nu: np.ndarray, factor: np.ndarray, y: np.ndarray) -> float:
    """Log posterior for Bernoulli likelihood and standard normal prior."""
    f = factor @ nu
    p = expit(f)
    log_lik = np.sum(y * np.log(p + 1e-10) + (1 - y) * np.log(1 - p + 1e-10))
    log_prior = -0.5 * np.dot(nu, nu)
    return float(log_lik + log_prior)


def run_rwm(
    factor: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    n_warmup: int,
    seed: int,
    target_accept: float = 0.30,
    adapt_interval: int = 50,
    sample_hook=None,
    adapt_during_warmup: bool = True,
    initial_step_size: float | None = None,
):
    """Run Random Walk Metropolis and return timing/statistics."""
    rng = np.random.default_rng(seed)
    dim = factor.shape[1]
    total_steps = n_warmup + n_samples
    if initial_step_size is None:
        step_size = 2.38 / np.sqrt(dim)
    else:
        step_size = float(initial_step_size)

    nu = np.zeros(dim, dtype=float)
    logp = log_posterior(nu, factor, y)

    step_times = np.zeros(total_steps, dtype=float)
    logp_trace = np.zeros(total_steps, dtype=float)
    accepts = np.zeros(total_steps, dtype=bool)
    nu_samples = np.zeros((n_samples, dim), dtype=float)
    post_idx = 0
    accepted_post_count = 0

    for i in range(total_steps):
        t0 = time.perf_counter()
        nu_prop = nu + step_size * rng.standard_normal(dim)
        logp_prop = log_posterior(nu_prop, factor, y)

        if np.log(rng.random()) < (logp_prop - logp):
            nu = nu_prop
            logp = logp_prop
            accepts[i] = True

        # Adaptive tuning during warmup only.
        if adapt_during_warmup and i < n_warmup and (i + 1) % adapt_interval == 0:
            window_start = i + 1 - adapt_interval
            accept_rate_window = float(np.mean(accepts[window_start : i + 1]))
            lower = target_accept - 0.10
            upper = target_accept + 0.10
            if accept_rate_window > upper:
                step_size *= 1.1
            elif accept_rate_window < lower:
                step_size *= 0.9

        step_times[i] = time.perf_counter() - t0
        logp_trace[i] = logp
        if i >= n_warmup:
            # Collect post-warmup chain regardless of accept/reject.
            nu_samples[post_idx, :] = nu
            if accepts[i]:
                accepted_post_count += 1
                if sample_hook is not None:
                    sample_hook(
                        accepted_post_count,
                        post_idx + 1,
                        float(logp),
                        nu.copy(),
                    )
            post_idx += 1

    post = slice(n_warmup, total_steps)
    warmup = slice(0, n_warmup)
    warmup_time = float(np.sum(step_times[warmup]))
    per_step_time = float(np.mean(step_times[post]))
    sampling_time = float(np.sum(step_times[post]))
    accept_rate = float(np.mean(accepts[post]))

    return {
        "per_step_time": per_step_time,
        "warmup_time": warmup_time,
        "sampling_time": sampling_time,
        "total_mcmc_time": sampling_time,
        "total_sampler_time": warmup_time + sampling_time,
        "accept_rate": accept_rate,
        "accepted_post_count": accepted_post_count,
        "logp_trace": logp_trace,
        "final_step_size": float(step_size),
        "nu_samples": nu_samples,
    }


def compute_acf(x: np.ndarray, max_lag: int = 500) -> np.ndarray:
    """Estimate the 1D normalized autocorrelation function using emcee."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 2 or np.var(x) < 1e-12:
        return np.zeros(max_lag + 1, dtype=float)

    acf = emcee.autocorr.function_1d(x)
    if acf.size >= max_lag + 1:
        return np.asarray(acf[: max_lag + 1], dtype=float)

    padded = np.zeros(max_lag + 1, dtype=float)
    padded[: acf.size] = acf
    return padded


def compute_tau(chain: np.ndarray) -> float:
    """
    Estimate integrated autocorrelation time using emcee.

    For 1D chains, treat the input as a single time series. For 2D chains,
    treat the last axis as walkers to match exp1b_emcee_gpc.py.
    """
    chain = np.asarray(chain, dtype=float)
    if chain.size < 2 or np.var(chain) < 1e-12:
        return float("nan")

    try:
        if chain.ndim == 1:
            tau = emcee.autocorr.integrated_time(
                chain, quiet=True, has_walkers=False
            )
        else:
            tau = emcee.autocorr.integrated_time(chain, quiet=True)
    except Exception as err:
        print(f"  WARNING: emcee tau estimate failed: {err}")
        return float("nan")

    tau = np.asarray(tau, dtype=float).reshape(-1)
    if tau.size == 0:
        return float("nan")
    return float(np.nanmean(tau))


def compute_ess_from_tau(
    n_steps: int, tau_or_n_walkers: float | int, tau: float | None = None
) -> float:
    """
    Convert integrated autocorrelation time to ESS.

    Supports both:
    - compute_ess_from_tau(n_samples, tau)
    - compute_ess_from_tau(n_steps, n_walkers, tau)
    """
    if tau is None:
        n_total = float(n_steps)
        tau_value = float(tau_or_n_walkers)
    else:
        n_total = float(n_steps) * float(tau_or_n_walkers)
        tau_value = float(tau)

    if not np.isfinite(tau_value) or tau_value <= 0:
        return float("nan")
    return float(n_total / tau_value)

def compute_ess(x: np.ndarray) -> float:
    """Estimate ESS using emcee's integrated autocorrelation time."""
    x = np.asarray(x, dtype=float).reshape(-1)
    return compute_ess_from_tau(len(x), compute_tau(x))


def get_approx_independent_samples(
    samples: np.ndarray, tau: float
) -> tuple[np.ndarray, int]:
    """
    Return a thinned subset using stride ceil(tau).

    This gives an approximately decorrelated subset for inspection or
    downstream use, but it does not increase the true ESS of the run.
    """
    stride = 1
    if np.isfinite(tau) and tau > 1:
        stride = int(np.ceil(tau))
    return samples[::stride], stride


def main() -> None:
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    output_prefix = "exp1_mcmc_gpc_2"
    rp_seed_base = 5000

    # MCMC setup
    n_samples = 10000
    n_warmup = 500
    rp_k_values = [20, 50, 100]
    diag_n_samples = 500
    diag_n_warmup = 200

    # Fake data and kernel
    output_path = os.path.abspath(
        os.path.join(os.getcwd(), f"{output_prefix}_output.txt")
    )
    X, y = make_fake_blobs(seed=42)
    A = KernelMatrix(X, kernel="gaussian", bandwidth=1.0)
    K_dense = A[:, :]
    n = K_dense.shape[0]

    # Dense factorization baseline
    t0 = time.perf_counter()
    L_dense = np.linalg.cholesky(K_dense + 1e-6 * np.eye(n))
    dense_factor_time = time.perf_counter() - t0

    dense_stats = run_rwm(
        factor=L_dense,
        y=y,
        n_samples=n_samples,
        n_warmup=n_warmup,
        seed=123,
    )

    results = [
        {
            "method": "Dense",
            "k": int(n),
            "factor_time": float(dense_factor_time),
            "per_step_time": dense_stats["per_step_time"],
            "warmup_time": dense_stats["warmup_time"],
            "sampling_time": dense_stats["sampling_time"],
            "total_time": dense_stats["total_mcmc_time"],
            "total_model_compute_time": float(dense_factor_time) + dense_stats["total_sampler_time"],
            "accept_rate": dense_stats["accept_rate"],
            "approx_error": 0.0,
            "logp_trace": dense_stats["logp_trace"],
            "final_step_size": dense_stats["final_step_size"],
        }
    ]

    fro_norm_K = np.linalg.norm(K_dense, "fro")
    rp_error_by_k = []
    rp_factor_times = []
    rp_total_times = []
    rp50_factor = None
    rp50_nu_samples = None
    rp50_logp_trace = None
    rp50_accepted_post_count = None

    for k in rp_k_values:
        # RPCholesky: O(Nk^2) vs dense Cholesky O(N^3)
        t0 = time.perf_counter()
        lra = arpcholesky(A, k=k, b=10, seed=rp_seed_base + k)
        F = lra.get_left_factor()  # shape (N, k_eff)
        rp_factor_time = time.perf_counter() - t0

        approx_err = float(
            np.linalg.norm(K_dense - (F @ F.T), "fro") / (fro_norm_K + 1e-12)
        )

        if k == 50:
            with open(output_path, "w", encoding="utf-8", buffering=1) as output_file:
                write_chain_sample_header(output_file, F.shape[1])
                rp_stats = run_rwm(
                    factor=F,
                    y=y,
                    n_samples=n_samples,
                    n_warmup=n_warmup,
                    seed=123 + k,
                    sample_hook=lambda accepted_idx, post_warmup_idx, logp, nu: append_chain_sample(
                        output_file,
                        accepted_idx,
                        post_warmup_idx,
                        logp,
                        nu,
                    ),
            )
            rp50_factor = F
            rp50_nu_samples = rp_stats["nu_samples"]
            rp50_logp_trace = rp_stats["logp_trace"]
            rp50_accepted_post_count = rp_stats["accepted_post_count"]
        else:
            rp_stats = run_rwm(
                factor=F,
                y=y,
                n_samples=n_samples,
                n_warmup=n_warmup,
                seed=123 + k,
            )

        results.append(
            {
                "method": "RPChol",
                "k": int(F.shape[1]),
                "factor_time": float(rp_factor_time),
                "per_step_time": rp_stats["per_step_time"],
                "warmup_time": rp_stats["warmup_time"],
                "sampling_time": rp_stats["sampling_time"],
                "total_time": rp_stats["total_mcmc_time"],
                "total_model_compute_time": float(rp_factor_time) + rp_stats["total_sampler_time"],
                "accept_rate": rp_stats["accept_rate"],
                "approx_error": approx_err,
                "logp_trace": rp_stats["logp_trace"],
                "final_step_size": rp_stats["final_step_size"],
            }
        )

        rp_error_by_k.append((int(F.shape[1]), approx_err))
        rp_factor_times.append((f"RPChol k={int(F.shape[1])}", float(rp_factor_time)))
        rp_total_times.append(
            (f"RPChol k={int(F.shape[1])}", float(rp_stats["total_mcmc_time"]))
        )

    # Print clean, aligned results table.
    row_fmt = (
        "{:<10} {:>6} {:>16} {:>17} {:>14} {:>12} {:>12} {:>13}"
    )
    print(
        row_fmt.format(
            "Method",
            "k",
            "Factor time(s)",
            "Per-step time(s)",
            "Total time(s)",
            "Accept rate",
            "Step size",
            "Approx error",
        )
    )
    print("-" * 108)
    for row in results:
        print(
            row_fmt.format(
                row["method"],
                row["k"],
                f"{row['factor_time']:.3f}",
                f"{row['per_step_time']:.6f}",
                f"{row['total_time']:.3f}",
                f"{row['accept_rate']:.3f}",
                f"{row['final_step_size']:.6f}",
                f"{row['approx_error']:.6f}",
            )
        )

    # =========================
    # Autocorrelation diagnostics (Dense vs RPChol k=50)
    # =========================
    print("\n=== Autocorrelation diagnostics (Dense vs RPChol k=50) ===")
    if rp50_logp_trace is None or rp50_nu_samples is None:
        raise RuntimeError("RPChol k=50 diagnostics are unavailable.")

    if rp50_factor is None:
        raise RuntimeError("RPChol k=50 factor is unavailable for diagnostics.")

    # Use an exp1b-comparable setup for tau/ESS:
    # same 500/200 chain length and fixed Gaussian proposal scale.
    dense_diag_stats = run_rwm(
        factor=L_dense,
        y=y,
        n_samples=diag_n_samples,
        n_warmup=diag_n_warmup,
        seed=1000,
        adapt_during_warmup=False,
    )
    rp50_diag_stats = run_rwm(
        factor=rp50_factor,
        y=y,
        n_samples=diag_n_samples,
        n_warmup=diag_n_warmup,
        seed=1050,
        adapt_during_warmup=False,
    )

    dense_logp_post = dense_diag_stats["logp_trace"][diag_n_warmup:]
    rp50_logp_post = rp50_diag_stats["logp_trace"][diag_n_warmup:]
    dense_nu0 = dense_diag_stats["nu_samples"][:, 0]
    rp50_nu0 = rp50_diag_stats["nu_samples"][:, 0]

    acf_dense = compute_acf(dense_logp_post, max_lag=500)
    acf_rp50 = compute_acf(rp50_logp_post, max_lag=500)

    tau_dense_logp = compute_tau(dense_logp_post)
    tau_rp50_logp = compute_tau(rp50_logp_post)
    tau_dense_nu0 = compute_tau(dense_nu0)
    tau_rp50_nu0 = compute_tau(rp50_nu0)

    ess_dense_logp = compute_ess_from_tau(len(dense_logp_post), tau_dense_logp)
    ess_rp50_logp = compute_ess_from_tau(len(rp50_logp_post), tau_rp50_logp)
    ess_dense_nu0 = compute_ess_from_tau(len(dense_nu0), tau_dense_nu0)
    ess_rp50_nu0 = compute_ess_from_tau(len(rp50_nu0), tau_rp50_nu0)

    dense_indep_logp, dense_stride = get_approx_independent_samples(
        dense_logp_post, tau_dense_logp
    )
    rp50_indep_logp, rp50_stride = get_approx_independent_samples(
        rp50_logp_post, tau_rp50_logp
    )

    rp50_ref_walkers = max(2 * rp50_factor.shape[1] + 2, 24)
    rp50_ensemble_equiv_ess = compute_ess_from_tau(
        diag_n_samples, rp50_ref_walkers, tau_rp50_logp
    )

    print(
        "Diagnostics below use a fixed-step 500/200 run to match "
        "exp1b_emcee_gpc.py more closely."
    )
    print("Approx independent count below is based on tau(log posterior).")
    diag_fmt = "{:<12} {:>12} {:>13} {:>12} {:>14} {:>10}"
    print(
        diag_fmt.format(
            "Method",
            "ESS (logp)",
            "ESS (nu[0])",
            "tau (logp)",
            "approx indep",
            "stride",
        )
    )
    print(
        diag_fmt.format(
            "Dense",
            f"{ess_dense_logp:.1f}",
            f"{ess_dense_nu0:.1f}",
            f"{tau_dense_logp:.2f}",
            str(len(dense_indep_logp)),
            str(dense_stride),
        )
    )
    print(
        diag_fmt.format(
            "RPChol-k50",
            f"{ess_rp50_logp:.1f}",
            f"{ess_rp50_nu0:.1f}",
            f"{tau_rp50_logp:.2f}",
            str(len(rp50_indep_logp)),
            str(rp50_stride),
        )
    )
    print(
        "RPChol-k50 ensemble-equivalent ESS using "
        f"{rp50_ref_walkers} exp1b walkers: {rp50_ensemble_equiv_ess:.1f}"
    )

    # Plot 5: ACF comparison for log posterior.
    plt.figure(figsize=(7, 4))
    plt.plot(acf_dense, label="Dense", color="tab:gray")
    plt.plot(acf_rp50, label="RPChol k=50", color="tab:blue")
    plt.axhline(
        0.05, color="black", linestyle="--", linewidth=0.8, label="0.05 threshold"
    )
    plt.axhline(0.0, color="black", linestyle="-", linewidth=0.5)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("ACF of log posterior — Dense vs RPChol k=50")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, 500)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"{output_prefix}_acf_comparison.png"), dpi=160)
    plt.close()

    # =========================
    # Classification visualization for RPChol k=50
    # =========================
    print("\n=== Classification visualization (RPChol k=50) ===")
    if rp50_factor is None or rp50_nu_samples is None:
        raise RuntimeError("RPChol k=50 run did not produce visualization samples.")

    # Posterior predictive probabilities on training points.
    f_samples = rp50_nu_samples @ rp50_factor.T  # shape (n_samples, N)
    p_samples = expit(f_samples)
    p_mean = np.mean(p_samples, axis=0)
    p_std = np.std(p_samples, axis=0)
    y_pred = (p_mean > 0.5).astype(int)
    accuracy = float(np.mean(y_pred == y))
    print("RPChol k=50 classification accuracy: {:.1f}%".format(accuracy * 100.0))

    # Plot 5: training scatter colored by posterior mean probability.
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        X[:, 0], X[:, 1], c=p_mean, cmap="RdBu_r", s=10, vmin=0.0, vmax=1.0
    )
    plt.colorbar(sc, label="P(y=1)")
    plt.title("Posterior mean P(y=1) — RPChol k=50")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(
        os.path.join(data_dir, f"{output_prefix}_classification_scatter.png"), dpi=160
    )
    plt.close()

    # Plot 6: decision boundary on a 2D grid (simple fallback).
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))
    X_grid = np.column_stack([xx.ravel(), yy.ravel()])

    # k_star shape: (n_grid, N)
    diff = X_grid[:, None, :] - X[None, :, :]
    k_star = np.exp(-np.sum(diff**2, axis=2) / (2.0 * 1.0**2))
    nu_mean = np.mean(rp50_nu_samples, axis=0)
    rhs = rp50_factor @ nu_mean
    # Approximate predictive mean using RPChol factor as kernel surrogate.
    # FF^T approximates K here — not the exact GP predictive mean.
    # Used for visualization only.
    alpha = np.linalg.solve(
        rp50_factor @ rp50_factor.T + 1e-6 * np.eye(rp50_factor.shape[0]),
        rhs,
    )
    f_grid_mean = k_star @ alpha
    p_grid = expit(f_grid_mean).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    cf = plt.contourf(
        xx, yy, p_grid, levels=50, cmap="RdBu_r", alpha=0.6, vmin=0.0, vmax=1.0
    )
    plt.colorbar(cf, label="P(y=1)")
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", s=8, label="Class 0", alpha=0.4)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red", s=8, label="Class 1", alpha=0.4)
    plt.contour(xx, yy, p_grid, levels=[0.5], colors="black", linewidths=1.5)
    plt.legend()
    plt.title("Decision boundary — RPChol k=50")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(
        os.path.join(data_dir, f"{output_prefix}_decision_boundary.png"), dpi=160
    )
    plt.close()

    # Plot 7: uncertainty on training points.
    plt.figure(figsize=(6, 5))
    su = plt.scatter(X[:, 0], X[:, 1], c=p_std, cmap="Oranges", s=10)
    plt.colorbar(su, label="Std of P(y=1)")
    plt.title("Posterior uncertainty — RPChol k=50")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"{output_prefix}_uncertainty.png"), dpi=160)
    plt.close()

    # Plot 1: bar chart factor computation time
    labels_factor = ["Dense"] + [x[0] for x in rp_factor_times]
    vals_factor = [dense_factor_time] + [x[1] for x in rp_factor_times]
    plt.figure(figsize=(7, 4))
    plt.bar(labels_factor, vals_factor, color=["tab:gray", "tab:blue", "tab:orange", "tab:green"])
    plt.ylabel("Seconds")
    plt.title("Factor computation time")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"{output_prefix}_factor_time_bar.png"), dpi=160)
    plt.close()

    # Plot 2: bar chart total MCMC time
    labels_total = ["Dense"] + [x[0] for x in rp_total_times]
    vals_total = [dense_stats["total_mcmc_time"]] + [x[1] for x in rp_total_times]
    plt.figure(figsize=(7, 4))
    plt.bar(labels_total, vals_total, color=["tab:gray", "tab:blue", "tab:orange", "tab:green"])
    plt.ylabel("Seconds")
    plt.title("Total MCMC runtime (post-warmup)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(data_dir, f"{output_prefix}_total_mcmc_time_bar.png"), dpi=160
    )
    plt.close()

    # Plot 3: approximation error vs k (RPChol only)
    rp_error_by_k = sorted(rp_error_by_k, key=lambda t: t[0])
    ks = [x[0] for x in rp_error_by_k]
    errs = [x[1] for x in rp_error_by_k]
    plt.figure(figsize=(6, 4))
    plt.plot(ks, errs, marker="o", linewidth=1.5)
    plt.xlabel("k")
    plt.ylabel(r"$||K-FF^T||_F / ||K||_F$")
    plt.title("RPCholesky approximation error vs rank")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(data_dir, f"{output_prefix}_approx_error_vs_k.png"), dpi=160
    )
    plt.close()

    # Plot 4: side-by-side trace plots (Dense vs RPChol k=50)
    dense_trace = results[0]["logp_trace"]
    rp50_trace = None
    for row in results[1:]:
        if row["k"] == 50:
            rp50_trace = row["logp_trace"]
            break

    if rp50_trace is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        axes[0].plot(dense_trace, linewidth=1.0, color="tab:gray")
        axes[0].set_title("Dense trace")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("log posterior")
        axes[0].grid(alpha=0.3)

        axes[1].plot(rp50_trace, linewidth=1.0, color="tab:blue")
        axes[1].set_title("RPChol trace (k=50)")
        axes[1].set_xlabel("Iteration")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(data_dir, f"{output_prefix}_trace_dense_vs_rp50.png"),
            dpi=160,
        )
        plt.close()

    if rp50_nu_samples is None or rp50_logp_trace is None:
        raise RuntimeError("RPChol k=50 chain is unavailable for output export.")
    if rp50_accepted_post_count is None:
        raise RuntimeError("RPChol k=50 accepted sample count is unavailable.")

    np.save(
        os.path.join(data_dir, f"{output_prefix}_results.npy"),
        {
            "results": results,
            "n_samples": n_samples,
            "n_warmup": n_warmup,
            "target_accept": 0.30,
            "adapt_interval": 50,
        },
        allow_pickle=True,
    )
    print("Saved:")
    print(f"- data/{output_prefix}_factor_time_bar.png")
    print(f"- data/{output_prefix}_total_mcmc_time_bar.png")
    print(f"- data/{output_prefix}_approx_error_vs_k.png")
    print(f"- data/{output_prefix}_trace_dense_vs_rp50.png")
    print(f"- data/{output_prefix}_acf_comparison.png")
    print(f"- data/{output_prefix}_classification_scatter.png")
    print(f"- data/{output_prefix}_decision_boundary.png")
    print(f"- data/{output_prefix}_uncertainty.png")
    print(f"- data/{output_prefix}_results.npy")
    print(
        f"- {output_path} "
        f"({rp50_accepted_post_count} accepted post-warmup RPChol k=50 draws)"
    )


if __name__ == "__main__":
    main()
