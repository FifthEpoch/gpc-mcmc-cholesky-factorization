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


def make_fake_blobs(seed: int = 42):
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
):
    """Run Random Walk Metropolis and return timing/statistics."""
    rng = np.random.default_rng(seed)
    dim = factor.shape[1]
    total_steps = n_warmup + n_samples
    step_size = 2.38 / np.sqrt(dim)

    nu = np.zeros(dim, dtype=float)
    logp = log_posterior(nu, factor, y)

    step_times = np.zeros(total_steps, dtype=float)
    logp_trace = np.zeros(total_steps, dtype=float)
    accepts = np.zeros(total_steps, dtype=bool)
    nu_samples = np.zeros((n_samples, dim), dtype=float)
    post_idx = 0

    for i in range(total_steps):
        t0 = time.perf_counter()
        nu_prop = nu + step_size * rng.standard_normal(dim)
        logp_prop = log_posterior(nu_prop, factor, y)

        if np.log(rng.random()) < (logp_prop - logp):
            nu = nu_prop
            logp = logp_prop
            accepts[i] = True

        # Adaptive tuning during warmup only.
        if i < n_warmup and (i + 1) % adapt_interval == 0:
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
            post_idx += 1

    post = slice(n_warmup, total_steps)
    per_step_time = float(np.mean(step_times[post]))
    total_mcmc_time = float(np.sum(step_times[post]))
    accept_rate = float(np.mean(accepts[post]))

    return {
        "per_step_time": per_step_time,
        "total_mcmc_time": total_mcmc_time,
        "accept_rate": accept_rate,
        "logp_trace": logp_trace,
        "final_step_size": float(step_size),
        "nu_samples": nu_samples,
    }


def compute_acf(x: np.ndarray, max_lag: int = 200) -> np.ndarray:
    """
    Compute normalized autocorrelation function of 1D array x
    up to max_lag. Returns array of length max_lag+1.
    """
    x = x - np.mean(x)
    var = np.var(x)
    if var < 1e-12:
        return np.zeros(max_lag + 1)
    acf = np.array(
        [np.mean(x[: len(x) - lag] * x[lag:]) / var for lag in range(max_lag + 1)]
    )
    return acf


def compute_ess(x: np.ndarray, max_lag: int = 500) -> float:
    """
    Estimate ESS via Geyer's initial positive sequence estimator.

    Forms pairs Gamma_k = rho(2k) + rho(2k+1) and sums until
    a pair first goes negative. This avoids the arbitrary 0.05
    threshold and is robust to noisy ACF estimates.

    tau = 1 + 2 * sum of Gamma_k until first negative pair
    ESS = n / tau

    Reference: Geyer (1992), "Practical Markov Chain Monte Carlo"
    """
    n = len(x)
    acf = compute_acf(x, max_lag=max_lag)

    tau = 1.0
    for k in range(1, max_lag // 2):
        gamma_k = acf[2 * k] + acf[2 * k + 1]
        if gamma_k < 0:
            break
        tau += 2 * gamma_k
    else:
        # Loop completed without finding negative pair —
        # chain is too short or mixing too slowly to estimate tau reliably
        print(
            f"  WARNING: Geyer estimator did not terminate within "
            f"{max_lag} lags — ESS is a lower bound. "
            f"Consider running longer chain."
        )

    return float(n / tau)


def main() -> None:
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    # MCMC setup
    n_samples = 2000
    n_warmup = 500
    rp_k_values = [20, 50, 100]

    # Fake data and kernel
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
            "total_time": dense_stats["total_mcmc_time"],
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

    for k in rp_k_values:
        # RPCholesky: O(Nk^2) vs dense Cholesky O(N^3)
        t0 = time.perf_counter()
        lra = arpcholesky(A, k=k, b=10)
        F = lra.get_left_factor()  # shape (N, k_eff)
        rp_factor_time = time.perf_counter() - t0

        approx_err = float(
            np.linalg.norm(K_dense - (F @ F.T), "fro") / (fro_norm_K + 1e-12)
        )

        if k == 50:
            rp_stats = run_rwm(
                factor=F,
                y=y,
                n_samples=n_samples,
                n_warmup=n_warmup,
                seed=123 + k,
            )
            rp50_factor = F
            rp50_nu_samples = rp_stats["nu_samples"]
            rp50_logp_trace = rp_stats["logp_trace"]
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
                "total_time": rp_stats["total_mcmc_time"],
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

    dense_logp_post = dense_stats["logp_trace"][n_warmup:]
    rp50_logp_post = rp50_logp_trace[n_warmup:]
    dense_nu0 = dense_stats["nu_samples"][:, 0]
    rp50_nu0 = rp50_nu_samples[:, 0]

    acf_dense = compute_acf(dense_logp_post, max_lag=500)
    acf_rp50 = compute_acf(rp50_logp_post, max_lag=500)

    ess_dense_logp = compute_ess(dense_logp_post, max_lag=500)
    ess_rp50_logp = compute_ess(rp50_logp_post, max_lag=500)
    ess_dense_nu0 = compute_ess(dense_nu0, max_lag=500)
    ess_rp50_nu0 = compute_ess(rp50_nu0, max_lag=500)

    acf_dense_full = compute_acf(dense_logp_post, max_lag=500)
    acf_rp50_full = compute_acf(rp50_logp_post, max_lag=500)

    def _geyer_tau(acf, max_lag=500):
        """Return tau directly from ACF using Geyer's rule."""
        tau = 1.0
        for k in range(1, max_lag // 2):
            gamma_k = acf[2 * k] + acf[2 * k + 1]
            if gamma_k < 0:
                break
            tau += 2 * gamma_k
        return tau

    tau_dense_logp = _geyer_tau(acf_dense_full)
    tau_rp50_logp = _geyer_tau(acf_rp50_full)

    diag_fmt = "{:<12} {:>12} {:>13} {:>12}"
    print(diag_fmt.format("Method", "ESS (logp)", "ESS (nu[0])", "tau (logp)"))
    print(diag_fmt.format("Dense", f"{ess_dense_logp:.1f}", f"{ess_dense_nu0:.1f}", f"{tau_dense_logp:.2f}"))
    print(diag_fmt.format("RPChol-k50", f"{ess_rp50_logp:.1f}", f"{ess_rp50_nu0:.1f}", f"{tau_rp50_logp:.2f}"))

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
    plt.savefig(os.path.join(data_dir, "exp1_acf_comparison.png"), dpi=160)
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
    plt.savefig(os.path.join(data_dir, "exp1_classification_scatter.png"), dpi=160)
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
    plt.savefig(os.path.join(data_dir, "exp1_decision_boundary.png"), dpi=160)
    plt.close()

    # Plot 7: uncertainty on training points.
    plt.figure(figsize=(6, 5))
    su = plt.scatter(X[:, 0], X[:, 1], c=p_std, cmap="Oranges", s=10)
    plt.colorbar(su, label="Std of P(y=1)")
    plt.title("Posterior uncertainty — RPChol k=50")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp1_uncertainty.png"), dpi=160)
    plt.close()

    # Plot 1: bar chart factor computation time
    labels_factor = ["Dense"] + [x[0] for x in rp_factor_times]
    vals_factor = [dense_factor_time] + [x[1] for x in rp_factor_times]
    plt.figure(figsize=(7, 4))
    plt.bar(labels_factor, vals_factor, color=["tab:gray", "tab:blue", "tab:orange", "tab:green"])
    plt.ylabel("Seconds")
    plt.title("Factor computation time")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp1_factor_time_bar.png"), dpi=160)
    plt.close()

    # Plot 2: bar chart total MCMC time
    labels_total = ["Dense"] + [x[0] for x in rp_total_times]
    vals_total = [dense_stats["total_mcmc_time"]] + [x[1] for x in rp_total_times]
    plt.figure(figsize=(7, 4))
    plt.bar(labels_total, vals_total, color=["tab:gray", "tab:blue", "tab:orange", "tab:green"])
    plt.ylabel("Seconds")
    plt.title("Total MCMC runtime (post-warmup)")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp1_total_mcmc_time_bar.png"), dpi=160)
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
    plt.savefig(os.path.join(data_dir, "exp1_approx_error_vs_k.png"), dpi=160)
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
        plt.savefig(os.path.join(data_dir, "exp1_trace_dense_vs_rp50.png"), dpi=160)
        plt.close()

    np.save(
        os.path.join(data_dir, "exp1_results.npy"),
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
    print("- data/exp1_factor_time_bar.png")
    print("- data/exp1_total_mcmc_time_bar.png")
    print("- data/exp1_approx_error_vs_k.png")
    print("- data/exp1_trace_dense_vs_rp50.png")
    print("- data/exp1_acf_comparison.png")
    print("- data/exp1_classification_scatter.png")
    print("- data/exp1_decision_boundary.png")
    print("- data/exp1_uncertainty.png")
    print("- data/exp1_results.npy")


if __name__ == "__main__":
    main()

