"""
Experiment 1b: Compare RWM vs MALA on the same RPCholesky k=50 factor.

This experiment uses fake GP binary classification data (two Gaussian blobs),
constructs one RPCholesky factor F, and compares:
- Random Walk Metropolis (RWM)
- Metropolis-adjusted Langevin Algorithm (MALA)

under the same non-centered parameterization f = F @ nu.
"""

from __future__ import annotations

import os
import sys
import time

import blackjax
import emcee
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np
from jax.nn import sigmoid as jax_sigmoid
from scipy.special import expit

# JAX + blackjax required: pip install jax jaxlib blackjax


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


def compute_tau_emcee(x: np.ndarray) -> float:
    """
    Estimate integrated autocorrelation time using emcee's implementation.

    Falls back to NaN if the chain is too short for a stable estimate.
    """
    x = np.asarray(x, dtype=float)
    try:
        tau = emcee.autocorr.integrated_time(x, quiet=True)
    except Exception as err:
        print(f"  WARNING: emcee tau estimate failed: {err}")
        return float("nan")

    tau = np.asarray(tau, dtype=float).reshape(-1)
    if tau.size == 0:
        return float("nan")
    return float(tau[0])


def compute_ess_from_tau(n: int, tau: float) -> float:
    """Convert integrated autocorrelation time into effective sample size."""
    if not np.isfinite(tau) or tau <= 0:
        return float("nan")
    return float(n / tau)


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

    # RPCholesky factor F: O(Nk) per step, nu lives in R^k not R^N.
    for i in range(total_steps):
        t0 = time.perf_counter()
        nu_prop = nu + step_size * rng.standard_normal(dim)
        logp_prop = log_posterior(nu_prop, factor, y)

        if np.log(rng.random()) < (logp_prop - logp):
            nu = nu_prop
            logp = logp_prop
            accepts[i] = True

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
            nu_samples[post_idx, :] = nu
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
        "logp_trace": logp_trace,
        "final_step_size": float(step_size),
        "nu_samples": nu_samples,
    }


# Gradient is available analytically — enables MALA without autograd.
def grad_log_posterior(nu: np.ndarray, factor: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Analytical gradient of log posterior wrt nu.
    d/dnu [ sum Bernoulli_loglik(sigmoid(F@nu), y) - 0.5*||nu||^2 ]
      = F.T @ (y - sigmoid(F@nu)) - nu
    """
    f = factor @ nu
    p = expit(f)
    return factor.T @ (y - p) - nu


def run_mala(
    factor: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    n_warmup: int,
    seed: int,
    target_accept: float = 0.57,
    adapt_interval: int = 50,
):
    """Run MALA with MH correction and warmup adaptation."""
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

    def log_q(nu_to, nu_from, grad_from, step):
        diff = nu_to - nu_from - (step**2 / 2.0) * grad_from
        return -0.5 * np.dot(diff, diff) / (step**2)

    # RPCholesky factor F: O(Nk) per step, nu lives in R^k not R^N.
    for i in range(total_steps):
        t0 = time.perf_counter()

        grad_current = grad_log_posterior(nu, factor, y)
        nu_prop = (
            nu
            + (step_size**2 / 2.0) * grad_current
            + step_size * rng.standard_normal(dim)
        )
        logp_prop = log_posterior(nu_prop, factor, y)
        grad_prop = grad_log_posterior(nu_prop, factor, y)

        log_accept = (
            logp_prop
            - logp
            + log_q(nu, nu_prop, grad_prop, step_size)
            - log_q(nu_prop, nu, grad_current, step_size)
        )

        if np.log(rng.random()) < log_accept:
            nu = nu_prop
            logp = logp_prop
            accepts[i] = True

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
            nu_samples[post_idx, :] = nu
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
        "logp_trace": logp_trace,
        "final_step_size": float(step_size),
        "nu_samples": nu_samples,
    }


def log_posterior_jax(nu, F, y):
    """
    JAX version of log_posterior — identical math, jax.numpy ops.
    Returns a scalar jax array so gradients are available automatically.
    """
    f = jnp.dot(F, nu)
    p = jax_sigmoid(f)
    log_lik = jnp.sum(y * jnp.log(p + 1e-10) + (1.0 - y) * jnp.log(1.0 - p + 1e-10))
    log_prior = -0.5 * jnp.dot(nu, nu)
    return log_lik + log_prior


def run_hmc(
    factor: np.ndarray, y: np.ndarray, n_samples: int, n_warmup: int, seed: int
) -> dict:
    F_jax = jnp.asarray(factor, dtype=jnp.float64)
    y_jax = jnp.asarray(y, dtype=jnp.float64)
    dim = factor.shape[1]

    def _logdensity(nu):
        return log_posterior_jax(nu, F_jax, y_jax)

    init_position = jnp.zeros(dim, dtype=jnp.float64)
    rng_key = jax.random.PRNGKey(seed)

    # Warmup (step size + mass matrix adaptation)
    adapt = blackjax.window_adaptation(
        blackjax.nuts,
        _logdensity,
        target_acceptance_rate=0.9,
        progress_bar=False,
    )

    t0 = time.perf_counter()
    rng_key, warmup_key = jax.random.split(rng_key)
    adapt_results, _ = adapt.run(warmup_key, init_position, num_steps=n_warmup)
    warmup_time = time.perf_counter() - t0
    state = adapt_results.state
    params = adapt_results.parameters

    # Sampling with tuned NUTS kernel
    nuts = blackjax.nuts(_logdensity, **params)
    nu_samples = np.zeros((n_samples, dim), dtype=np.float64)
    logp_trace = np.zeros(n_samples, dtype=np.float64)
    accept_rates = np.zeros(n_samples, dtype=np.float64)

    t0 = time.perf_counter()
    for i in range(n_samples):
        rng_key, step_key = jax.random.split(rng_key)
        state, info = nuts.step(step_key, state)
        nu_samples[i, :] = np.asarray(state.position)
        logp_trace[i] = float(state.logdensity)
        accept_rates[i] = float(info.acceptance_rate)

    sampling_time = time.perf_counter() - t0
    per_step_time = sampling_time / n_samples
    accept_rate = float(np.mean(accept_rates))
    step_size = float(np.asarray(params["step_size"]))

    return {
        "per_step_time": per_step_time,
        "warmup_time": float(warmup_time),
        "sampling_time": float(sampling_time),
        "total_mcmc_time": float(sampling_time),
        "total_sampler_time": float(warmup_time + sampling_time),
        "accept_rate": accept_rate,
        "logp_trace": logp_trace,
        "final_step_size": step_size,
        "nu_samples": nu_samples,
    }


def main() -> None:
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Setup
    n_samples = 500
    n_warmup = 50
    k_values = [2, 5]

    # Data and kernel matrix
    X, y = make_fake_blobs(seed=42)
    A = KernelMatrix(X, kernel="gaussian", bandwidth=1.0)
    results = []
    acf_k200_rwm = None
    acf_k200_mala = None
    acf_k200_hmc = None

    # Sweeping k to test whether MALA's ESS/sec advantage grows with
    # dimension, as predicted by Langevin diffusion theory
    for k in k_values:
        factor_start = time.perf_counter()
        lra = arpcholesky(A, k=k, b=10)
        F = lra.get_left_factor()
        factor_time = time.perf_counter() - factor_start
        n_warmup_k = n_warmup * max(1, k // 50)

        # RPCholesky factor F: O(Nk) per step, nu lives in R^k not R^N.
        rwm_stats = run_rwm(F, y, n_samples=n_samples, n_warmup=n_warmup_k, seed=123)
        mala_stats = run_mala(F, y, n_samples=n_samples, n_warmup=n_warmup_k, seed=456)
        hmc_stats = run_hmc(F, y, n_samples=n_samples, n_warmup=n_warmup_k, seed=789)

        rwm_logp_post = rwm_stats["logp_trace"][n_warmup_k:]
        mala_logp_post = mala_stats["logp_trace"][n_warmup_k:]
        # PyMC warmup handled internally; returned trace is post-warmup draws.
        hmc_logp_post = hmc_stats["logp_trace"]

        # acf_rwm = compute_acf(rwm_logp_post, max_lag=500)
        # acf_mala = compute_acf(mala_logp_post, max_lag=500)
        # acf_hmc = compute_acf(hmc_logp_post, max_lag=500)

        tau_rwm = compute_tau_emcee(rwm_logp_post)
        tau_mala = compute_tau_emcee(mala_logp_post)
        tau_hmc = compute_tau_emcee(hmc_logp_post)

        ess_rwm_logp = compute_ess_from_tau(len(rwm_logp_post), tau_rwm)
        ess_mala_logp = compute_ess_from_tau(len(mala_logp_post), tau_mala)
        ess_hmc_logp = compute_ess_from_tau(len(hmc_logp_post), tau_hmc)
        ess_rwm_nu0 = compute_ess_from_tau(
            len(rwm_stats["nu_samples"][:, 0]),
            compute_tau_emcee(rwm_stats["nu_samples"][:, 0]),
        )
        ess_mala_nu0 = compute_ess_from_tau(
            len(mala_stats["nu_samples"][:, 0]),
            compute_tau_emcee(mala_stats["nu_samples"][:, 0]),
        )
        ess_hmc_nu0 = compute_ess_from_tau(
            len(hmc_stats["nu_samples"][:, 0]),
            compute_tau_emcee(hmc_stats["nu_samples"][:, 0]),
        )

        ess_per_sec_rwm = ess_rwm_logp / max(rwm_stats["total_mcmc_time"], 1e-12)
        ess_per_sec_mala = ess_mala_logp / max(mala_stats["total_mcmc_time"], 1e-12)
        ess_per_sec_hmc = ess_hmc_logp / max(hmc_stats["total_mcmc_time"], 1e-12)

        results.append(
            {
                "k": k,
                "sampler": "RWM",
                "step_size": rwm_stats["final_step_size"],
                "factor_time": float(factor_time),
                "warmup_time": rwm_stats["warmup_time"],
                "sampling_time": rwm_stats["sampling_time"],
                "accept_rate": rwm_stats["accept_rate"],
                "per_step_time": rwm_stats["per_step_time"],
                "total_time": rwm_stats["total_mcmc_time"],
                "total_model_compute_time": float(factor_time) + rwm_stats["total_sampler_time"],
                "ess_logp": ess_rwm_logp,
                "ess_nu0": ess_rwm_nu0,
                "ess_per_sec": ess_per_sec_rwm,
                "tau": tau_rwm,
            }
        )
        results.append(
            {
                "k": k,
                "sampler": "MALA",
                "step_size": mala_stats["final_step_size"],
                "factor_time": float(factor_time),
                "warmup_time": mala_stats["warmup_time"],
                "sampling_time": mala_stats["sampling_time"],
                "accept_rate": mala_stats["accept_rate"],
                "per_step_time": mala_stats["per_step_time"],
                "total_time": mala_stats["total_mcmc_time"],
                "total_model_compute_time": float(factor_time) + mala_stats["total_sampler_time"],
                "ess_logp": ess_mala_logp,
                "ess_nu0": ess_mala_nu0,
                "ess_per_sec": ess_per_sec_mala,
                "tau": tau_mala,
            }
        )
        results.append(
            {
                "k": k,
                "sampler": "HMC",
                "step_size": hmc_stats["final_step_size"],
                "factor_time": float(factor_time),
                "warmup_time": hmc_stats["warmup_time"],
                "sampling_time": hmc_stats["sampling_time"],
                "accept_rate": hmc_stats["accept_rate"],
                "per_step_time": hmc_stats["per_step_time"],
                "total_time": hmc_stats["total_mcmc_time"],
                "total_model_compute_time": float(factor_time) + hmc_stats["total_sampler_time"],
                "ess_logp": ess_hmc_logp,
                "ess_nu0": ess_hmc_nu0,
                "ess_per_sec": ess_per_sec_hmc,
                "tau": tau_hmc,
            }
        )

        if k == 200:
            acf_k200_rwm = acf_rwm
            acf_k200_mala = acf_mala
            acf_k200_hmc = acf_hmc

    # Results table grouped by k.
    fmt = "{:<8} {:>10} {:>8} {:>12} {:>10} {:>8} {:>10} {:>8}"
    for k in k_values:
        n_warmup_k = n_warmup * max(1, k // 50)
        print(f"k={k}  (n_warmup={n_warmup_k})")
        print(
            fmt.format(
                "Sampler",
                "Step size",
                "Accept",
                "Per-step(s)",
                "Total(s)",
                "ESS",
                "ESS/sec",
                "tau",
            )
        )
        for row in [r for r in results if r["k"] == k]:
            print(
                fmt.format(
                    row["sampler"],
                    f"{row['step_size']:.6f}",
                    f"{row['accept_rate']:.3f}",
                    f"{row['per_step_time']:.6f}",
                    f"{row['total_time']:.3f}",
                    f"{row['ess_logp']:.1f}",
                    f"{row['ess_per_sec']:.2f}",
                    f"{row['tau']:.2f}",
                )
            )
        print()

    # Helpers for plotting trend lines.
    def series(metric: str, sampler: str):
        out = []
        for k in k_values:
            row = next(r for r in results if r["k"] == k and r["sampler"] == sampler)
            out.append(row[metric])
        return np.array(out, dtype=float)

    essps_rwm = series("ess_per_sec", "RWM")
    essps_mala = series("ess_per_sec", "MALA")
    essps_hmc = series("ess_per_sec", "HMC")
    tau_rwm_series = series("tau", "RWM")
    tau_mala_series = series("tau", "MALA")
    tau_hmc_series = series("tau", "HMC")
    step_rwm = series("per_step_time", "RWM")
    step_mala = series("per_step_time", "MALA")
    step_hmc = series("per_step_time", "HMC")

    # Plot 1: ESS/sec vs k with MALA/RWM and HMC/RWM ratio annotations.
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, essps_rwm, marker="o", color="tab:gray", label="RWM")
    plt.plot(k_values, essps_mala, marker="o", color="tab:orange", label="MALA")
    plt.plot(k_values, essps_hmc, marker="o", color="tab:green", label="HMC")
    for i, k in enumerate(k_values):
        ratio_mala = essps_mala[i] / max(essps_rwm[i], 1e-12)
        ratio_hmc = essps_hmc[i] / max(essps_rwm[i], 1e-12)
        y_text = max(essps_mala[i], essps_rwm[i], essps_hmc[i])
        plt.annotate(
            f"M/R x{ratio_mala:.2f}\nH/R x{ratio_hmc:.2f}",
            (k, y_text),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
        )
    plt.xlabel("k")
    plt.ylabel("ESS per second")
    plt.title("Sampling efficiency vs rank k (RPChol factor)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp1b_ess_per_sec_vs_k.png"), dpi=160)
    plt.close()

    # Plot 2: tau vs k.
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, tau_rwm_series, marker="o", color="tab:gray", label="RWM")
    plt.plot(k_values, tau_mala_series, marker="o", color="tab:orange", label="MALA")
    plt.plot(k_values, tau_hmc_series, marker="o", color="tab:green", label="HMC")
    plt.xlabel("k")
    plt.ylabel("Integrated autocorrelation time (tau)")
    plt.title("Autocorrelation time vs rank k")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp1b_tau_vs_k.png"), dpi=160)
    plt.close()

    # Plot 3: per-step runtime vs k.
    # Both RWM and MALA use the RPChol factor, so per-step cost should scale ~O(Nk).
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, step_rwm, marker="o", color="tab:gray", label="RWM")
    plt.plot(k_values, step_mala, marker="o", color="tab:orange", label="MALA")
    plt.plot(k_values, step_hmc, marker="o", color="tab:green", label="HMC")
    plt.xlabel("k")
    plt.ylabel("Per-step runtime (s)")
    plt.title("Per-step runtime vs rank k")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp1b_step_time_vs_k.png"), dpi=160)
    plt.close()

    # Plot 4: ACF comparison at k=200.
    if acf_k200_rwm is not None and acf_k200_mala is not None and acf_k200_hmc is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(acf_k200_rwm, label="RWM", color="tab:gray")
        plt.plot(acf_k200_mala, label="MALA", color="tab:orange")
        plt.plot(acf_k200_hmc, label="HMC", color="tab:green")
        plt.axhline(0.05, color="black", linestyle="--", linewidth=0.8, label="0.05 threshold")
        plt.axhline(0.0, color="black", linestyle="-", linewidth=0.5)
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.title("ACF at k=200 — RWM vs MALA")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(0, 500)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "exp1b_acf_k200.png"), dpi=160)
        plt.close()

    print("Saved:")
    print("- data/exp1b_ess_per_sec_vs_k.png")
    print("- data/exp1b_tau_vs_k.png")
    print("- data/exp1b_step_time_vs_k.png")
    print("- data/exp1b_acf_k200.png")


if __name__ == "__main__":
    main()

