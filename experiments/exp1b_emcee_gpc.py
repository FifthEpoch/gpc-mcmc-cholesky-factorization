"""
Experiment 1b: compare three samplers on the RPCholesky GP classification target.

- emcee-based RWM using a Gaussian random-walk proposal
- emcee-based MALA
- a self-contained HMC implementation

This keeps the same non-centered parameterization f = F @ nu used in the
other experiments. emcee drives the RWM and MALA runs; HMC is added directly
because emcee does not provide an HMC kernel.
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


def log_posterior_batch(coords: np.ndarray, factor: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized log posterior over shape (nwalkers, dim)."""
    f = coords @ factor.T
    p = expit(f)
    log_lik = np.sum(y[None, :] * np.log(p + 1e-10), axis=1)
    log_lik += np.sum((1 - y)[None, :] * np.log(1 - p + 1e-10), axis=1)
    log_prior = -0.5 * np.sum(coords * coords, axis=1)
    return log_lik + log_prior


def grad_log_posterior_batch(
    coords: np.ndarray, factor: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Vectorized gradient wrt nu for shape (nwalkers, dim)."""
    f = coords @ factor.T
    p = expit(f)
    return (y[None, :] - p) @ factor - coords


def grad_log_posterior(nu: np.ndarray, factor: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient wrt a single latent vector nu."""
    f = factor @ nu
    p = expit(f)
    return factor.T @ (y - p) - nu


def compute_tau_emcee(chain: np.ndarray) -> float:
    """
    Estimate integrated autocorrelation time using emcee's implementation.

    Expects an array shaped either (nsteps,) or (nsteps, nwalkers).
    """
    try:
        tau = emcee.autocorr.integrated_time(chain, quiet=True)
    except Exception as err:
        print(f"  WARNING: emcee tau estimate failed: {err}")
        return float("nan")

    tau = np.asarray(tau, dtype=float).reshape(-1)
    if tau.size == 0:
        return float("nan")
    return float(np.nanmean(tau))


def compute_ess_from_tau(n_steps: int, n_walkers: int, tau: float) -> float:
    """Convert tau to an approximate ensemble ESS."""
    if not np.isfinite(tau) or tau <= 0:
        return float("nan")
    return float((n_steps * n_walkers) / tau)


def posterior_mean_prob(
    factor: np.ndarray,
    nu_samples: np.ndarray,
    seed: int,
    max_draws: int = 400,
) -> np.ndarray:
    """Estimate mean posterior class probabilities at observed inputs."""
    if nu_samples.ndim != 2:
        raise ValueError("nu_samples must have shape (n_draws, dim)")

    rng = np.random.default_rng(seed)
    n_draws = nu_samples.shape[0]
    if n_draws > max_draws:
        idx = rng.choice(n_draws, size=max_draws, replace=False)
        nu_use = nu_samples[idx, :]
    else:
        nu_use = nu_samples

    logits = factor @ nu_use.T
    return np.mean(expit(logits), axis=1)


def make_mala_move(
    factor: np.ndarray,
    y: np.ndarray,
    step_size: float,
) -> emcee.moves.MHMove:
    """Create an emcee MH move that uses a MALA proposal."""

    def proposal_function(coords: np.ndarray, random) -> tuple[np.ndarray, np.ndarray]:
        grads = grad_log_posterior_batch(coords, factor, y)
        noise = random.randn(*coords.shape)
        drift = 0.5 * (step_size**2) * grads
        proposals = coords + drift + step_size * noise

        prop_grads = grad_log_posterior_batch(proposals, factor, y)

        forward_diff = proposals - coords - drift
        reverse_diff = coords - proposals - 0.5 * (step_size**2) * prop_grads

        # emcee expects log q(x | x') - log q(x' | x).
        forward = -0.5 * np.sum(forward_diff * forward_diff, axis=1) / (step_size**2)
        reverse = -0.5 * np.sum(reverse_diff * reverse_diff, axis=1) / (step_size**2)
        factors = reverse - forward
        return proposals, factors

    return emcee.moves.MHMove(proposal_function)


def run_emcee_sampler(
    factor: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    n_warmup: int,
    n_walkers: int,
    seed: int,
    move,
    init_scale: float = 0.1,
) -> dict:
    """Run emcee and return timing/statistics."""
    rng = np.random.RandomState(seed)
    dim = factor.shape[1]
    initial_state = init_scale * rng.randn(n_walkers, dim)

    sampler = emcee.EnsembleSampler(
        nwalkers=n_walkers,
        ndim=dim,
        log_prob_fn=log_posterior_batch,
        args=(factor, y),
        moves=move,
        vectorize=True,
    )

    t0 = time.perf_counter()
    sampler.run_mcmc(initial_state, n_warmup, progress=False)
    warmup_time = time.perf_counter() - t0
    sampler.reset()

    t0 = time.perf_counter()
    sampler.run_mcmc(None, n_samples, progress=False)
    sample_time = time.perf_counter() - t0

    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    flat_chain = sampler.get_chain(flat=True)
    flat_log_prob = sampler.get_log_prob(flat=True)

    return {
        "per_step_time": float(sample_time / max(n_samples, 1)),
        "total_mcmc_time": float(sample_time),
        "warmup_time": float(warmup_time),
        "accept_rate": float(np.mean(sampler.acceptance_fraction)),
        "nu_samples": flat_chain,
        "chain": chain,
        "logp_trace": np.mean(log_prob, axis=1),
        "logp_by_walker": log_prob,
        "flat_logp": flat_log_prob,
    }


def run_hmc(
    factor: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    n_warmup: int,
    seed: int,
    step_size: float,
    n_leapfrog: int,
) -> dict:
    """Run a simple Euclidean HMC sampler with fixed mass matrix."""
    rng = np.random.default_rng(seed)
    dim = factor.shape[1]
    total_steps = n_warmup + n_samples

    nu = np.zeros(dim, dtype=float)
    logp = log_posterior(nu, factor, y)

    step_times = np.zeros(total_steps, dtype=float)
    logp_trace = np.zeros(total_steps, dtype=float)
    accepts = np.zeros(total_steps, dtype=bool)
    nu_samples = np.zeros((n_samples, dim), dtype=float)
    post_idx = 0

    for i in range(total_steps):
        t0 = time.perf_counter()

        current_nu = nu.copy()
        current_logp = logp
        momentum = rng.standard_normal(dim)
        current_momentum = momentum.copy()

        grad = grad_log_posterior(current_nu, factor, y)
        proposal_nu = current_nu.copy()
        proposal_p = momentum + 0.5 * step_size * grad

        for leapfrog_idx in range(n_leapfrog):
            proposal_nu = proposal_nu + step_size * proposal_p
            grad = grad_log_posterior(proposal_nu, factor, y)
            if leapfrog_idx != n_leapfrog - 1:
                proposal_p = proposal_p + step_size * grad

        proposal_p = proposal_p + 0.5 * step_size * grad
        proposal_p = -proposal_p

        proposal_logp = log_posterior(proposal_nu, factor, y)
        current_h = -current_logp + 0.5 * np.dot(current_momentum, current_momentum)
        proposal_h = -proposal_logp + 0.5 * np.dot(proposal_p, proposal_p)
        log_accept = current_h - proposal_h

        if np.log(rng.random()) < log_accept:
            nu = proposal_nu
            logp = proposal_logp
            accepts[i] = True

        step_times[i] = time.perf_counter() - t0
        logp_trace[i] = logp
        if i >= n_warmup:
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
        "nu_samples": nu_samples,
        "step_size": float(step_size),
        "n_leapfrog": int(n_leapfrog),
    }


def main() -> None:
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    n_samples = 5000
    n_warmup = 200
    k_values = [10, 20, 50, 100, 200]

    X, y = make_fake_blobs(seed=42)
    A = KernelMatrix(X, kernel="gaussian", bandwidth=1.0)
    results = []
    trace_example = {}
    posterior_k200 = {}

    for k in k_values:
        lra = arpcholesky(A, k=k, b=10)
        F = lra.get_left_factor()
        dim = F.shape[1]
        n_walkers = max(2 * dim + 2, 24)

        gaussian_step = (2.38 / np.sqrt(dim)) ** 2
        mala_step = 0.6 / np.sqrt(dim)

        # emcee's GaussianMove is exactly a Gaussian random-walk MH proposal here.
        gaussian_move = emcee.moves.GaussianMove(cov=gaussian_step, mode="vector")
        mala_move = make_mala_move(F, y, step_size=mala_step)

        gaussian_stats = run_emcee_sampler(
            F,
            y,
            n_samples=n_samples,
            n_warmup=n_warmup,
            n_walkers=n_walkers,
            seed=1000 + k,
            move=gaussian_move,
        )
        mala_stats = run_emcee_sampler(
            F,
            y,
            n_samples=n_samples,
            n_warmup=n_warmup,
            n_walkers=n_walkers,
            seed=2000 + k,
            move=mala_move,
        )
        hmc_step = 0.08 / np.sqrt(max(dim, 1))
        hmc_leapfrog = 12
        hmc_stats = run_hmc(
            F,
            y,
            n_samples=n_samples,
            n_warmup=n_warmup,
            seed=3000 + k,
            step_size=hmc_step,
            n_leapfrog=hmc_leapfrog,
        )

        tau_gaussian = compute_tau_emcee(gaussian_stats["logp_by_walker"])
        tau_mala = compute_tau_emcee(mala_stats["logp_by_walker"])
        tau_hmc = compute_tau_emcee(hmc_stats["logp_trace"][..., None])
        ess_gaussian = compute_ess_from_tau(n_samples, n_walkers, tau_gaussian)
        ess_mala = compute_ess_from_tau(n_samples, n_walkers, tau_mala)
        ess_hmc = compute_ess_from_tau(n_samples, 1, tau_hmc)

        essps_gaussian = ess_gaussian / max(gaussian_stats["total_mcmc_time"], 1e-12)
        essps_mala = ess_mala / max(mala_stats["total_mcmc_time"], 1e-12)
        essps_hmc = ess_hmc / max(hmc_stats["total_mcmc_time"], 1e-12)

        results.append(
            {
                "k": dim,
                "sampler": "emcee-RWM",
                "n_walkers": n_walkers,
                "step_size": float(np.sqrt(gaussian_step)),
                "accept_rate": gaussian_stats["accept_rate"],
                "per_step_time": gaussian_stats["per_step_time"],
                "total_time": gaussian_stats["total_mcmc_time"],
                "ess_logp": ess_gaussian,
                "ess_per_sec": essps_gaussian,
                "tau": tau_gaussian,
            }
        )
        results.append(
            {
                "k": dim,
                "sampler": "emcee-MALA",
                "n_walkers": n_walkers,
                "step_size": mala_step,
                "accept_rate": mala_stats["accept_rate"],
                "per_step_time": mala_stats["per_step_time"],
                "total_time": mala_stats["total_mcmc_time"],
                "ess_logp": ess_mala,
                "ess_per_sec": essps_mala,
                "tau": tau_mala,
            }
        )
        results.append(
            {
                "k": dim,
                "sampler": "HMC",
                "n_walkers": 1,
                "step_size": hmc_stats["step_size"],
                "accept_rate": hmc_stats["accept_rate"],
                "per_step_time": hmc_stats["per_step_time"],
                "total_time": hmc_stats["total_mcmc_time"],
                "ess_logp": ess_hmc,
                "ess_per_sec": essps_hmc,
                "tau": tau_hmc,
            }
        )

        if dim == 50:
            trace_example["RWM"] = gaussian_stats["logp_trace"]
            trace_example["MALA"] = mala_stats["logp_trace"]
            trace_example["HMC"] = hmc_stats["logp_trace"][n_warmup:]

        if dim == 200:
            posterior_k200["RWM"] = posterior_mean_prob(
                F,
                gaussian_stats["nu_samples"],
                seed=4000 + k,
            )
            posterior_k200["MALA"] = posterior_mean_prob(
                F,
                mala_stats["nu_samples"],
                seed=5000 + k,
            )
            posterior_k200["HMC"] = posterior_mean_prob(
                F,
                hmc_stats["nu_samples"],
                seed=6000 + k,
            )

    fmt = "{:<18} {:>8} {:>10} {:>8} {:>12} {:>10} {:>10} {:>8}"
    for k in k_values:
        print(f"k={k}")
        print(
            fmt.format(
                "Sampler",
                "Walkers",
                "Step size",
                "Accept",
                "Per-step(s)",
                "ESS",
                "ESS/sec",
                "tau",
            )
        )
        for row in [r for r in results if r["k"] == k]:
            print(
                fmt.format(
                    row["sampler"],
                    f"{row['n_walkers']}",
                    f"{row['step_size']:.4f}",
                    f"{row['accept_rate']:.3f}",
                    f"{row['per_step_time']:.6f}",
                    f"{row['ess_logp']:.1f}",
                    f"{row['ess_per_sec']:.2f}",
                    f"{row['tau']:.2f}",
                )
            )
        print()

    def series(metric: str, sampler: str):
        return np.array(
            [
                next(r for r in results if r["k"] == k and r["sampler"] == sampler)[metric]
                for k in k_values
            ],
            dtype=float,
        )

    essps_gaussian = series("ess_per_sec", "emcee-RWM")
    essps_mala = series("ess_per_sec", "emcee-MALA")
    essps_hmc = series("ess_per_sec", "HMC")
    accept_gaussian = series("accept_rate", "emcee-RWM")
    accept_mala = series("accept_rate", "emcee-MALA")
    accept_hmc = series("accept_rate", "HMC")
    tau_gaussian = series("tau", "emcee-RWM")
    tau_mala = series("tau", "emcee-MALA")
    tau_hmc = series("tau", "HMC")
    step_gaussian = series("per_step_time", "emcee-RWM")
    step_mala = series("per_step_time", "emcee-MALA")
    step_hmc = series("per_step_time", "HMC")

    plt.figure(figsize=(7, 4))
    plt.plot(k_values, essps_gaussian, marker="o", color="tab:gray", label="emcee RWM")
    plt.plot(k_values, essps_mala, marker="o", color="tab:orange", label="emcee MALA")
    plt.plot(k_values, essps_hmc, marker="o", color="tab:green", label="HMC")
    plt.xlabel("k")
    plt.ylabel("ESS per second")
    plt.title("Sampler efficiency vs rank k")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp1b_emcee_ess_per_sec_vs_k.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(k_values, tau_gaussian, marker="o", color="tab:gray", label="emcee RWM")
    plt.plot(k_values, tau_mala, marker="o", color="tab:orange", label="emcee MALA")
    plt.plot(k_values, tau_hmc, marker="o", color="tab:green", label="HMC")
    plt.xlabel("k")
    plt.ylabel("Integrated autocorrelation time (tau)")
    plt.title("Sampler tau vs rank k")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp1b_emcee_tau_vs_k.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(k_values, step_gaussian, marker="o", color="tab:gray", label="emcee RWM")
    plt.plot(k_values, step_mala, marker="o", color="tab:orange", label="emcee MALA")
    plt.plot(k_values, step_hmc, marker="o", color="tab:green", label="HMC")
    plt.xlabel("k")
    plt.ylabel("Per-step runtime (s)")
    plt.title("Sampler per-step runtime vs rank k")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp1b_emcee_step_time_vs_k.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(k_values, accept_gaussian, marker="o", color="tab:gray", label="emcee RWM")
    plt.plot(k_values, accept_mala, marker="o", color="tab:orange", label="emcee MALA")
    plt.plot(k_values, accept_hmc, marker="o", color="tab:green", label="HMC")
    plt.xlabel("k")
    plt.ylabel("Acceptance rate")
    plt.title("Sampler acceptance rate vs rank k")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp1b_emcee_accept_rate_vs_k.png"), dpi=160)
    plt.close()

    if (
        "RWM" in trace_example
        and "MALA" in trace_example
        and "HMC" in trace_example
    ):
        plt.figure(figsize=(7, 4))
        plt.plot(trace_example["RWM"], color="tab:gray", label="emcee RWM")
        plt.plot(trace_example["MALA"], color="tab:orange", label="emcee MALA")
        plt.plot(trace_example["HMC"], color="tab:green", label="HMC")
        plt.xlabel("Iteration")
        plt.ylabel("Log posterior")
        plt.title("Sampler log posterior trace at k=50")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "exp1b_emcee_trace_k50.png"), dpi=160)
        plt.close()

    if (
        "RWM" in posterior_k200
        and "MALA" in posterior_k200
        and "HMC" in posterior_k200
    ):
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.06])

        axes = np.empty((2, 3), dtype=object)
        axes[0, 0] = fig.add_subplot(gs[0, 0])
        axes[0, 1] = fig.add_subplot(gs[0, 1], sharex=axes[0, 0], sharey=axes[0, 0])
        axes[0, 2] = fig.add_subplot(gs[0, 2], sharex=axes[0, 0], sharey=axes[0, 0])
        axes[1, 0] = fig.add_subplot(gs[1, 0], sharex=axes[0, 0], sharey=axes[0, 0])
        axes[1, 1] = fig.add_subplot(gs[1, 1], sharex=axes[0, 0], sharey=axes[0, 0])
        axes[1, 2] = fig.add_subplot(gs[1, 2], sharex=axes[0, 0], sharey=axes[0, 0])

        cax_top = fig.add_subplot(gs[0, 3])
        cax_bottom = fig.add_subplot(gs[1, 3])
        method_keys = ["RWM", "MALA", "HMC"]
        method_titles = ["emcee RWM", "emcee MALA", "HMC"]

        posterior_mappable = None
        label_mappable = None
        for col, (method_key, title) in enumerate(zip(method_keys, method_titles)):
            posterior_mappable = axes[0, col].scatter(
                X[:, 0],
                X[:, 1],
                c=posterior_k200[method_key],
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                s=10,
                alpha=0.85,
            )
            axes[0, col].set_title(f"{title} posterior mean p(y=1)")
            axes[0, col].grid(alpha=0.25)

            label_mappable = axes[1, col].scatter(
                X[:, 0],
                X[:, 1],
                c=y,
                cmap="coolwarm",
                vmin=0,
                vmax=1,
                s=10,
                alpha=0.85,
            )
            axes[1, col].set_title(f"{title} data labels")
            axes[1, col].grid(alpha=0.25)

        for row in range(2):
            axes[row, 0].set_ylabel("x2")
        for col in range(3):
            axes[1, col].set_xlabel("x1")

        cbar_top = fig.colorbar(posterior_mappable, cax=cax_top)
        cbar_top.set_label("Posterior mean probability")
        cbar_bottom = fig.colorbar(label_mappable, cax=cax_bottom, ticks=[0, 1])
        cbar_bottom.set_label("Observed class label")

        fig.suptitle("Posterior and data points by sampler at k=200", fontsize=12)
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
        plt.savefig(
            os.path.join(data_dir, "exp1b_emcee_posterior_and_data_k200.png"),
            dpi=170,
        )
        plt.close()

    np.save(
        os.path.join(data_dir, "exp1b_emcee_results.npy"),
        {
            "results": results,
            "n_samples": n_samples,
            "n_warmup": n_warmup,
            "k_values": k_values,
        },
        allow_pickle=True,
    )

    print("Saved:")
    print("- data/exp1b_emcee_ess_per_sec_vs_k.png")
    print("- data/exp1b_emcee_tau_vs_k.png")
    print("- data/exp1b_emcee_step_time_vs_k.png")
    print("- data/exp1b_emcee_accept_rate_vs_k.png")
    print("- data/exp1b_emcee_trace_k50.png")
    print("- data/exp1b_emcee_posterior_and_data_k200.png")
    print("- data/exp1b_emcee_results.npy")


if __name__ == "__main__":
    main()
