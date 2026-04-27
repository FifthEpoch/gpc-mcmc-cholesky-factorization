"""
Experiment 1: Predictive latent distribution and predictive probability
for GP classification using HMC.

Corrected version.

Structure:
1) sample posterior over latent training values via HMC,
2) compute conditional Gaussian predictive distribution for test latents,
3) sample from that conditional distribution,
4) estimate predictive probability by averaging sigmoid outputs.
"""

import json
import os
import sys
import time

import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_solve, cholesky
from scipy.special import expit


# Allow direct script execution without package install.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from my_cholesky.kernels import GaussianKernel_mtx
from my_cholesky.result_logging import append_result_row
from predictive_metrics import (
    evaluate_binary_probabilistic_predictions,
    print_metric_table,
    print_posterior_statistics,
    summarize_predictive_distribution,
)


def compute_tau_emcee(chain):
    """Estimate integrated autocorrelation time with emcee's implementation."""
    chain = np.asarray(chain, dtype=float)
    print(f"  compute_tau_emcee received chain shape: {chain.shape}")
    
    try:
        tau = emcee.autocorr.integrated_time(chain, quiet=True)
        print(f"    tau result (raw): {tau}")
    except Exception as err:
        print(f"WARNING: emcee autocorr.integrated_time failed: {err}")
        return float("nan")

    tau = np.asarray(tau, dtype=float).reshape(-1)
    if tau.size == 0:
        return float("nan")
    return float(np.nanmean(tau))


def make_fake_blobs(seed=42, n_per_class=100):
    """Generate two-blob binary data in R^2."""
    rng = np.random.default_rng(seed)

    cov = 0.5 * np.eye(2)

    x0 = rng.multivariate_normal(
        mean=[-1.0, 0.0],
        cov=cov,
        size=n_per_class,
    )

    x1 = rng.multivariate_normal(
        mean=[1.0, 0.0],
        cov=cov,
        size=n_per_class,
    )

    X = np.vstack([x0, x1])
    y = np.concatenate(
        [
            np.zeros(n_per_class, dtype=int),
            np.ones(n_per_class, dtype=int),
        ]
    )

    return X, y


def log_posterior(nu, factor, y):
    """
    Log posterior for Bernoulli likelihood and standard normal prior.

    We use the reparameterization

        f = L nu,

    where L L^T = K and nu ~ N(0, I).
    """
    f = factor @ nu
    p = expit(f)

    eps = 1e-10

    log_lik = np.sum(
        y * np.log(p + eps)
        + (1 - y) * np.log(1 - p + eps)
    )

    log_prior = -0.5 * np.dot(nu, nu)

    return float(log_lik + log_prior)


def grad_log_posterior(nu, factor, y):
    """
    Gradient of log posterior with respect to nu.

    Since f = L nu,

        grad_nu log p(y | f) = L^T (y - sigmoid(f)).

    The prior contributes -nu.
    """
    f = factor @ nu
    p = expit(f)

    return factor.T @ (y - p) - nu


def run_hmc(
    factor,
    y,
    n_samples,
    n_warmup,
    seed,
    step_size,
    n_leapfrog,
):
    """Simple Euclidean HMC with fixed mass matrix."""
    rng = np.random.default_rng(seed)

    dim = factor.shape[1]
    total_steps = n_warmup + n_samples

    nu = np.zeros(dim, dtype=float)
    logp = log_posterior(nu, factor, y)

    nu_samples = np.zeros((n_samples, dim), dtype=float)
    logp_trace = np.zeros((n_samples,), dtype=float)
    post_idx = 0

    n_accept = 0

    for i in range(total_steps):
        current_nu = nu.copy()
        current_logp = logp

        momentum = rng.standard_normal(dim)
        current_momentum = momentum.copy()

        # First half momentum step
        grad = grad_log_posterior(current_nu, factor, y)
        proposal_nu = current_nu.copy()
        proposal_p = momentum + 0.5 * step_size * grad

        # Leapfrog steps
        for leapfrog_idx in range(n_leapfrog):
            proposal_nu = proposal_nu + step_size * proposal_p
            grad = grad_log_posterior(proposal_nu, factor, y)

            if leapfrog_idx != n_leapfrog - 1:
                proposal_p = proposal_p + step_size * grad

        # Final half momentum step
        proposal_p = proposal_p + 0.5 * step_size * grad

        # Momentum flip for reversibility
        proposal_p = -proposal_p

        proposal_logp = log_posterior(proposal_nu, factor, y)

        current_h = -current_logp + 0.5 * np.dot(current_momentum, current_momentum)
        proposal_h = -proposal_logp + 0.5 * np.dot(proposal_p, proposal_p)

        accept_log_prob = current_h - proposal_h

        # Debug first few iterations
        if i < 5:
            grad_norm = np.linalg.norm(grad)
            proposal_distance = np.linalg.norm(proposal_nu - current_nu)
            print(f"iter {i}: grad_norm={grad_norm:.4f}, proposal_dist={proposal_distance:.6f}, "
                  f"accept_prob={np.exp(min(0, accept_log_prob)):.6f}")

        if np.log(rng.random()) < accept_log_prob:
            nu = proposal_nu
            logp = proposal_logp
            n_accept += 1

        if i >= n_warmup:
            nu_samples[post_idx, :] = nu
            logp_trace[post_idx] = logp
            post_idx += 1

    accept_rate = n_accept / total_steps

    return {
        "nu_samples": nu_samples,
        "logp_trace": logp_trace,
        "step_size": float(step_size),
        "n_leapfrog": int(n_leapfrog),
        "accept_rate": float(accept_rate),
    }


def sample_predictive_probabilities(
    K_train,
    K_test_train,
    K_test_test,
    f_train_samples,
    n_conditional_draws=10,
    seed=0,
):
    """
    Sample predictive probabilities by drawing conditional test latents.

    We approximate

        p(y_* = 1 | X, y, X_*)
        = E[ sigmoid(f_*) | X, y, X_* ]

    using Monte Carlo.

    For each posterior sample f^(s), we sample

        f_*^(s,r) ~ p(f_* | X, X_*, f^(s)).

    Then we average sigmoid(f_*^(s,r)).
    """
    rng = np.random.default_rng(seed)

    n_test = K_test_train.shape[0]
    n_draws = f_train_samples.shape[1]

    # Cholesky factor of K_train.
    # K_train = L L^T.
    cho = cholesky(K_train, lower=True)

    # Compute K_train^{-1} K_{X,*}.
    K_inv_KtestT = cho_solve((cho, True), K_test_train.T)

    # Conditional covariance:
    #
    # S = K_{**} - K_{*X} K_{XX}^{-1} K_{X*}
    #
    S = K_test_test - K_test_train @ K_inv_KtestT

    # Symmetrize to remove tiny numerical asymmetry.
    S = 0.5 * (S + S.T)

    # Add jitter for numerical stability.
    S += 1e-8 * np.eye(n_test)

    # Correct Cholesky factor.
    # Since lower=True, we get S = L_S L_S^T.
    cond_chol = cholesky(S, lower=True)

    # Conditional means for every posterior sample:
    #
    # mean_test[:, j] = K_{*X} K_{XX}^{-1} f_train_samples[:, j]
    #
    K_inv_f = cho_solve((cho, True), f_train_samples)
    mean_test = K_test_train @ K_inv_f

    p_samples = []
    latent_samples = []

    for j in range(n_draws):
        m_j = mean_test[:, j]

        for _ in range(n_conditional_draws):
            z = rng.standard_normal(n_test)

            # Correct sampling:
            # f_* = m_j + L_S z,
            # so Cov(f_*) = L_S L_S^T = S.
            f_star = m_j + cond_chol @ z

            latent_samples.append(f_star)
            p_samples.append(expit(f_star))

    return np.asarray(p_samples), np.asarray(latent_samples)


def main():
    print("Starting HMC predictive computation...")

    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Training data
    X_train, y_train = make_fake_blobs(seed=42, n_per_class=1000)
    n_train = X_train.shape[0]

    # Labeled test set for evaluation
    X_test_labeled, y_test = make_fake_blobs(seed=123, n_per_class=500)
    n_test_labeled = X_test_labeled.shape[0]

    # Plotting grid for posteriors
    x1 = np.linspace(-3, 3, 20)
    x2 = np.linspace(-3, 3, 20)
    X1, X2 = np.meshgrid(x1, x2)
    X_test_plot = np.column_stack([X1.ravel(), X2.ravel()])
    n_test_plot = X_test_plot.shape[0]

    # Kernel matrices
    bandwidth = 1.0

    K_train = (
        GaussianKernel_mtx(X_train, X_train, bandwidth=bandwidth)
        + 1e-6 * np.eye(n_train)
    )

    K_test_train_plot = GaussianKernel_mtx(
        X_test_plot,
        X_train,
        bandwidth=bandwidth,
    )
    K_test_test_plot = GaussianKernel_mtx(
        X_test_plot,
        X_test_plot,
        bandwidth=bandwidth,
    )

    K_test_train_eval = GaussianKernel_mtx(
        X_test_labeled,
        X_train,
        bandwidth=bandwidth,
    )
    K_test_test_eval = GaussianKernel_mtx(
        X_test_labeled,
        X_test_labeled,
        bandwidth=bandwidth,
    )

    # Reparameterization f = L nu, nu ~ N(0, I).
    L_dense = np.linalg.cholesky(K_train)

    # HMC parameters
    n_samples = 5000
    n_warmup = 500
    hmc_step = 4 / np.sqrt(n_train)
    n_leapfrog = 12

    print(f"HMC setup:")
    print(f"  n_train: {n_train}")
    print(f"  step_size: {hmc_step:.6f}")
    print(f"  n_leapfrog: {n_leapfrog}")
    print(f"  n_samples: {n_samples}")
    print(f"  n_warmup: {n_warmup}")

    t_train_start = time.perf_counter()
    hmc_stats = run_hmc(
        L_dense,
        y_train,
        n_samples=n_samples,
        n_warmup=n_warmup,
        seed=123,
        step_size=hmc_step,
        n_leapfrog=n_leapfrog,
    )
    t_train_elapsed = time.perf_counter() - t_train_start

    print(f"\nTraining (HMC) time: {t_train_elapsed:.3f}s for {n_samples} samples ({t_train_elapsed / n_samples * 1000:.2f}ms per sample)")
    print(f"HMC acceptance rate: {hmc_stats['accept_rate']:.3f}")

    # Diagnostics on chain
    print(f"\nChain sizes:")
    print(f"  nu_samples shape: {hmc_stats['nu_samples'].shape}")
    print(f"  logp_trace shape: {hmc_stats['logp_trace'].shape}")
    print(f"  logp_trace min/max: {np.min(hmc_stats['logp_trace']):.4f} / {np.max(hmc_stats['logp_trace']):.4f}")
    print(f"  logp_trace std: {np.std(hmc_stats['logp_trace']):.4f}")

    # Check if chain is moving
    nu_samples = hmc_stats['nu_samples']
    mean_step = np.mean(np.linalg.norm(np.diff(nu_samples, axis=0), axis=1))
    std_nu = np.std(nu_samples, axis=0).mean()
    print(f"\nChain movement diagnostics:")
    print(f"  Mean step size between consecutive samples: {mean_step:.6f}")
    print(f"  Mean std of nu chain: {std_nu:.6f}")
    print(f"  nu_samples min/max: {np.min(nu_samples):.4f} / {np.max(nu_samples):.4f}")
    print(f"  nu_samples std (overall): {np.std(nu_samples):.4f}")

    tau_nu = compute_tau_emcee(hmc_stats["nu_samples"])
    tau_logp = compute_tau_emcee(hmc_stats["logp_trace"])
    print(f"HMC tau (nu mean): {tau_nu:.2f}")
    print(f"HMC tau (logp): {tau_logp:.2f}")

    # Convert nu samples to latent training samples f samples.
    nu_samples = hmc_stats["nu_samples"]
    f_train_samples = L_dense @ nu_samples.T

    # Sample predictive probabilities for the labeled test set with timing.
    t_pred_start = time.perf_counter()
    p_test_samples, latent_test_samples = sample_predictive_probabilities(
        K_train,
        K_test_train_eval,
        K_test_test_eval,
        f_train_samples,
        n_conditional_draws=10,
        seed=999,
    )
    t_pred_elapsed = time.perf_counter() - t_pred_start

    pred_summary = summarize_predictive_distribution(
        p_samples=p_test_samples,
        latent_samples=latent_test_samples,
    )

    predictive_prob = pred_summary["prob_mean"]
    predictive_var = pred_summary["prob_variance"]
    predictive_std = pred_summary["prob_std"]

    predictive_latent_mean = pred_summary["latent_mean"]
    predictive_latent_var = pred_summary["latent_variance"]
    predictive_latent_std = pred_summary["latent_std"]

    prob_q05 = pred_summary["prob_q05"]
    prob_q50 = pred_summary["prob_q50"]
    prob_q95 = pred_summary["prob_q95"]

    latent_q05 = pred_summary["latent_q05"]
    latent_q50 = pred_summary["latent_q50"]
    latent_q95 = pred_summary["latent_q95"]

    print(f"\nPrediction timing for test set ({n_test_labeled} points)")
    print("-" * 50)
    print(f"Prediction time: {t_pred_elapsed:.3f}s")

    print_posterior_statistics(
        latent_mean=predictive_latent_mean,
        latent_var=predictive_latent_var,
        prob_mean=predictive_prob,
        prob_var=predictive_var,
        title="HMC GP test set posterior statistics",
    )

    test_metrics = evaluate_binary_probabilistic_predictions(
        y_true=y_test,
        p_pred=predictive_prob,
        threshold=0.5,
        n_bins=15,
        p_samples=p_test_samples,
    )
    print_metric_table(test_metrics, title="HMC GP test metrics")

    # Sample predictive probabilities for plotting on the grid.
    p_plot_samples, _ = sample_predictive_probabilities(
        K_train,
        K_test_train_plot,
        K_test_test_plot,
        f_train_samples,
        n_conditional_draws=10,
        seed=999,
    )
    predictive_prob_plot = np.mean(p_plot_samples, axis=0)
    predictive_std_plot = np.std(p_plot_samples, axis=0)
    prob_grid = predictive_prob_plot.reshape(X1.shape)
    std_grid = predictive_std_plot.reshape(X1.shape)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axes[0].contourf(
        X1,
        X2,
        prob_grid,
        levels=20,
        cmap="RdYlBu_r",
    )

    axes[0].scatter(
        X_train[y_train == 0, 0],
        X_train[y_train == 0, 1],
        c="blue",
        label="Class 0",
        alpha=0.6,
    )

    axes[0].scatter(
        X_train[y_train == 1, 0],
        X_train[y_train == 1, 1],
        c="red",
        label="Class 1",
        alpha=0.6,
    )

    axes[0].set_title("Predictive Probability (HMC GP)")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].legend()
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].contourf(
        X1,
        X2,
        std_grid,
        levels=20,
        cmap="viridis",
    )

    axes[1].scatter(
        X_train[y_train == 0, 0],
        X_train[y_train == 0, 1],
        c="blue",
        alpha=0.6,
    )

    axes[1].scatter(
        X_train[y_train == 1, 0],
        X_train[y_train == 1, 1],
        c="red",
        alpha=0.6,
    )

    axes[1].set_title("Predictive Std (HMC GP)")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()

    plot_path = os.path.join(data_dir, "exp1_predictive_hmc.png")
    results_path = os.path.join(data_dir, "exp1_predictive_hmc_results.npz")

    plt.savefig(plot_path, dpi=150)

    np.savez(
        results_path,
        X_test_plot=X_test_plot,
        X_test_labeled=X_test_labeled,
        y_test=y_test,
        predictive_prob_plot=predictive_prob_plot,
        predictive_std_plot=predictive_std_plot,
        predictive_prob_test=predictive_prob,
        predictive_std_test=predictive_std,
        predictive_latent_mean_test=predictive_latent_mean,
        predictive_latent_var_test=predictive_latent_var,
        prob_q05=prob_q05,
        prob_q50=prob_q50,
        prob_q95=prob_q95,
        latent_q05=latent_q05,
        latent_q50=latent_q50,
        latent_q95=latent_q95,
        X_train=X_train,
        y_train=y_train,
        accept_rate=hmc_stats["accept_rate"],
        tau_nu=tau_nu,
        tau_logp=tau_logp,
        **test_metrics,
    )

    print("HMC predictive computation completed.")
    print(f"Saved predictive plot to: {plot_path}")
    print(f"Saved predictive results to: {results_path}")
    csv_path = append_result_row(
        {
            "experiment": "exp1",
            "script_path": "experiments/exp1_predictive.py",
            "artifacts": json.dumps([plot_path, results_path]),
            "dataset": "synthetic_two_blob",
            "data_seed": 42,
            "sampler": "hmc",
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "accept_rate": hmc_stats["accept_rate"],
            "tau": tau_logp,
            **test_metrics,
        }
    )
    print(f"Appended CSV metrics to {csv_path}")


if __name__ == "__main__":
    main()
