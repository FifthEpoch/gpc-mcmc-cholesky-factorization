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

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_solve, cholesky
from scipy.special import expit


# Allow direct script execution without package install.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from my_cholesky.kernels import GaussianKernel_mtx


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

        if np.log(rng.random()) < accept_log_prob:
            nu = proposal_nu
            logp = proposal_logp
            n_accept += 1

        if i >= n_warmup:
            nu_samples[post_idx, :] = nu
            post_idx += 1

    accept_rate = n_accept / total_steps

    return {
        "nu_samples": nu_samples,
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

    for j in range(n_draws):
        m_j = mean_test[:, j]

        for _ in range(n_conditional_draws):
            z = rng.standard_normal(n_test)

            # Correct sampling:
            # f_* = m_j + L_S z,
            # so Cov(f_*) = L_S L_S^T = S.
            f_star = m_j + cond_chol @ z

            p_samples.append(expit(f_star))

    return np.asarray(p_samples)


def main():
    print("Starting HMC predictive computation...")

    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Training data
    X_train, y_train = make_fake_blobs(seed=42, n_per_class=100)
    n_train = X_train.shape[0]

    # Test grid
    x1 = np.linspace(-3, 3, 20)
    x2 = np.linspace(-3, 3, 20)
    X1, X2 = np.meshgrid(x1, x2)
    X_test = np.column_stack([X1.ravel(), X2.ravel()])
    n_test = X_test.shape[0]

    # Kernel matrices
    bandwidth = 1.0

    K_train = (
        GaussianKernel_mtx(X_train, X_train, bandwidth=bandwidth)
        + 1e-6 * np.eye(n_train)
    )

    K_test_train = GaussianKernel_mtx(
        X_test,
        X_train,
        bandwidth=bandwidth,
    )

    K_test_test = GaussianKernel_mtx(
        X_test,
        X_test,
        bandwidth=bandwidth,
    )

    # Reparameterization f = L nu, nu ~ N(0, I).
    L_dense = np.linalg.cholesky(K_train)

    # HMC parameters
    n_samples = 200
    n_warmup = 200
    hmc_step = 0.08 / np.sqrt(n_train)
    n_leapfrog = 12

    hmc_stats = run_hmc(
        L_dense,
        y_train,
        n_samples=n_samples,
        n_warmup=n_warmup,
        seed=123,
        step_size=hmc_step,
        n_leapfrog=n_leapfrog,
    )

    print(f"HMC acceptance rate: {hmc_stats['accept_rate']:.3f}")

    # Convert nu samples to latent training samples f samples.
    nu_samples = hmc_stats["nu_samples"]
    f_train_samples = L_dense @ nu_samples.T

    # Sample predictive probabilities
    p_test_samples = sample_predictive_probabilities(
        K_train,
        K_test_train,
        K_test_test,
        f_train_samples,
        n_conditional_draws=10,
        seed=999,
    )

    predictive_prob = np.mean(p_test_samples, axis=0)
    predictive_std = np.std(p_test_samples, axis=0)

    prob_grid = predictive_prob.reshape(X1.shape)
    std_grid = predictive_std.reshape(X1.shape)

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
        X_test=X_test,
        predictive_prob=predictive_prob,
        predictive_std=predictive_std,
        X_train=X_train,
        y_train=y_train,
        accept_rate=hmc_stats["accept_rate"],
    )

    print("HMC predictive computation completed.")
    print(f"Saved predictive plot to: {plot_path}")
    print(f"Saved predictive results to: {results_path}")


if __name__ == "__main__":
    main()
