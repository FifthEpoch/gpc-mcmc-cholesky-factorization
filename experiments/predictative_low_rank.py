"""
Low-rank GP classification predictive sampling using:
1) RP-Cholesky factor F for K_XX,
2) HMC in the non-centered training coordinates f = F nu,
3) Woodbury solves with K_XX approx = F F^T + nugget I,
4) RP-Cholesky factor G for the predictive covariance S,
5) non-centered predictive sampling f_* = m + G xi.

This avoids dense Cholesky of K_XX and uses a nugget-stabilized
low-rank approximation on the training side.
"""

import os
import sys
import emcee
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.special import expit

# Allow direct script execution without package install.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from my_cholesky.arpcholesky import arpcholesky
from my_cholesky.kernels import GaussianKernel_mtx


def make_fake_blobs(seed=42, n_per_class=100):
    rng = np.random.default_rng(seed)
    cov = 0.5 * np.eye(2)
    x0 = rng.multivariate_normal(mean=[-1.0, 0.0], cov=cov, size=n_per_class)
    x1 = rng.multivariate_normal(mean=[1.0, 0.0], cov=cov, size=n_per_class)
    X = np.vstack([x0, x1])
    y = np.concatenate(
        [np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)]
    )
    return X, y


def low_rank_spd_solve(F, rhs, nugget):
    """
    Solve (F F^T + nugget I) x = rhs using Woodbury.

    F:   (n, r)
    rhs: (n,) or (n, m)

    Returns x with the same shape as rhs.
    """
    rhs_arr = np.asarray(rhs)
    is_vector = (rhs_arr.ndim == 1)
    rhs_2d = rhs_arr[:, None] if is_vector else rhs_arr

    nugget_inv = 1.0 / nugget
    middle = np.eye(F.shape[1]) + nugget_inv * (F.T @ F)

    tmp = F.T @ rhs_2d
    correction = solve(middle, tmp, assume_a="pos")
    sol = nugget_inv * rhs_2d - (nugget_inv ** 2) * F @ correction

    return sol[:, 0] if is_vector else sol


def log_posterior(nu, factor, y):
    """
    Approximate low-rank posterior with f = factor @ nu and nu ~ N(0, I).
    """
    f = factor @ nu
    p = expit(f)
    eps = 1e-10
    log_lik = np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    log_prior = -0.5 * np.dot(nu, nu)
    return float(log_lik + log_prior)


def grad_log_posterior(nu, factor, y):
    f = factor @ nu
    p = expit(f)
    return factor.T @ (y - p) - nu


def run_hmc(factor, y, n_samples, n_warmup, seed, step_size, n_leapfrog):
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

        if np.log(rng.random()) < (current_h - proposal_h):
            nu = proposal_nu
            logp = proposal_logp
            n_accept += 1

        if i >= n_warmup:
            nu_samples[post_idx] = nu
            logp_trace[post_idx] = logp
            post_idx += 1

    return {
        "nu_samples": nu_samples,
        "logp_trace": logp_trace,
        "accept_rate": n_accept / total_steps,
    }


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


def make_psd_matrix(A, jitter=1e-8):
    """Symmetrize and enforce PSD by adding minimal jitter."""
    A = 0.5 * (A + A.T)
    min_eig = np.min(np.linalg.eigvalsh(A))
    if min_eig < 0:
        A = A + (-min_eig + jitter) * np.eye(A.shape[0])
    else:
        A = A + jitter * np.eye(A.shape[0])
    return A

def build_low_rank_predictive_covariance(
    F,
    K_test_train,
    K_test_test,
    nugget,
    test_jitter=1e-8,
):
    """
    Build the approximate predictive covariance

        S = K_** - K_*X (F F^T + nugget I)^{-1} K_X*

    and symmetrize it.
    """
    Kinv_Kxt = low_rank_spd_solve(F, K_test_train.T, nugget)
    S = K_test_test - K_test_train @ Kinv_Kxt
    S = make_psd_matrix(S, jitter=test_jitter)
    return S


def sample_predictive_probabilities_lowrank_nugget(
    F,
    K_test_train,
    K_test_test,
    f_train_samples,
    nugget=1e-4,
    test_rank=100,
    test_jitter=1e-8,
    n_conditional_draws=10,
    seed=0,
):
    """
    Predictive sampler with training-side nugget:

      K_XX approx = F F^T + nugget I

    For each posterior sample f^(s), compute
      m^(s) = K_*X (F F^T + nugget I)^{-1} f^(s)

    Then approximate the predictive covariance
      S = K_** - K_*X (F F^T + nugget I)^{-1} K_X*
    with a low-rank factor
      S ≈ G G^T

    and sample
      f_*^(s,r) = m^(s) + G xi.
    """
    rng = np.random.default_rng(seed)

    n_test = K_test_train.shape[0]
    n_draws = f_train_samples.shape[1]

    # Mean terms for all posterior draws
    Kinv_f = low_rank_spd_solve(F, f_train_samples, nugget)
    mean_test = K_test_train @ Kinv_f  # (n_test, n_draws)

    # Predictive covariance
    S = build_low_rank_predictive_covariance(
        F=F,
        K_test_train=K_test_train,
        K_test_test=K_test_test,
        nugget=nugget,
    )

    # Small jitter for numerical safety before RP-Cholesky
    S = S + test_jitter * np.eye(n_test)

    # Low-rank factor of S
    rank_s = min(test_rank, n_test)
    low_rank_S = arpcholesky(S, k=rank_s, b=10, seed=seed)
    G = low_rank_S.get_left_factor()

    p_samples = []
    latent_samples = []

    for j in range(n_draws):
        m_j = mean_test[:, j]

        for _ in range(n_conditional_draws):
            xi = rng.standard_normal(G.shape[1])
            f_star = m_j + G @ xi
            latent_samples.append(f_star)
            p_samples.append(expit(f_star))

    return {
        "p_samples": np.asarray(p_samples),
        "latent_samples": np.asarray(latent_samples),
        "predictive_cov_approx_factor": G,
    }


def main():
    print("Starting low-rank HMC predictive computation with nugget...")

    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    # -------------------------
    # Data
    # -------------------------
    X_train, y_train = make_fake_blobs(seed=42, n_per_class=100)
    n_train = X_train.shape[0]

    x1 = np.linspace(-3, 3, 20)
    x2 = np.linspace(-3, 3, 20)
    X1, X2 = np.meshgrid(x1, x2)
    X_test = np.column_stack([X1.ravel(), X2.ravel()])
    n_test = X_test.shape[0]

    # -------------------------
    # Kernel matrices
    # -------------------------
    bandwidth = 1.0
    K_train = GaussianKernel_mtx(X_train, X_train, bandwidth=bandwidth)
    K_test_train = GaussianKernel_mtx(X_test, X_train, bandwidth=bandwidth)
    K_test_test = GaussianKernel_mtx(X_test, X_test, bandwidth=bandwidth)

    # -------------------------
    # Low-rank factor for K_XX
    # -------------------------
    train_rank = min(80, n_train)
    nugget = 1e-4

    low_rank_train = arpcholesky(K_train, k=train_rank, b=10, seed=42)
    F = low_rank_train.get_left_factor()   # shape (n_train, train_rank)

    # -------------------------
    # HMC in low-rank coordinates
    # -------------------------
    n_samples = 200
    n_warmup = 200
    hmc_step = 0.08 / np.sqrt(train_rank)
    n_leapfrog = 12

    hmc_stats = run_hmc(
        factor=F,
        y=y_train,
        n_samples=n_samples,
        n_warmup=n_warmup,
        seed=123,
        step_size=hmc_step,
        n_leapfrog=n_leapfrog,
    )

    print(f"HMC acceptance rate: {hmc_stats['accept_rate']:.3f}")

    nu_samples = hmc_stats["nu_samples"]               # (n_samples, train_rank)
    f_train_samples = F @ nu_samples.T                 # (n_train, n_samples)

    # -------------------------
    # Predictive sampling
    # -------------------------
    pred = sample_predictive_probabilities_lowrank_nugget(
        F=F,
        K_test_train=K_test_train,
        K_test_test=K_test_test,
        f_train_samples=f_train_samples,
        nugget=nugget,
        test_rank=min(120, n_test),
        test_jitter=1e-8,
        n_conditional_draws=10,
        seed=999,
    )

    p_test_samples = pred["p_samples"]
    latent_test_samples = pred["latent_samples"]

    predictive_prob = np.mean(p_test_samples, axis=0)
    predictive_std = np.std(p_test_samples, axis=0)

    predictive_latent_mean = np.mean(latent_test_samples, axis=0)
    predictive_latent_std = np.std(latent_test_samples, axis=0)

    prob_grid = predictive_prob.reshape(X1.shape)
    std_grid = predictive_std.reshape(X1.shape)

    # -------------------------
    # Plot
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axes[0].contourf(X1, X2, prob_grid, levels=20, cmap="RdYlBu_r")
    axes[0].scatter(
        X_train[y_train == 0, 0], X_train[y_train == 0, 1],
        c="blue", label="Class 0", alpha=0.6
    )
    axes[0].scatter(
        X_train[y_train == 1, 0], X_train[y_train == 1, 1],
        c="red", label="Class 1", alpha=0.6
    )
    axes[0].set_title("Predictive Probability (Low-Rank HMC GP)")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].legend()
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].contourf(X1, X2, std_grid, levels=20, cmap="viridis")
    axes[1].scatter(
        X_train[y_train == 0, 0], X_train[y_train == 0, 1],
        c="blue", alpha=0.6
    )
    axes[1].scatter(
        X_train[y_train == 1, 0], X_train[y_train == 1, 1],
        c="red", alpha=0.6
    )
    axes[1].set_title("Predictive Std (Low-Rank HMC GP)")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()

    plot_path = os.path.join(data_dir, "exp1_predictive_hmc_lowrank_nugget.png")
    results_path = os.path.join(data_dir, "exp1_predictive_hmc_lowrank_nugget_results.npz")

    plt.savefig(plot_path, dpi=150)

    np.savez(
        results_path,
        X_test=X_test,
        predictive_prob=predictive_prob,
        predictive_std=predictive_std,
        predictive_latent_mean=predictive_latent_mean,
        predictive_latent_std=predictive_latent_std,
        X_train=X_train,
        y_train=y_train,
        accept_rate=hmc_stats["accept_rate"],
        train_rank=train_rank,
        test_rank=min(120, n_test),
        nugget=nugget,
    )

    print("Low-rank HMC predictive computation completed.")
    print(f"Saved predictive plot to: {plot_path}")
    print(f"Saved predictive results to: {results_path}")


if __name__ == "__main__":
    main()