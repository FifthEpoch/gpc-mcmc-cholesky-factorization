"""
Full HMC GP classification on real embedding datasets.

This is the real-data companion to exp1_predictive.py. It uses the same
stratified subsampling policy as exp1_pygp_approx_gpc_embeddings.py so that
the 2,000-train / 2,000-test run is directly comparable to the pyGPs LA/EP
embedding experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import emcee
import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.linalg import cho_solve, cholesky
from scipy.special import expit


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from my_cholesky.kernels import GaussianKernel_mtx  # noqa: E402
from predictive_metrics import (  # noqa: E402
    evaluate_binary_probabilistic_predictions,
    print_metric_table,
    summarize_predictive_distribution,
)


def compute_tau_emcee(chain: np.ndarray) -> float:
    chain = np.asarray(chain, dtype=float)
    try:
        tau = emcee.autocorr.integrated_time(chain, quiet=True)
    except Exception as err:
        print(f"WARNING: emcee autocorr.integrated_time failed: {err}")
        return float("nan")

    tau = np.asarray(tau, dtype=float).reshape(-1)
    if tau.size == 0:
        return float("nan")
    return float(np.nanmean(tau))


def print_posterior_statistics(
    latent_mean: np.ndarray,
    latent_var: np.ndarray,
    prob_mean: np.ndarray,
    prob_var: np.ndarray,
    title: str,
) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"latent_mean min/mean/max : {np.min(latent_mean):.6f} / {np.mean(latent_mean):.6f} / {np.max(latent_mean):.6f}")
    print(f"latent_var  min/mean/max : {np.min(latent_var):.6f} / {np.mean(latent_var):.6f} / {np.max(latent_var):.6f}")
    print(f"prob_mean   min/mean/max : {np.min(prob_mean):.6f} / {np.mean(prob_mean):.6f} / {np.max(prob_mean):.6f}")
    print(f"prob_var    min/mean/max : {np.min(prob_var):.6f} / {np.mean(prob_var):.6f} / {np.max(prob_var):.6f}")


def _split_name_variants(split: str) -> list[str]:
    variants = [split]
    if split == "val":
        variants.append("valid")
    if split == "valid":
        variants.append("val")
    return variants


def load_embeddings(
    emb_dir: Path,
    dataset: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings from either:
    - flat files: <dataset>_<split>_embeddings.npy
    - HG layout : <split>/embeddings/projected_512.npy
    """
    for split_name in _split_name_variants(split):
        flat_emb = emb_dir / f"{dataset}_{split_name}_embeddings.npy"
        flat_lbl = emb_dir / f"{dataset}_{split_name}_labels.npy"
        if flat_emb.exists() and flat_lbl.exists():
            print(f"[load_embeddings] {dataset}:{split} features <- {flat_emb}")
            print(f"[load_embeddings] {dataset}:{split} labels   <- {flat_lbl}")
            X = np.load(flat_emb).astype(np.float64, copy=False)
            y = np.load(flat_lbl).astype(np.int64, copy=False).squeeze()
            return X, y

    for split_name in _split_name_variants(split):
        split_emb_dir = emb_dir / split_name / "embeddings"
        projected_emb = split_emb_dir / "projected_512.npy"
        raw_emb = split_emb_dir / "embeddings.npy"
        lbl_path = split_emb_dir / "y_embeddings.npy"

        feature_path = projected_emb if projected_emb.exists() else raw_emb
        if feature_path.exists() and lbl_path.exists():
            print(f"[load_embeddings] {dataset}:{split} features <- {feature_path}")
            print(f"[load_embeddings] {dataset}:{split} labels   <- {lbl_path}")
            X = np.load(feature_path).astype(np.float64, copy=False)
            y = np.load(lbl_path).astype(np.int64, copy=False).squeeze()
            return X, y

    raise FileNotFoundError(
        f"Could not find embeddings for dataset={dataset}, split={split} under {emb_dir}."
    )


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_n: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same stratified random subsample policy as pyGPs embedding experiment."""
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


def standardize_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def log_posterior(nu: np.ndarray, factor: np.ndarray, y: np.ndarray) -> float:
    f = factor @ nu
    p = expit(f)
    eps = 1e-10
    log_lik = np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    log_prior = -0.5 * np.dot(nu, nu)
    return float(log_lik + log_prior)


def grad_log_posterior(nu: np.ndarray, factor: np.ndarray, y: np.ndarray) -> np.ndarray:
    f = factor @ nu
    p = expit(f)
    return factor.T @ (y - p) - nu


def run_hmc(
    factor: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    n_warmup: int,
    seed: int,
    step_size: float,
    n_leapfrog: int,
) -> dict[str, Any]:
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
        accept_log_prob = current_h - proposal_h

        if i < 5:
            print(
                f"iter {i}: grad_norm={np.linalg.norm(grad):.4f}, "
                f"proposal_dist={np.linalg.norm(proposal_nu - current_nu):.6f}, "
                f"accept_prob={np.exp(min(0.0, accept_log_prob)):.6f}"
            )

        if np.log(rng.random()) < accept_log_prob:
            nu = proposal_nu
            logp = proposal_logp
            n_accept += 1

        if i >= n_warmup:
            nu_samples[post_idx, :] = nu
            logp_trace[post_idx] = logp
            post_idx += 1

        if (i + 1) % max(1, total_steps // 10) == 0:
            print(
                f"HMC step {i + 1}/{total_steps}: "
                f"accept_rate_so_far={n_accept / (i + 1):.3f}, logp={logp:.3f}"
            )

    return {
        "nu_samples": nu_samples,
        "logp_trace": logp_trace,
        "step_size": float(step_size),
        "n_leapfrog": int(n_leapfrog),
        "accept_rate": float(n_accept / total_steps),
    }


def sample_predictive_probabilities(
    K_train: np.ndarray,
    K_test_train: np.ndarray,
    K_test_test: np.ndarray,
    f_train_samples: np.ndarray,
    n_conditional_draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_test = K_test_train.shape[0]
    n_draws = f_train_samples.shape[1]

    cho = cholesky(K_train, lower=True)
    K_inv_KtestT = cho_solve((cho, True), K_test_train.T)
    S = K_test_test - K_test_train @ K_inv_KtestT
    S = 0.5 * (S + S.T)
    S += 1e-8 * np.eye(n_test)
    cond_chol = cholesky(S, lower=True)

    K_inv_f = cho_solve((cho, True), f_train_samples)
    mean_test = K_test_train @ K_inv_f

    p_samples = []
    latent_samples = []
    for j in range(n_draws):
        m_j = mean_test[:, j]
        for _ in range(n_conditional_draws):
            f_star = m_j + cond_chol @ rng.standard_normal(n_test)
            latent_samples.append(f_star)
            p_samples.append(expit(f_star))

    return np.asarray(p_samples), np.asarray(latent_samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full HMC GP classification on subsampled embedding data."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pcam",
        choices=["pcam", "camelyon17", "embed"],
    )
    parser.add_argument("--embedding-dir", type=str, default="datasets/pcam-hg")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-test-samples", type=int, default=2000)
    parser.add_argument("--no-standardize", action="store_false", dest="standardize")
    parser.add_argument("--kernel-bandwidth", type=float, default=20.0)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-warmup", type=int, default=200)
    parser.add_argument("--hmc-step-size", type=float, default=None)
    parser.add_argument("--n-leapfrog", type=int, default=12)
    parser.add_argument("--n-conditional-draws", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(standardize=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_dir = Path(args.embedding_dir)
    X_train_raw, y_train_raw = load_embeddings(emb_dir, args.dataset, args.train_split)
    X_test_raw, y_test_raw = load_embeddings(emb_dir, args.dataset, args.test_split)

    X_train, y_train, train_idx = stratified_subsample(
        X_train_raw,
        y_train_raw,
        args.max_train_samples,
        args.seed,
    )
    X_test, y_test, test_idx = stratified_subsample(
        X_test_raw,
        y_test_raw,
        args.max_test_samples,
        args.seed + 1,
    )

    if args.standardize:
        X_train, X_test, train_mean, train_std = standardize_features(X_train, X_test)
    else:
        train_mean = np.zeros((1, X_train.shape[1]), dtype=float)
        train_std = np.ones((1, X_train.shape[1]), dtype=float)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    hmc_step_size = (
        float(args.hmc_step_size)
        if args.hmc_step_size is not None
        else float(4.0 / np.sqrt(n_train))
    )

    print("==============================================================================")
    print("Experiment 1: full HMC GP classification on embeddings")
    print("==============================================================================")
    print(f"Dataset              : {args.dataset}")
    print(f"Embedding dir        : {args.embedding_dir}")
    print(f"Train split          : {args.train_split}")
    print(f"Test split           : {args.test_split}")
    print(f"Original train/test  : {len(y_train_raw)} / {len(y_test_raw)}")
    print(f"Used train/test      : {n_train} / {n_test}")
    print(f"Feature dim          : {X_train.shape[1]}")
    print(f"Standardize          : {args.standardize}")
    print(f"Kernel bandwidth     : {args.kernel_bandwidth}")
    print(f"HMC samples/warmup   : {args.n_samples} / {args.n_warmup}")
    print(f"HMC step/leapfrog    : {hmc_step_size:.6f} / {args.n_leapfrog}")
    print(f"Conditional draws    : {args.n_conditional_draws}")
    print(f"Seed                 : {args.seed}")
    print("==============================================================================")

    t_kernel_start = time.perf_counter()
    K_train = (
        GaussianKernel_mtx(X_train, X_train, bandwidth=args.kernel_bandwidth)
        + 1e-6 * np.eye(n_train)
    )
    K_test_train = GaussianKernel_mtx(X_test, X_train, bandwidth=args.kernel_bandwidth)
    K_test_test = GaussianKernel_mtx(X_test, X_test, bandwidth=args.kernel_bandwidth)
    kernel_time = time.perf_counter() - t_kernel_start
    print(f"Kernel construction time: {kernel_time:.3f}s")

    t_chol_start = time.perf_counter()
    L_dense = cholesky(K_train, lower=True)
    chol_time = time.perf_counter() - t_chol_start
    print(f"Training kernel Cholesky time: {chol_time:.3f}s")

    t_train_start = time.perf_counter()
    hmc_stats = run_hmc(
        L_dense,
        y_train,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        seed=args.seed + 81,
        step_size=hmc_step_size,
        n_leapfrog=args.n_leapfrog,
    )
    train_time = time.perf_counter() - t_train_start

    print(f"Training HMC time: {train_time:.3f}s")
    print(f"HMC acceptance rate: {hmc_stats['accept_rate']:.3f}")
    print(
        "logp_trace min/mean/max: "
        f"{np.min(hmc_stats['logp_trace']):.3f} / "
        f"{np.mean(hmc_stats['logp_trace']):.3f} / "
        f"{np.max(hmc_stats['logp_trace']):.3f}"
    )

    tau_nu = compute_tau_emcee(hmc_stats["nu_samples"])
    tau_logp = compute_tau_emcee(hmc_stats["logp_trace"])
    print(f"HMC tau (nu mean): {tau_nu:.2f}")
    print(f"HMC tau (logp): {tau_logp:.2f}")

    f_train_samples = L_dense @ hmc_stats["nu_samples"].T

    t_pred_start = time.perf_counter()
    p_test_samples, latent_test_samples = sample_predictive_probabilities(
        K_train,
        K_test_train,
        K_test_test,
        f_train_samples,
        n_conditional_draws=args.n_conditional_draws,
        seed=args.seed + 957,
    )
    prediction_time = time.perf_counter() - t_pred_start
    print(f"Prediction time: {prediction_time:.3f}s")

    pred_summary = summarize_predictive_distribution(
        p_samples=p_test_samples,
        latent_samples=latent_test_samples,
    )
    predictive_prob = pred_summary["prob_mean"]
    predictive_var = pred_summary["prob_variance"]
    predictive_latent_mean = pred_summary["latent_mean"]
    predictive_latent_var = pred_summary["latent_variance"]

    print_posterior_statistics(
        latent_mean=predictive_latent_mean,
        latent_var=predictive_latent_var,
        prob_mean=predictive_prob,
        prob_var=predictive_var,
        title="Full HMC GP embedding test posterior statistics",
    )

    test_metrics = evaluate_binary_probabilistic_predictions(
        y_true=y_test,
        p_pred=predictive_prob,
        threshold=0.5,
        n_bins=15,
    )
    test_metrics["kernel_time_sec"] = round(float(kernel_time), 3)
    test_metrics["cholesky_time_sec"] = round(float(chol_time), 3)
    test_metrics["train_time_sec"] = round(float(train_time), 3)
    test_metrics["prediction_time_sec"] = round(float(prediction_time), 3)
    test_metrics["n_train"] = int(n_train)
    test_metrics["n_test"] = int(n_test)
    test_metrics["feature_dim"] = int(X_train.shape[1])
    test_metrics["kernel_bandwidth"] = float(args.kernel_bandwidth)
    test_metrics["n_samples"] = int(args.n_samples)
    test_metrics["n_warmup"] = int(args.n_warmup)
    test_metrics["hmc_step_size"] = float(hmc_step_size)
    test_metrics["n_leapfrog"] = int(args.n_leapfrog)
    test_metrics["n_conditional_draws"] = int(args.n_conditional_draws)
    test_metrics["accept_rate"] = float(hmc_stats["accept_rate"])
    test_metrics["tau_nu"] = float(tau_nu)
    test_metrics["tau_logp"] = float(tau_logp)
    test_metrics["prob_min"] = float(np.min(predictive_prob))
    test_metrics["prob_mean"] = float(np.mean(predictive_prob))
    test_metrics["prob_max"] = float(np.max(predictive_prob))

    print_metric_table(test_metrics, title="Full HMC GP embedding test metrics")

    prefix = f"exp1_full_hmc_{args.dataset}_{args.train_split}_to_{args.test_split}"
    metrics_path = out_dir / f"{prefix}_metrics.json"
    results_path = out_dir / f"{prefix}_results.npz"

    with open(metrics_path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "embedding_dir": args.embedding_dir,
                "train_split": args.train_split,
                "test_split": args.test_split,
                "max_train_samples": args.max_train_samples,
                "max_test_samples": args.max_test_samples,
                "standardize": args.standardize,
                "kernel_bandwidth": args.kernel_bandwidth,
                "n_samples": args.n_samples,
                "n_warmup": args.n_warmup,
                "hmc_step_size": hmc_step_size,
                "n_leapfrog": args.n_leapfrog,
                "n_conditional_draws": args.n_conditional_draws,
                "seed": args.seed,
                "metrics": test_metrics,
            },
            f,
            indent=2,
        )

    np.savez(
        results_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_indices=train_idx,
        test_indices=test_idx,
        train_mean=train_mean,
        train_std=train_std,
        predictive_prob_test=predictive_prob,
        predictive_prob_var_test=predictive_var,
        predictive_latent_mean_test=predictive_latent_mean,
        predictive_latent_var_test=predictive_latent_var,
        prob_q05=pred_summary["prob_q05"],
        prob_q50=pred_summary["prob_q50"],
        prob_q95=pred_summary["prob_q95"],
        latent_q05=pred_summary["latent_q05"],
        latent_q50=pred_summary["latent_q50"],
        latent_q95=pred_summary["latent_q95"],
        logp_trace=hmc_stats["logp_trace"],
        accept_rate=hmc_stats["accept_rate"],
        tau_nu=tau_nu,
        tau_logp=tau_logp,
        kernel_bandwidth=args.kernel_bandwidth,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        hmc_step_size=hmc_step_size,
        n_leapfrog=args.n_leapfrog,
        n_conditional_draws=args.n_conditional_draws,
    )

    print("Saved:")
    print(f"- {metrics_path}")
    print(f"- {results_path}")


if __name__ == "__main__":
    main()
