#!/usr/bin/env python3
"""Benchmark float32 vs float64 HMC arithmetic on the real RPCholesky factor."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.special import expit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark float32 vs float64 matvec and log-posterior arithmetic."
    )
    parser.add_argument("--factor", type=str, default="data/rpchol_full/factor_k200.npy")
    parser.add_argument("--labels", type=str, default="data/rpchol_full/labels.npy")
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--n-precision-draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def log_posterior_terms(f: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = expit(f)
    return y * np.log(p + 1e-10) + (1.0 - y) * np.log(1.0 - p + 1e-10)


def bench(name: str, F: np.ndarray, y: np.ndarray, nu: np.ndarray, n_iter: int) -> dict:
    for _ in range(3):
        f = F @ nu
        p = expit(f)
        _ = F.T @ (y - p)

    t0 = time.perf_counter()
    last_logp = np.nan
    grad_norm = np.nan
    for _ in range(n_iter):
        f = F @ nu
        p = expit(f)
        residual = y - p
        grad = F.T @ residual - nu
        log_lik = np.sum(log_posterior_terms(f, y))
        log_prior = -0.5 * np.dot(nu, nu)
        last_logp = float(log_lik + log_prior)
        grad_norm = float(np.linalg.norm(grad))

    elapsed = time.perf_counter() - t0
    ms_per_iter = 1000.0 * elapsed / max(n_iter, 1)
    print(
        f"{name:<24} {ms_per_iter:>10.2f} ms / grad+logp"
        f"   logp={last_logp: .6f}   ||grad||={grad_norm: .6e}"
    )
    return {"ms_per_iter": ms_per_iter, "logp": last_logp, "grad_norm": grad_norm}


def precision_diagnostic(
    F32: np.ndarray,
    y32: np.ndarray,
    F64: np.ndarray,
    y64: np.ndarray,
    rng: np.random.Generator,
    n_draws: int,
) -> None:
    logp_32 = []
    logp_32_sum64 = []
    logp_64 = []

    for _ in range(n_draws):
        nu32 = rng.standard_normal(F32.shape[1]).astype(np.float32)
        nu64 = nu32.astype(np.float64)

        f32 = F32 @ nu32
        terms32 = log_posterior_terms(f32, y32)
        prior32 = -0.5 * np.dot(nu32, nu32)
        logp_32.append(float(np.sum(terms32) + prior32))
        logp_32_sum64.append(float(np.sum(terms32.astype(np.float64)) + float(prior32)))

        f64 = F64 @ nu64
        terms64 = log_posterior_terms(f64, y64)
        prior64 = -0.5 * np.dot(nu64, nu64)
        logp_64.append(float(np.sum(terms64) + prior64))

    logp_32 = np.asarray(logp_32)
    logp_32_sum64 = np.asarray(logp_32_sum64)
    logp_64 = np.asarray(logp_64)

    pure32_vs_sum64 = np.abs(logp_32 - logp_32_sum64)
    pure32_vs_64 = np.abs(logp_32 - logp_64)
    sum64_vs_64 = np.abs(logp_32_sum64 - logp_64)

    print("\nLog-posterior precision diagnostic over random nu draws")
    print(f"  draws: {n_draws}")
    print(
        "  |float32 sum - float32 matvec/float64 sum|: "
        f"mean={pure32_vs_sum64.mean():.3e}, max={pure32_vs_sum64.max():.3e}"
    )
    print(
        "  |float32 full - float64 full|:              "
        f"mean={pure32_vs_64.mean():.3e}, max={pure32_vs_64.max():.3e}"
    )
    print(
        "  |float32 matvec/float64 sum - float64 full|:"
        f" mean={sum64_vs_64.mean():.3e}, max={sum64_vs_64.max():.3e}"
    )


def main() -> None:
    args = parse_args()
    factor_path = Path(args.factor)
    labels_path = Path(args.labels)

    print(f"Loading factor: {factor_path}")
    F32 = np.load(factor_path).astype(np.float32, copy=False)
    print(f"Loading labels: {labels_path}")
    y32 = np.load(labels_path).astype(np.float32, copy=False).squeeze()

    print("Promoting copies to float64...")
    F64 = F32.astype(np.float64)
    y64 = y32.astype(np.float64)

    rng = np.random.default_rng(args.seed)
    nu32 = rng.standard_normal(F32.shape[1]).astype(np.float32)
    nu64 = nu32.astype(np.float64)

    print(f"\nShape: F={F32.shape}, y={y32.shape}, rank={F32.shape[1]}")
    print(f"Iterations per timing run: {args.n_iter}\n")
    print(f"{'mode':<24} {'time':>28}")
    print("-" * 78)
    stats32 = bench("float32", F32, y32, nu32, args.n_iter)
    stats64 = bench("float64", F64, y64, nu64, args.n_iter)

    ratio = stats64["ms_per_iter"] / stats32["ms_per_iter"]
    print("-" * 78)
    print(f"float64 / float32 time ratio: {ratio:.3f}x")

    precision_diagnostic(
        F32=F32,
        y32=y32,
        F64=F64,
        y64=y64,
        rng=rng,
        n_draws=args.n_precision_draws,
    )


if __name__ == "__main__":
    main()
