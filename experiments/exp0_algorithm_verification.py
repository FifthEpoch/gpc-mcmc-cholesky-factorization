"""
Experiment 0/1 style benchmark adapted from:
Randomly-Pivoted-Cholesky/block_experiments/test_accelerated.py

Kernel choice rationale:
- We avoid smile()/expspiral()/outliers() and use a synthetic Gaussian
  kernel matrix A = KernelMatrix(X) with X ~ Uniform([0,1]^d), d=10.
- This is the most neutral, scalable baseline for comparing algorithmic
  behavior because it avoids geometry-specific structure and keeps data
  generation cheap and reproducible.

N selection rationale:
- We benchmark fixed-N sweeps at N=20000 for error-vs-k and time-vs-k.
- For time-vs-N, k scales with N to keep rank ratio ~2%, so approximation
  difficulty is constant across N values.
- Basic RPCholesky can become very slow at large N, so it is skipped
  when N > 10000.
"""

from __future__ import annotations

import os
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


# Allow running this script directly without installing the package.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from my_cholesky.arpcholesky import arpcholesky
from my_cholesky.matrix import KernelMatrix
from my_cholesky.rpcholesky_variants import block_rpcholesky, simple_rpcholesky


def build_kernel_matrix(n: int, d: int = 10, bandwidth: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    return KernelMatrix(X, kernel="gaussian", bandwidth=bandwidth), X


def rpcholesky(A, k: int, accelerated: bool = True, b: int = 120):
    """
    Local dispatcher to match the reference benchmark's calling style.
    """
    if accelerated:
        return arpcholesky(A, k, b=b)
    return block_rpcholesky(A, k, b=b)


def main() -> None:
    os.makedirs("data", exist_ok=True)

    # Sweep settings
    fixed_n = 20000
    ks = [10] + list(range(100, 700, 100))
    trials = 5
    n_values = [1000, 5000, 10000, 50000]

    b = 120
    d = 10
    bandwidth = 1.0

    # ---------- Sweep 1 & 2: error/time vs k (fixed N) ----------
    A_fixed, X_fixed = build_kernel_matrix(fixed_n, d=d, bandwidth=bandwidth, seed=11)
    trace_fixed = A_fixed.trace()

    methods = {
        "Accel": lambda k: rpcholesky(A_fixed, k, accelerated=True, b=b),
        "Block": lambda k: rpcholesky(A_fixed, k, accelerated=False, b=b),
        "Basic": lambda k: simple_rpcholesky(A_fixed, k),
    }

    times = {method: np.zeros((len(ks), trials)) for method in methods}
    errs = {method: np.zeros((len(ks), trials)) for method in methods}

    print("=== Fixed N sweep (N={}, d={}, bandwidth={}) ===".format(fixed_n, d, bandwidth))
    print("k\tmethod\tmean_time_sec\tmean_rel_err")
    for idx, k in enumerate(ks):
        for method_name, method in methods.items():
            for trial in range(trials):
                start = time()
                lra = method(k)
                times[method_name][idx, trial] = time() - start
                errs[method_name][idx, trial] = (trace_fixed - lra.trace()) / trace_fixed

            print(
                "{}\t{}\t{:.6f}\t{:.8f}".format(
                    k,
                    method_name,
                    np.mean(times[method_name][idx, :]),
                    np.mean(errs[method_name][idx, :]),
                )
            )
        print()

    # ---------- Sweep 3: time vs N (rank ratio fixed ~2%) ----------
    times_vs_n = {method: np.full(len(n_values), np.nan) for method in methods}
    k_values_for_n = np.zeros(len(n_values), dtype=int)

    print("=== Variable k sweep (k=max(50, N//50), approx rank ratio ~2%) ===")
    print("N\tk\tmethod\tmean_time_sec")
    for i, n in enumerate(n_values):
        # k scales with N to keep rank ratio ~2%,
        # so approximation difficulty is constant across N values.
        k_for_n = max(50, n // 50)
        k_values_for_n[i] = k_for_n

        A_n, _ = build_kernel_matrix(n, d=d, bandwidth=bandwidth, seed=100 + i)
        methods_n = {
            "Accel": lambda A=A_n, k=k_for_n: rpcholesky(A, k, accelerated=True, b=b),
            "Block": lambda A=A_n, k=k_for_n: rpcholesky(A, k, accelerated=False, b=b),
            "Basic": lambda A=A_n, k=k_for_n: simple_rpcholesky(A, k),
        }

        for method_name, method in methods_n.items():
            if method_name == "Basic" and n > 10000:
                print("{}\t{}\t{}\t{}".format(n, k_for_n, method_name, "SKIPPED"))
                continue

            trial_times = np.zeros(trials)
            for t in range(trials):
                start = time()
                _ = method()
                trial_times[t] = time() - start

            times_vs_n[method_name][i] = np.mean(trial_times)
            print(
                "{}\t{}\t{}\t{:.6f}".format(
                    n,
                    k_for_n,
                    method_name,
                    times_vs_n[method_name][i],
                )
            )
        print()

    # ---------- Save MAT output ----------
    mat_output = {
        **{"{}_times".format(method): times[method] for method in methods},
        **{"{}_errs".format(method): errs[method] for method in methods},
        **{"{}_times_vs_N".format(method): times_vs_n[method] for method in methods},
        "N_values": np.array(n_values, dtype=int),
        "k_values": np.array(ks, dtype=int),
        "k_values_for_N": k_values_for_n,
        "fixed_N": np.array([fixed_n], dtype=int),
        "X_fixed": X_fixed,
    }
    sp.io.savemat("data/exp0_results.mat", mat_output)

    # ---------- Plot 1: error vs k ----------
    plt.figure(figsize=(7, 5))
    for method in methods:
        plt.plot(ks, np.mean(errs[method], axis=1), marker="o", label=method)
    plt.xlabel("Rank k")
    plt.ylabel("Relative trace error")
    plt.title("Approximation Error vs Rank (N={})".format(fixed_n))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/exp0_error_vs_k.png", dpi=160)
    plt.close()

    # ---------- Plot 2: runtime vs k ----------
    plt.figure(figsize=(7, 5))
    for method in methods:
        plt.plot(ks, np.mean(times[method], axis=1), marker="o", label=method)
    plt.xlabel("Rank k")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs Rank (N={})".format(fixed_n))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/exp0_time_vs_k.png", dpi=160)
    plt.close()

    # ---------- Plot 3: runtime vs N ----------
    plt.figure(figsize=(7, 5))
    for method in methods:
        mask = np.isfinite(times_vs_n[method])
        plt.plot(
            np.array(n_values)[mask],
            times_vs_n[method][mask],
            marker="o",
            label=method,
        )
    plt.xlabel("N")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs N (k=max(50, N//50), ~2% rank ratio)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/exp0_time_vs_N.png", dpi=160)
    plt.close()

    print("Saved:")
    print("- data/exp0_results.mat")
    print("- data/exp0_error_vs_k.png")
    print("- data/exp0_time_vs_k.png")
    print("- data/exp0_time_vs_N.png")


if __name__ == "__main__":
    main()

