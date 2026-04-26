"""
Experiment 0 variant: RPCholesky benchmark on a kernel matrix chosen to be
closer to the paper's "smile" example.

Compared with the baseline exp0 setup, this script uses:
- a smile-shaped point cloud in R^2
- a Gaussian kernel with bandwidth close to the reference benchmark
- a log-scale error plot to reveal small differences at high rank

This is intended to produce behavior that is qualitatively closer to the paper
than the uniform-in-[0,1]^10 baseline.
"""

from __future__ import annotations

import os
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from my_cholesky.arpcholesky import arpcholesky
from my_cholesky.matrix import KernelMatrix
from my_cholesky.rpcholesky_variants import block_rpcholesky, simple_rpcholesky


def make_smile_data(
    n: int,
    eye_points: int | None = None,
    mouth_points: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Reproduce the smile-style geometry used in the reference code.

    The point cloud lives in R^2 and contains:
    - two filled disk "eyes"
    - a parabola-shaped mouth
    - a circular face outline
    """
    rng = np.random.default_rng(seed)
    small = int(np.ceil(n ** 0.5))
    if eye_points is None:
        eye_points = small
    if mouth_points is None:
        mouth_points = int(np.ceil(n / 10.0))
    face_points = n - 2 * eye_points - mouth_points

    X = np.zeros((n, 2), dtype=float)
    idx = 0

    for x_shift in (-4.0, 4.0):
        for _ in range(eye_points):
            while True:
                x = 2.0 * rng.random() - 1.0
                y = 2.0 * rng.random() - 1.0
                if x * x + y * y <= 1.0:
                    X[idx, 0] = x + x_shift
                    X[idx, 1] = y + 4.0
                    idx += 1
                    break

    for x in np.linspace(-5.0, 5.0, mouth_points):
        X[idx, 0] = x
        X[idx, 1] = x**2 / 16.0 - 5.0
        idx += 1

    for theta in np.linspace(0.0, 2.0 * np.pi, face_points, endpoint=False):
        X[idx, 0] = 10.0 * np.cos(theta)
        X[idx, 1] = 10.0 * np.sin(theta)
        idx += 1

    return X


def build_fast_decay_kernel_matrix(
    n: int,
    bandwidth: float = 0.2,
    extra_stability: bool = True,
    seed: int = 0,
):
    X = make_smile_data(n, seed=seed)
    return (
        KernelMatrix(
            X,
            kernel="gaussian",
            bandwidth=bandwidth,
            extra_stability=extra_stability,
        ),
        X,
    )


def rpcholesky(A, k: int, accelerated: bool = True, b: int = 120):
    """Local dispatcher to match the reference benchmark's calling style."""
    if accelerated:
        return arpcholesky(A, k, b=b)
    return block_rpcholesky(A, k, b=b)


def save_eigenvalue_decay_plot(
    data_dir: str,
    bandwidth: float,
    extra_stability: bool,
) -> None:
    """
    Save a dense eigenspectrum plot on a moderate-size instance to show the
    intended fast spectral decay explicitly.
    """
    eig_n = 3000
    A_eig, _ = build_fast_decay_kernel_matrix(
        eig_n,
        bandwidth=bandwidth,
        extra_stability=extra_stability,
        seed=999,
    )
    K = A_eig[:, :]
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.sort(np.maximum(eigvals, 1e-14))[::-1]

    plt.figure(figsize=(7, 5))
    plt.semilogy(np.arange(1, len(eigvals) + 1), eigvals, linewidth=1.5)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue (log scale)")
    plt.title("Fast-decay kernel eigenspectrum (N={})".format(eig_n))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp0_fast_decay_eigs.png"), dpi=160)
    plt.close()


def main() -> None:
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    fixed_n = 20000
    ks = [10] + list(range(100, 700, 100))
    trials = 5
    n_values = [1000, 5000, 10000, 50000]

    b = 120
    bandwidth = 0.2
    extra_stability = True

    A_fixed, X_fixed = build_fast_decay_kernel_matrix(
        fixed_n,
        bandwidth=bandwidth,
        extra_stability=extra_stability,
        seed=11,
    )
    print("Sample of X_fixed:\n", X_fixed[:5])
    print("Sample of kernel matrix:\n", A_fixed[:5, :5])
    trace_fixed = A_fixed.trace()

    methods = {
        "Accel": lambda k: rpcholesky(A_fixed, k, accelerated=True, b=b),
        "Block": lambda k: rpcholesky(A_fixed, k, accelerated=False, b=b),
        "Basic": lambda k: simple_rpcholesky(A_fixed, k),
    }

    times = {method: np.zeros((len(ks), trials)) for method in methods}
    errs = {method: np.zeros((len(ks), trials)) for method in methods}

    print(
        "=== Smile-style fixed N sweep (N={}, dim=2, bandwidth={}, extra_stability={}) ===".format(
            fixed_n, bandwidth, extra_stability
        )
    )
    print("k\tmethod\tmean_time_sec\tmean_rel_err")
    for idx, k in enumerate(ks):
        for method_name, method in methods.items():
            for trial in range(trials):
                start = perf_counter()
                lra = method(k)
                times[method_name][idx, trial] = perf_counter() - start
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

    times_vs_n = {method: np.full(len(n_values), np.nan) for method in methods}
    k_values_for_n = np.zeros(len(n_values), dtype=int)

    print("=== Variable k sweep (k=max(50, N//50), approx rank ratio ~2%) ===")
    print("N\tk\tmethod\tmean_time_sec")
    for i, n in enumerate(n_values):
        k_for_n = max(50, n // 50)
        k_values_for_n[i] = k_for_n

        A_n, _ = build_fast_decay_kernel_matrix(
            n,
            bandwidth=bandwidth,
            extra_stability=extra_stability,
            seed=100 + i,
        )
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
                start = perf_counter()
                _ = method()
                trial_times[t] = perf_counter() - start

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

    mat_output = {
        **{"{}_times".format(method): times[method] for method in methods},
        **{"{}_errs".format(method): errs[method] for method in methods},
        **{"{}_times_vs_N".format(method): times_vs_n[method] for method in methods},
        "N_values": np.array(n_values, dtype=int),
        "k_values": np.array(ks, dtype=int),
        "k_values_for_N": k_values_for_n,
        "fixed_N": np.array([fixed_n], dtype=int),
        "X_fixed": X_fixed,
        "bandwidth": np.array([bandwidth], dtype=float),
        "extra_stability": np.array([int(extra_stability)], dtype=int),
    }
    sp.io.savemat(os.path.join(data_dir, "exp0_fast_decay_results.mat"), mat_output)

    plt.figure(figsize=(7, 5))
    for method in methods:
        plt.semilogy(ks, np.mean(errs[method], axis=1), marker="o", label=method)
    plt.xlabel("Rank k")
    plt.ylabel("Relative trace error")
    plt.title("Smile-style approximation error vs rank (N={})".format(fixed_n))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp0_fast_decay_error_vs_k.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    for method in methods:
        plt.plot(ks, np.mean(times[method], axis=1), marker="o", label=method)
    plt.xlabel("Rank k")
    plt.ylabel("Runtime (s)")
    plt.title("Smile-style runtime vs rank (N={})".format(fixed_n))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp0_fast_decay_time_vs_k.png"), dpi=160)
    plt.close()

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
    plt.title("Smile-style runtime vs N (k=max(50, N//50), ~2% rank ratio)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp0_fast_decay_time_vs_N.png"), dpi=160)
    plt.close()

    save_eigenvalue_decay_plot(
        data_dir=data_dir,
        bandwidth=bandwidth,
        extra_stability=extra_stability,
    )

    print("Saved:")
    print("- data/exp0_fast_decay_results.mat")
    print("- data/exp0_fast_decay_error_vs_k.png")
    print("- data/exp0_fast_decay_time_vs_k.png")
    print("- data/exp0_fast_decay_time_vs_N.png")
    print("- data/exp0_fast_decay_eigs.png")


if __name__ == "__main__":
    main()
