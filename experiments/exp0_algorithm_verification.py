"""
Experiment 0/1 style benchmark adapted from:
Randomly-Pivoted-Cholesky/block_experiments/test_accelerated.py

Kernel choice rationale:
- Two synthetic generators are run side-by-side for the same RPCholesky
  variants:
    * "uniform"  : X ~ Uniform([0,1]^d), d=10. Near-flat spectrum baseline
                   (high intrinsic dimension + small bandwidth).
    * "clusters" : X drawn from a mixture of n_centers=5 tight Gaussian blobs
                   (cluster_std=0.05) in d=3. This produces a kernel matrix
                   whose eigenvalues decay rapidly (effective rank ~ number
                   of clusters), which is the regime where low-rank Cholesky
                   variants are most useful.
- Both use the Gaussian kernel via KernelMatrix.

N selection rationale:
- Fixed-N sweep at N=20000 for error-vs-k and time-vs-k.
- Time-vs-N sweep: k = max(50, N//50) (~2% rank ratio) so approximation
  difficulty is roughly constant across N values.
- Basic RPCholesky is skipped when N > 10000 because it gets very slow.

Spectrum diagnostic:
- For each generator we save a dense-spectrum semilog plot at n=2000
  (data/exp0/spectrum_<generator>.png) so the eigenvalue-decay assumption
  is visually verifiable before trusting the rank sweep results.
"""

from __future__ import annotations

import os
import sys
from time import perf_counter

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
from my_cholesky.result_logging import append_result_rows
from my_cholesky.rpcholesky_variants import block_rpcholesky, simple_rpcholesky


OUTPUT_DIR = "data/exp0"


def sample_clusters(rng, n: int, d: int, n_centers: int = 5, cluster_std: float = 0.05):
    centers = rng.random((n_centers, d))
    assignments = rng.integers(0, n_centers, size=n)
    noise = rng.standard_normal((n, d)) * cluster_std
    return centers[assignments] + noise


def build_kernel_matrix(
    n: int,
    d: int = 10,
    bandwidth: float = 1.0,
    seed: int = 0,
    data_generator: str = "uniform",
    n_centers: int = 5,
    cluster_std: float = 0.05,
):
    rng = np.random.default_rng(seed)
    if data_generator == "uniform":
        X = rng.random((n, d))
    elif data_generator == "clusters":
        X = sample_clusters(rng, n, d, n_centers=n_centers, cluster_std=cluster_std)
    else:
        raise ValueError(f"Unknown data_generator: {data_generator}")
    return KernelMatrix(X, kernel="gaussian", bandwidth=bandwidth), X


def rpcholesky(A, k: int, accelerated: bool = True, b: int = 120):
    """
    Local dispatcher to match the reference benchmark's calling style.
    """
    if accelerated:
        return arpcholesky(A, k, b=b)
    return block_rpcholesky(A, k, b=b)


def plot_spectrum(generator_name: str, gen_kwargs: dict, n_diag: int = 2000, seed: int = 999):
    A, _ = build_kernel_matrix(
        n_diag, seed=seed, data_generator=generator_name, **gen_kwargs
    )
    M = np.asarray(A[:, :])
    eig = np.linalg.eigvalsh(M)[::-1]
    eig = np.maximum(eig, 1e-16)

    plt.figure(figsize=(7, 5))
    plt.semilogy(np.arange(1, len(eig) + 1), eig)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue (log scale)")
    plt.title(
        "Spectrum (generator={}, n={}, d={}, bandwidth={})".format(
            generator_name, n_diag, gen_kwargs["d"], gen_kwargs["bandwidth"]
        )
    )
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    out_path = "{}/spectrum_{}.png".format(OUTPUT_DIR, generator_name)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print("Saved spectrum diagnostic: {}".format(out_path))


def run_sweeps(
    generator_name: str,
    gen_kwargs: dict,
    fixed_n: int,
    ks: list,
    n_values: list,
    trials: int,
    b: int,
):
    suffix = "_{}".format(generator_name)
    print("\n##############################")
    print("# Generator: {}".format(generator_name))
    print("##############################")

    # ---------- Sweep 1 & 2: error/time vs k (fixed N) ----------
    A_fixed, X_fixed = build_kernel_matrix(
        fixed_n, seed=11, data_generator=generator_name, **gen_kwargs
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
        "=== Fixed N sweep (gen={}, N={}, d={}, bandwidth={}) ===".format(
            generator_name, fixed_n, gen_kwargs["d"], gen_kwargs["bandwidth"]
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

    # ---------- Accel vs Block comparison (fixed N) ----------
    accel_t = np.mean(times["Accel"], axis=1)
    block_t = np.mean(times["Block"], axis=1)
    accel_e = np.mean(errs["Accel"], axis=1)
    block_e = np.mean(errs["Block"], axis=1)
    speedup_k = np.divide(
        block_t, accel_t, out=np.full_like(block_t, np.nan), where=accel_t > 0
    )
    err_ratio_k = np.divide(
        accel_e, block_e, out=np.full_like(accel_e, np.nan), where=block_e > 0
    )

    print(
        "=== Accel vs Block comparison (gen={}, N={}) ===".format(
            generator_name, fixed_n
        )
    )
    print("k\tBlock_t\t\tAccel_t\t\tspeedup\t\tBlock_err\t\tAccel_err\t\terr_ratio")
    for idx, k in enumerate(ks):
        print(
            "{}\t{:.6f}\t{:.6f}\t{:.4f}\t\t{:.3e}\t\t{:.3e}\t\t{:.4f}".format(
                k,
                block_t[idx],
                accel_t[idx],
                speedup_k[idx],
                block_e[idx],
                accel_e[idx],
                err_ratio_k[idx],
            )
        )
    print()

    # ---------- Sweep 3: time vs N (rank ratio fixed ~2%) ----------
    times_vs_n = {method: np.full(len(n_values), np.nan) for method in methods}
    k_values_for_n = np.zeros(len(n_values), dtype=int)

    print(
        "=== Variable k sweep (gen={}, k=max(50, N//50), approx rank ratio ~2%) ===".format(
            generator_name
        )
    )
    print("N\tk\tmethod\tmean_time_sec")
    for i, n in enumerate(n_values):
        k_for_n = max(50, n // 50)
        k_values_for_n[i] = k_for_n

        A_n, _ = build_kernel_matrix(
            n, seed=100 + i, data_generator=generator_name, **gen_kwargs
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

    # ---------- Accel vs Block comparison (variable N) ----------
    accel_t_n = times_vs_n["Accel"]
    block_t_n = times_vs_n["Block"]
    speedup_n = np.divide(
        block_t_n,
        accel_t_n,
        out=np.full_like(block_t_n, np.nan),
        where=np.isfinite(block_t_n) & np.isfinite(accel_t_n) & (accel_t_n > 0),
    )

    print(
        "=== Accel vs Block comparison (gen={}, variable N) ===".format(generator_name)
    )
    print("N\tk\tBlock_t\t\tAccel_t\t\tspeedup")
    for i, n in enumerate(n_values):
        print(
            "{}\t{}\t{:.6f}\t{:.6f}\t{:.4f}".format(
                n,
                k_values_for_n[i],
                block_t_n[i],
                accel_t_n[i],
                speedup_n[i],
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
        "accel_vs_block_speedup": speedup_k,
        "accel_vs_block_err_ratio": err_ratio_k,
        "accel_vs_block_speedup_vs_N": speedup_n,
    }
    sp.io.savemat("{}/results{}.mat".format(OUTPUT_DIR, suffix), mat_output)

    # ---------- Plot 1: error vs k ----------
    plt.figure(figsize=(7, 5))
    for method in methods:
        plt.plot(ks, np.mean(errs[method], axis=1), marker="o", label=method)
    plt.xlabel("Rank k")
    plt.ylabel("Relative trace error")
    plt.title(
        "Approximation Error vs Rank ({}, N={})".format(generator_name, fixed_n)
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}/error_vs_k{}.png".format(OUTPUT_DIR, suffix), dpi=160)
    plt.close()

    # ---------- Plot 2: runtime vs k ----------
    plt.figure(figsize=(7, 5))
    for method in methods:
        plt.plot(ks, np.mean(times[method], axis=1), marker="o", label=method)
    plt.xlabel("Rank k")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs Rank ({}, N={})".format(generator_name, fixed_n))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}/time_vs_k{}.png".format(OUTPUT_DIR, suffix), dpi=160)
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
    plt.title(
        "Runtime vs N ({}, k=max(50, N//50), ~2% rank ratio)".format(generator_name)
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}/time_vs_N{}.png".format(OUTPUT_DIR, suffix), dpi=160)
    plt.close()

    # ---------- Plot 4: Accel vs Block speedup vs k ----------
    plt.figure(figsize=(7, 5))
    plt.plot(ks, speedup_k, marker="o", label="Block / Accel (fixed N)")
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="parity")
    plt.xlabel("Rank k")
    plt.ylabel("Speedup (Block_time / Accel_time)")
    plt.title(
        "Accel vs Block Speedup ({}, N={})".format(generator_name, fixed_n)
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "{}/accel_vs_block_speedup{}.png".format(OUTPUT_DIR, suffix), dpi=160
    )
    plt.close()

    # ---------- CSV rows ----------
    csv_rows = []
    data_gen_label = "{}_gaussian_kernel".format(generator_name)
    artifact_path = "{}/results{}.mat".format(OUTPUT_DIR, suffix)

    for method_name in methods:
        for idx, k in enumerate(ks):
            csv_rows.append(
                {
                    "experiment": "exp0",
                    "script_path": "experiments/exp0_algorithm_verification.py",
                    "artifacts": artifact_path,
                    "data_generator": data_gen_label,
                    "data_seed": 11,
                    "kernel": "gaussian",
                    "kernel_bandwidth": gen_kwargs["bandwidth"],
                    "synthetic_d": gen_kwargs["d"],
                    "fixed_n": fixed_n,
                    "n": fixed_n,
                    "k": int(k),
                    "trials": trials,
                    "block_size_b": b,
                    "method_name": method_name,
                    "timing_scope": "factorization_only",
                    "mean_time_sec": float(np.mean(times[method_name][idx, :])),
                    "std_time_sec": float(np.std(times[method_name][idx, :])),
                    "factor_time_sec": float(np.mean(times[method_name][idx, :])),
                    "approx_error_fro_rel": float(np.mean(errs[method_name][idx, :])),
                    "unavailable_reason": "classification metrics are not applicable to kernel factorization benchmark",
                }
            )
    for method_name in methods:
        for idx, n in enumerate(n_values):
            if not np.isfinite(times_vs_n[method_name][idx]):
                continue
            csv_rows.append(
                {
                    "experiment": "exp0",
                    "script_path": "experiments/exp0_algorithm_verification.py",
                    "artifacts": artifact_path,
                    "data_generator": data_gen_label,
                    "data_seed": 100 + idx,
                    "kernel": "gaussian",
                    "kernel_bandwidth": gen_kwargs["bandwidth"],
                    "synthetic_d": gen_kwargs["d"],
                    "n": int(n),
                    "k": int(k_values_for_n[idx]),
                    "trials": trials,
                    "block_size_b": b,
                    "method_name": method_name,
                    "timing_scope": "factorization_only",
                    "mean_time_sec": float(times_vs_n[method_name][idx]),
                    "factor_time_sec": float(times_vs_n[method_name][idx]),
                    "unavailable_reason": "classification metrics are not applicable to kernel factorization benchmark",
                }
            )

    artifacts = [
        "{}/results{}.mat".format(OUTPUT_DIR, suffix),
        "{}/error_vs_k{}.png".format(OUTPUT_DIR, suffix),
        "{}/time_vs_k{}.png".format(OUTPUT_DIR, suffix),
        "{}/time_vs_N{}.png".format(OUTPUT_DIR, suffix),
        "{}/accel_vs_block_speedup{}.png".format(OUTPUT_DIR, suffix),
    ]
    return csv_rows, artifacts


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sweep settings (shared across generators)
    fixed_n = 20000
    ks = [10] + list(range(100, 700, 100))
    trials = 5
    n_values = [1000, 5000, 10000, 50000]
    b = 120

    # Per-generator config. "uniform" mirrors the original baseline; "clusters"
    # is the new fast-decay regime.
    generators = {
        "uniform": {"d": 10, "bandwidth": 1.0},
        "clusters": {
            "d": 3,
            "bandwidth": 1.0,
            "n_centers": 5,
            "cluster_std": 0.05,
        },
    }

    # Spectrum diagnostics first so decay can be sanity-checked before the
    # heavy sweeps run.
    for gen_name, gen_kwargs in generators.items():
        plot_spectrum(gen_name, gen_kwargs)

    all_csv_rows = []
    all_artifacts = []
    for gen_name, gen_kwargs in generators.items():
        rows, artifacts = run_sweeps(
            generator_name=gen_name,
            gen_kwargs=gen_kwargs,
            fixed_n=fixed_n,
            ks=ks,
            n_values=n_values,
            trials=trials,
            b=b,
        )
        all_csv_rows.extend(rows)
        all_artifacts.extend(artifacts)

    csv_path = append_result_rows(all_csv_rows)

    print("\nSaved:")
    for path in all_artifacts:
        print("- {}".format(path))
    for gen_name in generators:
        print("- {}/spectrum_{}.png".format(OUTPUT_DIR, gen_name))
    print("- appended CSV metrics to {}".format(csv_path))


if __name__ == "__main__":
    main()
