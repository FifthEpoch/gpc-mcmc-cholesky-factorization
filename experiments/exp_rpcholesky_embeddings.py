#!/usr/bin/env python3
"""
Driver: accelerated RPCholesky on pre-computed embeddings (e.g., 300k x 512).

Loads embeddings from a .npy file, builds a lazy KernelMatrix (never
materializes the full N x N kernel), and runs accelerated RPCholesky at
a list of ranks. Saves each low-rank factor and a summary of timing /
trace-error / query counts.

Example (small test):
    python exp_rpcholesky_embeddings.py \
        --embeddings data/embeddings/pcam_train_embeddings.npy \
        --labels data/embeddings/pcam_train_labels.npy \
        --subsample 5000 \
        --k-values 10 20 50 100 200 \
        --output-dir data/rpchol_small

Example (full 300k):
    python exp_rpcholesky_embeddings.py \
        --embeddings data/embeddings/pcam_train_embeddings.npy \
        --labels data/embeddings/pcam_train_labels.npy \
        --k-values 10 20 50 100 200 \
        --output-dir data/rpchol_full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# Project-local imports: same pattern as your existing experiments.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from my_cholesky.matrix import KernelMatrix  # noqa: E402
from my_cholesky.arpcholesky import arpcholesky  # noqa: E402


def stratified_subsample_indices(y: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    """Same stratified random subsample policy as exp1 full-HMC embeddings."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).reshape(-1)
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
    return idx


def load_embeddings(
    emb_path: Path,
    lbl_path: Path | None,
    subsample: int | None,
    seed: int,
    stratified_subsample: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Load embeddings (and optional labels) from .npy files."""
    X = np.load(emb_path).astype(np.float32, copy=False)
    y = np.load(lbl_path) if lbl_path is not None else None
    if y is not None and y.ndim > 1:
        y = y.squeeze()
    print(f"Loaded embeddings: {X.shape}  dtype={X.dtype}")
    if y is not None:
        print(f"Loaded labels:     {y.shape}  dtype={y.dtype}")
    idx = np.arange(X.shape[0])
    if subsample is not None and subsample < X.shape[0]:
        if stratified_subsample:
            if y is None:
                raise ValueError("--stratified-subsample requires --labels.")
            idx = stratified_subsample_indices(y, subsample, seed)
        else:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X.shape[0], size=subsample, replace=False)
            idx.sort()  # keep label order stable for downstream convenience
        X = X[idx]
        if y is not None:
            y = y[idx]
        print(f"Subsampled to:     {X.shape}")
    return X, y, idx


def standardize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize selected train features using their own moments."""
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((X - mean) / std).astype(np.float32, copy=False), mean, std


def build_kernel_matrix(
    X: np.ndarray, kernel: str, bandwidth: str | float
) -> KernelMatrix:
    """Construct the lazy KernelMatrix wrapper around X."""
    t0 = time.perf_counter()
    A = KernelMatrix(X, kernel=kernel, bandwidth=bandwidth)
    setup_time = time.perf_counter() - t0
    print(f"KernelMatrix built:   bandwidth={A.bandwidth:.4f}  setup={setup_time:.3f}s")
    return A


def run_rank(A: KernelMatrix, k: int, b: int, stoptol: float) -> dict:
    """Run accelerated RPCholesky at rank k; return factor and stats."""
    print(f"\n--- rank k={k} (block b={b}) ---")
    A.reset()  # zero the query counter for a clean per-rank measurement

    t0 = time.perf_counter()
    lra = arpcholesky(A, k=k, b=b, stoptol=stoptol)
    elapsed = time.perf_counter() - t0

    F = lra.get_left_factor()          # shape (N, k_actual)
    pivots = lra.get_indices()
    k_actual = F.shape[1]
    n = A.shape[0]

    # Trace of K is n (RBF diagonal is 1 everywhere). Trace of F F^T is ||F||_F^2.
    trace_K = float(A.trace())
    trace_FFt = float(np.sum(F * F))
    trace_residual = trace_K - trace_FFt
    rel_trace_error = trace_residual / trace_K if trace_K > 0 else float("nan")

    stats = {
        "k_requested": int(k),
        "k_actual": int(k_actual),
        "n": int(n),
        "elapsed_sec": float(elapsed),
        "trace_K": trace_K,
        "trace_FFt": trace_FFt,
        "trace_residual": trace_residual,
        "rel_trace_error": float(rel_trace_error),
        "kernel_queries": int(A.num_queries()),
    }

    saved_query_count = A.num_queries()
    K_pivots = A[pivots, pivots]
    K_pivots = 0.5 * (K_pivots + K_pivots.T)
    A.queries = saved_query_count

    print(f"  done in {elapsed:.2f}s  actual rank={k_actual}")
    print(f"  trace(K)={trace_K:.2f}  trace(FF^T)={trace_FFt:.2f}")
    print(f"  relative trace error: {rel_trace_error:.4e}")
    print(f"  kernel entries queried: {stats['kernel_queries']:,}  "
          f"(vs {n*n:,} for full matrix)")

    return {"factor": F, "pivots": pivots, "K_pivots": K_pivots, "stats": stats}


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    emb_path = Path(args.embeddings)
    lbl_path = Path(args.labels) if args.labels else None
    X, y, selected_indices = load_embeddings(
        emb_path,
        lbl_path,
        args.subsample,
        args.seed,
        args.stratified_subsample,
    )
    train_mean = np.zeros((1, X.shape[1]), dtype=np.float32)
    train_std = np.ones((1, X.shape[1]), dtype=np.float32)
    if args.standardize:
        X, train_mean, train_std = standardize_features(X)
        print("Standardized selected train embeddings with train-set moments.")

    # 2. Build the lazy kernel matrix
    A = build_kernel_matrix(X, kernel=args.kernel, bandwidth=args.bandwidth)

    # 3. Sweep ranks
    all_stats = []
    for k in args.k_values:
        result = run_rank(A, k=k, b=args.block_size, stoptol=args.stoptol)
        F = result["factor"]
        pivots = result["pivots"]
        K_pivots = result["K_pivots"]
        stats = result["stats"]

        # Save the factor for this rank
        factor_path = out_dir / f"factor_k{stats['k_actual']}.npy"
        np.save(factor_path, F.astype(np.float32, copy=False))
        stats["factor_path"] = str(factor_path)
        stats["factor_bytes"] = int(F.nbytes)
        print(f"  saved factor: {factor_path}  ({F.nbytes/1e6:.1f} MB)")

        pivots_path = out_dir / f"pivots_k{stats['k_actual']}.npy"
        np.save(pivots_path, pivots.astype(np.int64))
        stats["pivots_path"] = str(pivots_path)
        print(f"  saved pivots: {pivots_path}")

        kpp_path = out_dir / f"kernel_submatrix_k{stats['k_actual']}.npy"
        np.save(kpp_path, K_pivots.astype(np.float64, copy=False))
        stats["kernel_submatrix_path"] = str(kpp_path)
        print(f"  saved kernel submatrix: {kpp_path}  ({K_pivots.shape})")

        all_stats.append(stats)

    # 4. Save embeddings and labels alongside factors so downstream MCMC can load both easily
    emb_out = out_dir / "embeddings.npy"
    np.save(emb_out, X.astype(np.float32, copy=False))
    print(f"\nSaved embeddings: {emb_out}")

    idx_out = out_dir / "indices.npy"
    np.save(idx_out, selected_indices.astype(np.int64, copy=False))
    print(f"Saved selected indices: {idx_out}")

    if args.standardize:
        mean_out = out_dir / "train_mean.npy"
        std_out = out_dir / "train_std.npy"
        np.save(mean_out, train_mean.astype(np.float32, copy=False))
        np.save(std_out, train_std.astype(np.float32, copy=False))
        print(f"Saved train mean/std: {mean_out}, {std_out}")

    if y is not None:
        lbl_out = out_dir / "labels.npy"
        np.save(lbl_out, y)
        print(f"Saved labels: {lbl_out}")

    # 5. Save summary JSON
    summary = {
        "embeddings_path": str(emb_path),
        "labels_path": str(lbl_path) if lbl_path else None,
        "n_total": int(X.shape[0]),
        "d": int(X.shape[1]),
        "kernel": args.kernel,
        "bandwidth_spec": args.bandwidth,
        "bandwidth_actual": float(A.bandwidth),
        "block_size": int(args.block_size),
        "stoptol": float(args.stoptol),
        "subsample": args.subsample,
        "stratified_subsample": bool(args.stratified_subsample),
        "standardize": bool(args.standardize),
        "indices_path": str(idx_out),
        "train_mean_path": str(out_dir / "train_mean.npy") if args.standardize else None,
        "train_std_path": str(out_dir / "train_std.npy") if args.standardize else None,
        "seed": int(args.seed),
        "runs": all_stats,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    # Printed table at the end, easier on the eyes than scrolling
    print("\n" + "=" * 72)
    print(f"{'k':>6} {'time(s)':>10} {'rel_err':>12} {'queries':>15} "
          f"{'factor_MB':>12}")
    print("-" * 72)
    for s in all_stats:
        print(f"{s['k_actual']:>6} {s['elapsed_sec']:>10.2f} "
              f"{s['rel_trace_error']:>12.4e} {s['kernel_queries']:>15,} "
              f"{s['factor_bytes']/1e6:>12.1f}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Accelerated RPCholesky on embedding datasets."
    )
    p.add_argument("--embeddings", type=str, required=True,
                   help="Path to .npy file of shape (N, d).")
    p.add_argument("--labels", type=str, default=None,
                   help="Optional path to .npy label file of shape (N,).")
    p.add_argument("--subsample", type=int, default=None,
                   help="If set, randomly subsample N down to this many rows.")
    p.add_argument(
        "--stratified-subsample",
        action="store_true",
        help="Use the same stratified subsampling policy as exp1 full HMC.",
    )
    p.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize selected train embeddings before factorization and save mean/std.",
    )
    p.add_argument("--kernel", type=str, default="gaussian",
                   choices=["gaussian", "matern", "laplace"])
    p.add_argument("--bandwidth", type=str, default="approx_median",
                   help='Kernel bandwidth: a float, "median", or "approx_median".')
    p.add_argument("--k-values", type=int, nargs="+",
                   default=[10, 20, 50, 100, 200],
                   help="List of ranks to try.")
    p.add_argument("--block-size", type=int, default=10,
                   help="Block size b for accelerated RPCholesky.")
    p.add_argument("--stoptol", type=float, default=1e-13,
                   help="Stopping tolerance on residual trace fraction.")
    p.add_argument("--output-dir", type=str, default="data/rpchol_out")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    # bandwidth can be a string keyword or a float; parse accordingly.
    try:
        args.bandwidth = float(args.bandwidth)
    except ValueError:
        pass  # leave as string ("median" / "approx_median")

    return args


if __name__ == "__main__":
    main()
