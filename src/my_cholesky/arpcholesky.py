"""
Accelerated randomly pivoted Cholesky (Algorithm 2.1 and 2.2).

This module provides:
- `rejection_cholesky` (block-level rejection sampling, Algorithm 2.1)
- `accelerated_rpcholesky` (matrix-level accelerated RPCholesky, Algorithm 2.2)

The implementation is adapted from the Randomly-Pivoted-Cholesky
reference code, with a minimal dependency footprint.
"""

from __future__ import annotations

from time import time
from typing import Tuple

import numpy as np

from .lra import PSDLowRank
from .matrix import AbstractPSDMatrix, PSDMatrix


def rejection_cholesky(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rejection-sampling partial Cholesky on a small PSD block H.

    Parameters
    ----------
    H : np.ndarray, shape (b, b)
        Symmetric positive semidefinite block.

    Returns
    -------
    L : np.ndarray, shape (r, r)
        Cholesky-like factor for the accepted pivots (r <= b).
    idx : np.ndarray, shape (r,)
        Indices of accepted pivots within the block.
    """
    b = H.shape[0]
    if H.shape[0] != H.shape[1]:
        raise RuntimeError("rejection_cholesky requires a square matrix")
    if np.trace(H) <= 0:
        raise RuntimeError("rejection_cholesky requires a strictly positive trace")

    u = np.array([H[j, j] for j in range(b)])

    idx: list[int] = []
    L = np.zeros((b, b), dtype=H.dtype)

    for j in range(b):
        if np.random.rand() * u[j] < H[j, j]:
            idx.append(j)
            L[j:, j] = H[j:, j] / np.sqrt(H[j, j])
            H[(j + 1) :, (j + 1) :] -= np.outer(L[(j + 1) :, j], L[(j + 1) :, j])

    idx = np.array(idx, dtype=int)
    L = L[np.ix_(idx, idx)]
    return L, idx


def accelerated_rpcholesky(
    A,
    k: int,
    b: int | str = "auto",
    stoptol: float | None = 1e-13,
    verbose: bool = False,
) -> PSDLowRank:
    """
    Accelerated randomly pivoted Cholesky factorization (Algorithm 2.2).

    Computes a rank-k Nyström approximation to a PSD matrix A using
    block proposals and rejection sampling to match the distribution
    of standard RPCholesky while reducing kernel evaluations.

    Parameters
    ----------
    A : array-like or AbstractPSDMatrix
        Underlying positive semidefinite matrix or matrix-like object.
    k : int
        Target rank (maximum number of pivots).
    b : int or "auto", optional
        Block size for candidate pivots. If "auto", an adaptive
        heuristic is used to tune b during the run.
    stoptol : float or None, optional
        Relative trace tolerance; if the residual trace drops below
        `stoptol * trace(A)` the algorithm stops early. If None,
        no early stopping based on trace is applied.
    verbose : bool, optional
        If True, prints per-block acceptance statistics.

    Returns
    -------
    PSDLowRank
        Low-rank Nyström approximation to A.
    """
    if not isinstance(A, AbstractPSDMatrix):
        A = PSDMatrix(A)

    diags = A.diag()
    n = A.shape[0]
    orig_trace = float(np.sum(diags))

    if stoptol is None:
        stoptol = 1e-13

    if b == "auto":
        b = int(np.ceil(k / 10))
        auto_b = True
    else:
        b = int(b)
        auto_b = False

    G = np.zeros((k, n), dtype=float)
    rows = np.zeros((k, n), dtype=float)

    rng = np.random.default_rng()
    arr_idx = np.zeros(k, dtype=int)

    counter = 0
    while counter < k:
        idx = rng.choice(range(n), size=b, p=diags / np.sum(diags), replace=True)

        if auto_b:
            start = time()

        H = A[idx, idx] - G[0:counter, idx].T @ G[0:counter, idx]
        L, accepted = rejection_cholesky(H)
        num_sel = len(accepted)

        if num_sel > k - counter:
            num_sel = k - counter
            accepted = accepted[:num_sel]
            L = L[:num_sel, :num_sel]

        idx = idx[accepted]

        if auto_b:
            rejection_time = time() - start
            start = time()

        arr_idx[counter : counter + num_sel] = idx
        rows[counter : counter + num_sel, :] = A[idx, :]
        G[counter : counter + num_sel, :] = (
            rows[counter : counter + num_sel, :]
            - G[0:counter, idx].T @ G[0:counter, :]
        )
        G[counter : counter + num_sel, :] = np.linalg.solve(
            L, G[counter : counter + num_sel, :]
        )

        diags -= np.sum(G[counter : counter + num_sel, :] ** 2, axis=0)
        diags = diags.clip(min=0.0)

        if auto_b:
            process_time = time() - start
            # Adapt b so that rejection_time ≈ process_time / 4
            target = int(np.ceil(b * process_time / (4.0 * rejection_time)))
            b = max(
                [
                    min([target, int(np.ceil(1.5 * b)), int(np.ceil(k / 3.0))]),
                    int(np.ceil(b / 3.0)),
                    10,
                ]
            )

        counter += num_sel

        if stoptol > 0 and float(np.sum(diags)) <= stoptol * orig_trace:
            G = G[:counter, :]
            rows = rows[:counter, :]
            break

        if verbose:
            print(f"Accepted {num_sel} / {b}")

    return PSDLowRank(G, idx=arr_idx[:counter], rows=rows[:counter, :])


def arpcholesky(
    A,
    k: int,
    b: int | str = "auto",
    stoptol: float | None = 1e-13,
    verbose: bool = False,
) -> PSDLowRank:
    """
    Convenience wrapper for accelerated_rpcholesky.
    """
    return accelerated_rpcholesky(A, k, b=b, stoptol=stoptol, verbose=verbose)

