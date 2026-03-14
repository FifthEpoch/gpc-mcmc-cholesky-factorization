"""
Basic and block RPCholesky variants for Experiment 0.

This module provides:
- cholesky_helper:    sequential RPCholesky core
- block_cholesky_helper: block RPCholesky core (regularized strategy)
- simple_rpcholesky:  basic RPCholesky (thin wrapper)
- block_rpcholesky:   block RPCholesky (thin wrapper)

These are adapted from the Randomly-Pivoted-Cholesky reference
implementation, restricted to the randomized ('rp') variants and the
regularized block strategy, which is sufficient for Exp0.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .lra import PSDLowRank
from .matrix import AbstractPSDMatrix, PSDMatrix


def cholesky_helper(A, k: int, alg: str, stoptol: float = 0.0) -> PSDLowRank:
    """
    Basic (sequential) partial Cholesky / RPCholesky helper.

    Parameters
    ----------
    A : array-like or AbstractPSDMatrix
        PSD matrix or matrix-like object.
    k : int
        Target rank (maximum number of pivots).
    alg : {"rp"}
        Pivoting rule. For Exp0 we only use 'rp' (randomized).
    stoptol : float, optional
        Relative trace tolerance; if residual trace falls below
        stoptol * trace(A), stop early. If 0, run full k steps.
    """
    if not isinstance(A, AbstractPSDMatrix):
        A = PSDMatrix(A)

    n = A.shape[0]
    diags = A.diag()
    orig_trace = float(np.sum(diags))
    if stoptol is None:
        stoptol = 0.0

    G = np.zeros((k, n), dtype=float)
    rows = np.zeros((k, n), dtype=float)
    rng = np.random.default_rng()

    arr_idx: List[int] = []

    for i in range(k):
        if alg == "rp":
            idx = rng.choice(range(n), p=diags / np.sum(diags))
        else:
            raise RuntimeError(f"Algorithm '{alg}' not supported in cholesky_helper")

        arr_idx.append(int(idx))
        rows[i, :] = A[idx, :]
        G[i, :] = (rows[i, :] - G[:i, idx].T @ G[:i, :]) / np.sqrt(diags[idx])
        diags -= G[i, :] ** 2
        diags = diags.clip(min=0.0)

        if stoptol > 0.0 and float(np.sum(diags)) <= stoptol * orig_trace:
            G = G[:i, :]
            rows = rows[:i, :]
            break

    return PSDLowRank(G, idx=arr_idx, rows=rows)


def block_cholesky_helper(
    A,
    k: int,
    b: int,
    alg: str,
    stoptol: float = 1e-14,
    strategy: str = "regularize",
) -> PSDLowRank:
    """
    Block RPCholesky helper (regularized strategy only).

    Parameters
    ----------
    A : array-like or AbstractPSDMatrix
        PSD matrix or matrix-like object.
    k : int
        Target rank (maximum number of pivots).
    b : int
        Block size.
    alg : {"rp"}
        Pivoting rule. For Exp0 we only use 'rp' (randomized).
    stoptol : float, optional
        Relative trace tolerance; if residual trace falls below
        stoptol * trace(A), stop early.
    strategy : {"regularize"}, optional
        Block strategy. For Exp0 we only implement the regularized
        strategy from the reference code.
    """
    if not isinstance(A, AbstractPSDMatrix):
        A = PSDMatrix(A)

    diags = A.diag()
    n = A.shape[0]
    orig_trace = float(np.sum(diags))
    scale = 2.0 * float(np.max(diags))
    if stoptol is None:
        stoptol = 1e-14

    G = np.zeros((k, n), dtype=float)
    rows = np.zeros((k, n), dtype=float)

    rng = np.random.default_rng()

    arr_idx: List[int] = []

    counter = 0
    while counter < k:
        block_size = min(k - counter, b)

        if alg == "rp":
            idx = rng.choice(
                range(n),
                size=2 * block_size,
                p=diags / np.sum(diags),
                replace=True,
            )
            idx = np.unique(idx)[:block_size]
            block_size = len(idx)
        else:
            raise RuntimeError(f"Algorithm '{alg}' not supported in block_cholesky_helper")

        if strategy not in {"regularize", "regularized"}:
            raise ValueError(
                f"Strategy '{strategy}' not supported in block_cholesky_helper "
                "(only 'regularize' is implemented for Exp0)."
            )

        # Regularized block RPCholesky
        arr_idx.extend(idx.tolist())
        rows[counter : counter + block_size, :] = A[idx, :]
        G[counter : counter + block_size, :] = (
            rows[counter : counter + block_size, :]
            - G[0:counter, idx].T @ G[0:counter, :]
        )
        C = G[counter : counter + block_size, idx]

        # Regularize the core C for numeric stability
        L = np.linalg.cholesky(
            C + np.finfo(float).eps * b * scale * np.identity(block_size)
        )
        G[counter : counter + block_size, :] = np.linalg.solve(
            L, G[counter : counter + block_size, :]
        )

        diags -= np.sum(G[counter : counter + block_size, :] ** 2, axis=0)
        diags = diags.clip(min=0.0)

        counter += block_size

        if stoptol > 0.0 and float(np.sum(diags)) <= stoptol * orig_trace:
            G = G[:counter, :]
            rows = rows[:counter, :]
            break

    return PSDLowRank(G, idx=arr_idx, rows=rows)


def simple_rpcholesky(A, k: int, **kwargs) -> PSDLowRank:
    """
    Basic randomized RPCholesky (sequential).
    """
    return cholesky_helper(A, k, alg="rp", **kwargs)


def block_rpcholesky(A, k: int, b: int = 100, **kwargs) -> PSDLowRank:
    """
    Block RPCholesky using randomized block selection and regularized core.
    """
    return block_cholesky_helper(A, k, b, alg="rp", **kwargs)

