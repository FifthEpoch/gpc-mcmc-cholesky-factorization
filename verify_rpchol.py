#!/usr/bin/env python3
"""Verify regenerated RPCholesky factor, pivot, and kernel-submatrix files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
CHECK = "✓"
CROSS = "✗"

RPCHOL_DIRS = [
    Path("data/rpchol_smoke"),
    Path("data/rpchol_20k"),
    Path("data/rpchol_50k"),
    Path("data/rpchol_full"),
]


def verify_run(directory: Path, run: dict, n_rows: int) -> list[str]:
    errors: list[str] = []
    k = int(run["k_actual"])

    factor_path = directory / f"factor_k{k}.npy"
    pivots_path = directory / f"pivots_k{k}.npy"
    kpp_path = directory / f"kernel_submatrix_k{k}.npy"

    if not factor_path.exists():
        errors.append(f"missing {factor_path}")
    else:
        factor = np.load(factor_path, mmap_mode="r")
        expected_shape = (n_rows, k)
        if factor.shape != expected_shape:
            errors.append(f"{factor_path} shape {factor.shape}, expected {expected_shape}")

    if not pivots_path.exists():
        errors.append(f"missing {pivots_path}")
    else:
        pivots = np.load(pivots_path, mmap_mode="r")
        if pivots.shape != (k,):
            errors.append(f"{pivots_path} shape {pivots.shape}, expected ({k},)")

    if not kpp_path.exists():
        errors.append(f"missing {kpp_path}")
    else:
        kpp = np.load(kpp_path, mmap_mode="r")
        expected_shape = (k, k)
        if kpp.shape != expected_shape:
            errors.append(f"{kpp_path} shape {kpp.shape}, expected {expected_shape}")
        if kpp.dtype != np.float64:
            errors.append(f"{kpp_path} dtype {kpp.dtype}, expected float64")

    return errors


def verify_directory(directory: Path) -> list[str]:
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return [f"missing {summary_path}"]

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    n_rows = int(summary["n_total"])
    runs = summary.get("runs", [])
    if not runs:
        return ["summary.json contains no runs"]

    errors: list[str] = []
    for run in runs:
        errors.extend(verify_run(directory, run, n_rows))
        k = int(run["k_actual"])
        expected_path = str(directory / f"kernel_submatrix_k{k}.npy")
        actual_path = run.get("kernel_submatrix_path")
        if actual_path is None:
            errors.append(f"summary run k={k} missing kernel_submatrix_path")
        elif Path(actual_path) != Path(expected_path):
            errors.append(
                f"summary run k={k} kernel_submatrix_path {actual_path}, expected {expected_path}"
            )

    return errors


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ok = True
    for directory in RPCHOL_DIRS:
        errors = verify_directory(directory)
        if errors:
            ok = False
            print(f"{RED}{CROSS}{RESET} {directory}")
            for error in errors:
                print(f"    - {error}")
        else:
            print(f"{GREEN}{CHECK}{RESET} {directory}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
