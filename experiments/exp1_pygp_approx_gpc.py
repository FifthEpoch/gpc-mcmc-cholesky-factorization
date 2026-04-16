"""
Experiment: pyGPs approximate inference variants for binary GP classification.

Requested method tags:
- LA
- EP
- KL
- VB
- FV

This script mirrors the synthetic two-blob setup from exp1b_emcee_gpc.
It runs whichever requested variants are actually supported by the installed
pyGPs build and plots posterior probability surfaces for available variants.
"""

from __future__ import annotations

import os
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_fake_blobs(seed: int = 42):
    """Generate N=2000 two-blob binary data in R^2."""
    rng = np.random.default_rng(seed)
    n_per_class = 1000
    cov = 0.5 * np.eye(2)
    x0 = rng.multivariate_normal(mean=[-1.0, 0.0], cov=cov, size=n_per_class)
    x1 = rng.multivariate_normal(mean=[1.0, 0.0], cov=cov, size=n_per_class)
    X = np.vstack([x0, x1])
    y = np.concatenate(
        [np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)]
    )
    return X, y


def _import_pygp() -> Any:
    """Import pyGPs (the Python-3-compatible package used in this repo)."""
    try:
        import pyGPs  # type: ignore

        return pyGPs
    except Exception as err:  # pragma: no cover - import failure path
        raise ImportError(
            "Could not import pyGPs. Install it first in your environment with: "
            'pip install pyGPs'
        ) from err


def _to_pm1_labels(y01: np.ndarray) -> np.ndarray:
    """pyGPs classification expects labels in {+1, -1}."""
    return np.where(y01 > 0, 1.0, -1.0)


def _method_to_inference_name(method_code: str) -> str:
    """Map requested tags to pyGPs inference names."""
    method_map = {
        "LA": "Laplace",
        "EP": "EP",
        "KL": "KL",
        "VB": "VB",
        "FV": "FV",
    }
    return method_map[method_code]


def _configure_inference(model: Any, pygp_module: Any, method_code: str) -> None:
    """Configure pyGPs inference robustly across known API quirks."""
    if method_code == "LA":
        model.inffunc = pygp_module.inf.Laplace()
        return

    if method_code == "EP":
        if hasattr(pygp_module.inf, "EP"):
            model.inffunc = pygp_module.inf.EP()
            return
        raise RuntimeError("EP is not present in this pyGPs.inf module.")

    if method_code in {"KL", "VB", "FV"}:
        raise RuntimeError(
            f"{method_code} is not implemented in this pyGPs GPC build; "
            "available inference backends here are EP and Laplace."
        )

    raise RuntimeError(f"Unsupported method code: {method_code}")


def fit_predict_method(
    pygp_module: Any,
    X: np.ndarray,
    y01: np.ndarray,
    X_grid: np.ndarray,
    method_code: str,
) -> tuple[np.ndarray, float]:
    """Fit one requested method if supported and return p(y=1) on the grid."""
    y_pm1 = _to_pm1_labels(y01)

    model = pygp_module.GPC()
    _configure_inference(model, pygp_module, method_code)
    t0 = time.perf_counter()
    model.getPosterior(X, y_pm1)
    ym, _, _, _, _ = model.predict(X_grid)
    elapsed = time.perf_counter() - t0

    # pyGPs returns predictive mean in [-1, 1]; map to probability-like [0, 1].
    p = 0.5 * (np.asarray(ym, dtype=float).reshape(-1) + 1.0)
    return np.clip(p, 0.0, 1.0), float(elapsed)


def main() -> None:
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    pygp = _import_pygp()

    X, y = make_fake_blobs(seed=42)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120), np.linspace(y_min, y_max, 120))
    X_grid = np.column_stack([xx.ravel(), yy.ravel()])

    requested_methods = ["LA", "EP", "KL", "VB", "FV"]
    available_results: dict[str, dict[str, Any]] = {}
    unavailable: dict[str, str] = {}

    print("Requested pyGPs methods:", ", ".join(requested_methods))
    for method in requested_methods:
        try:
            probs_grid, elapsed = fit_predict_method(pygp, X, y, X_grid, method)
            available_results[method] = {
                "probs_grid": probs_grid,
                "fit_predict_time": elapsed,
            }
            print(f"  {method}: available, fit+predict={elapsed:.3f}s")
        except Exception as err:
            unavailable[method] = str(err)
            print(f"  {method}: unavailable ({err})")

    available_methods = list(available_results.keys())
    if not available_methods:
        raise RuntimeError(
            "No requested methods are available in this pyGPs build. "
            "Try another GP package or Python version."
        )

    n_panels = len(available_methods)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 4.2 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_arr = np.atleast_1d(axes).reshape(-1)

    posterior_mappable = None
    for idx, method in enumerate(available_methods):
        ax = axes_arr[idx]
        p_grid = available_results[method]["probs_grid"].reshape(xx.shape)
        posterior_mappable = ax.contourf(
            xx,
            yy,
            p_grid,
            levels=40,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            alpha=0.9,
        )
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=8, alpha=0.45, vmin=0, vmax=1)
        ax.set_title(f"{method} (t={available_results[method]['fit_predict_time']:.2f}s)")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    for ax in axes_arr[n_panels:]:
        ax.set_axis_off()

    fig.colorbar(
        posterior_mappable,
        ax=axes_arr[:n_panels],
        shrink=0.9,
        label="Posterior probability P(y=1)",
    )

    title = "pyGPs posterior surfaces for available requested variants"
    if unavailable:
        title += "\nUnavailable: " + ", ".join(unavailable.keys())
    fig.suptitle(title, fontsize=12)

    out_png = os.path.join(data_dir, "exp1_pygp_posterior_methods.png")
    plt.savefig(out_png, dpi=170)
    plt.close()

    np.save(
        os.path.join(data_dir, "exp1_pygp_results.npy"),
        {
            "requested_methods": requested_methods,
            "available_methods": available_methods,
            "available_results": {
                m: {
                    "fit_predict_time": available_results[m]["fit_predict_time"],
                }
                for m in available_methods
            },
            "unavailable": unavailable,
        },
        allow_pickle=True,
    )

    print("Saved:")
    print("- data/exp1_pygp_posterior_methods.png")
    print("- data/exp1_pygp_results.npy")


if __name__ == "__main__":
    main()
