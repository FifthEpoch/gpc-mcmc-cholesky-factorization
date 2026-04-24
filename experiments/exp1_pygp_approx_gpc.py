"""
Experiment: pyGPs approximate inference variants for binary GP classification.

Requested method tags:
- LA
- EP

This script mirrors the synthetic two-blob setup from exp1b_emcee_gpc.
It runs LA and EP, extracts latent predictive Gaussian and predictive probability,
and plots them in a single figure with six panels.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from predictive_metrics import (
    evaluate_binary_probabilistic_predictions,
    print_metric_table,
)


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


def _configure_inference(model: Any, pygp_module: Any, method_code: str) -> None:
    """Only keep LA and EP."""
    if method_code == "LA":
        model.inffunc = pygp_module.inf.Laplace()
        return

    if method_code == "EP":
        if hasattr(pygp_module.inf, "EP"):
            model.inffunc = pygp_module.inf.EP()
            return
        raise RuntimeError("EP is not present in this pyGPs.inf module.")

    raise RuntimeError(f"Unsupported method code: {method_code}")


def fit_predict_method(
    pygp_module: Any,
    X: np.ndarray,
    y01: np.ndarray,
    X_grid: np.ndarray,
    method_code: str,
) -> dict[str, Any]:
    """
    Fit LA or EP and return:
    - latent predictive mean fm
    - latent predictive variance fs2
    - predictive class probability ym
    - log predictive probability lp
    """
    y_pm1 = _to_pm1_labels(y01)

    model = pygp_module.GPC()
    _configure_inference(model, pygp_module, method_code)

    t0 = time.perf_counter()
    model.getPosterior(X, y_pm1)

    # pyGPs predict convention:
    # ym  : predictive mean of observed output / class probability-like output
    # ys2 : predictive variance of observed output
    # fm  : latent predictive mean
    # fs2 : latent predictive variance
    # lp  : log predictive probability
    ym, ys2, fm, fs2, lp = model.predict(X_grid)
    elapsed = time.perf_counter() - t0

    ym = np.asarray(ym, dtype=float).reshape(-1)
    ys2 = np.asarray(ys2, dtype=float).reshape(-1)
    fm = np.asarray(fm, dtype=float).reshape(-1)
    fs2 = np.asarray(fs2, dtype=float).reshape(-1)
    lp = np.asarray(lp, dtype=float).reshape(-1)

    # Safety clipping for probabilities if needed
    prob = np.clip(ym, 0.0, 1.0)

    return {
        "fit_predict_time": float(elapsed),
        "prob": prob,
        "y_var": ys2,
        "latent_mean": fm,
        "latent_var": np.maximum(fs2, 0.0),
        "log_pred_prob": lp,
    }


def _plot_surface(
    ax: Any,
    xx: np.ndarray,
    yy: np.ndarray,
    values: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    mappable = ax.contourf(
        xx,
        yy,
        values,
        levels=40,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.9,
    )
    ax.scatter(
        X[:, 0], X[:, 1],
        c=y, cmap="coolwarm", s=8, alpha=0.45, vmin=0, vmax=1
    )
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    return mappable


def main() -> None:
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    pygp = _import_pygp()

    X, y = make_fake_blobs(seed=42)
    X_test_labeled, y_test = make_fake_blobs(seed=123)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 120),
        np.linspace(y_min, y_max, 120),
    )
    X_grid = np.column_stack([xx.ravel(), yy.ravel()])

    methods = ["LA", "EP"]
    results: dict[str, dict[str, Any]] = {}

    print("Running pyGPs methods:", ", ".join(methods))
    for method in methods:
        results[method] = fit_predict_method(pygp, X, y, X_grid, method)
        test_results = fit_predict_method(pygp, X, y, X_test_labeled, method)
        results[method]["prob_test"] = test_results["prob"]
        results[method]["latent_mean_test"] = test_results["latent_mean"]
        results[method]["latent_var_test"] = test_results["latent_var"]
        results[method]["test_metrics"] = evaluate_binary_probabilistic_predictions(
            y_true=y_test,
            p_pred=results[method]["prob_test"],
            threshold=0.5,
            n_bins=15,
        )
        print(f"  {method}: fit+predict={results[method]['fit_predict_time']:.3f}s")
        print_metric_table(results[method]["test_metrics"], title=f"pyGPs {method} test metrics")

    # ------------------------------------------------------------
    # Single figure with 6 panels: 3 rows (latent mean, latent std, prob) x 2 columns (LA, EP)
    # ------------------------------------------------------------
    fig, axes = plt.subplots(
        3, 2, figsize=(10, 12), sharex=True, sharey=True, constrained_layout=True
    )

    # Row 0: latent predictive mean μ_*(x)
    for col, method in enumerate(methods):
        ax = axes[0, col]
        z = results[method]["latent_mean"].reshape(xx.shape)
        mappable = _plot_surface(
            ax, xx, yy, z, X, y,
            title=f"{method} latent mean",
            cmap="coolwarm"
        )
        if col == 1:
            fig.colorbar(mappable, ax=ax, shrink=0.9, label="Latent predictive mean $\\mu_*(x)$")

    # Row 1: latent predictive std sqrt(σ_*^2(x))
    for col, method in enumerate(methods):
        ax = axes[1, col]
        z = np.sqrt(results[method]["latent_var"]).reshape(xx.shape)
        mappable = _plot_surface(
            ax, xx, yy, z, X, y,
            title=f"{method} latent std",
            cmap="magma"
        )
        if col == 1:
            fig.colorbar(mappable, ax=ax, shrink=0.9, label="Latent predictive std $\\sqrt{\\sigma_*^2(x)}$")

    # Row 2: predictive class probability P(y=1 | x, D)
    for col, method in enumerate(methods):
        ax = axes[2, col]
        z = results[method]["prob"].reshape(xx.shape)
        mappable = _plot_surface(
            ax, xx, yy, z, X, y,
            title=f"{method} predictive probability",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0
        )
        if col == 1:
            fig.colorbar(mappable, ax=ax, shrink=0.9, label="Predictive probability $P(y=1\\mid x,D)$")

    plt.savefig(os.path.join(data_dir, "exp1_pygp_la_ep_all.png"), dpi=170)
    plt.close()

    np.save(
        os.path.join(data_dir, "exp1_pygp_la_ep_results.npy"),
        {
            "methods": methods,
            "fit_predict_time": {m: results[m]["fit_predict_time"] for m in methods},
            "results": results,
            "y_test": y_test,
        },
        allow_pickle=True,
    )

    print("Saved:")
    print("- data/exp1_pygp_la_ep_all.png")
    print("- data/exp1_pygp_la_ep_results.npy")


if __name__ == "__main__":
    main()
