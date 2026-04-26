"""
Experiment: pyGPs approximate inference (LA / EP) on real embedding datasets.

This is the real-data companion to exp1_pygp_approx_gpc.py, which remains the
toy two-blob experiment. This file loads precomputed embeddings, optionally
subsamples them to a manageable size, fits pyGPs binary GP classifiers with
Laplace approximation and Expectation Propagation, and evaluates on a held-out
split.

Because pyGPs LA/EP uses dense kernel algebra, it is not practical to fit on
the full PCam training split (262,144 samples). This file is intended only for
small sanity-check runs, with a hard cap of 10,000 train and 10,000 test
samples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from predictive_metrics import (  # noqa: E402
    evaluate_binary_probabilistic_predictions,
    print_metric_table,
)


SANITY_CHECK_MAX_TRAIN_SAMPLES = 10_000
SANITY_CHECK_MAX_TEST_SAMPLES = 10_000


def _import_pygp() -> Any:
    """Import pyGPs with a helpful installation error."""
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
    if method_code == "LA":
        model.inffunc = pygp_module.inf.Laplace()
        return

    if method_code == "EP":
        if hasattr(pygp_module.inf, "EP"):
            model.inffunc = pygp_module.inf.EP()
            return
        raise RuntimeError("EP is not present in this pyGPs.inf module.")

    raise RuntimeError(f"Unsupported method code: {method_code}")


def fit_method(
    pygp_module: Any,
    X: np.ndarray,
    y01: np.ndarray,
    method_code: str,
) -> tuple[Any, float]:
    """Fit one pyGPs classifier and return the model plus fit time."""
    y_pm1 = _to_pm1_labels(y01)

    model = pygp_module.GPC()
    _configure_inference(model, pygp_module, method_code)

    t0 = time.perf_counter()
    model.getPosterior(X, y_pm1)
    elapsed = time.perf_counter() - t0
    return model, float(elapsed)


def predict_with_model(model: Any, X_pred: np.ndarray) -> dict[str, Any]:
    """Predict from a fitted pyGPs model."""
    ym, ys2, fm, fs2, lp = model.predict(X_pred)

    ym = np.asarray(ym, dtype=float).reshape(-1)
    ys2 = np.asarray(ys2, dtype=float).reshape(-1)
    fm = np.asarray(fm, dtype=float).reshape(-1)
    fs2 = np.asarray(fs2, dtype=float).reshape(-1)
    lp = np.asarray(lp, dtype=float).reshape(-1)

    prob = np.clip(ym, 0.0, 1.0)
    return {
        "prob": prob,
        "y_var": ys2,
        "latent_mean": fm,
        "latent_var": np.maximum(fs2, 0.0),
        "log_pred_prob": lp,
    }


def _split_name_variants(split: str) -> list[str]:
    variants = [split]
    if split == "val":
        variants.append("valid")
    if split == "valid":
        variants.append("val")
    return variants


def load_embeddings(
    emb_dir: Path, dataset: str, split: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings from either:
    - flat files: <dataset>_<split>_embeddings.npy
    - HG layout : <split>/embeddings/projected_512.npy
    """
    for split_name in _split_name_variants(split):
        flat_emb = emb_dir / f"{dataset}_{split_name}_embeddings.npy"
        flat_lbl = emb_dir / f"{dataset}_{split_name}_labels.npy"
        if flat_emb.exists() and flat_lbl.exists():
            print(f"[load_embeddings] {dataset}:{split} features <- {flat_emb}")
            print(f"[load_embeddings] {dataset}:{split} labels   <- {flat_lbl}")
            X = np.load(flat_emb).astype(np.float64, copy=False)
            y = np.load(flat_lbl).astype(np.int64, copy=False).squeeze()
            return X, y

    for split_name in _split_name_variants(split):
        split_emb_dir = emb_dir / split_name / "embeddings"
        projected_emb = split_emb_dir / "projected_512.npy"
        raw_emb = split_emb_dir / "embeddings.npy"
        lbl_path = split_emb_dir / "y_embeddings.npy"

        feature_path = projected_emb if projected_emb.exists() else raw_emb
        if feature_path.exists() and lbl_path.exists():
            print(f"[load_embeddings] {dataset}:{split} features <- {feature_path}")
            print(f"[load_embeddings] {dataset}:{split} labels   <- {lbl_path}")
            X = np.load(feature_path).astype(np.float64, copy=False)
            y = np.load(lbl_path).astype(np.int64, copy=False).squeeze()
            return X, y

    raise FileNotFoundError(
        f"Could not find embeddings for dataset={dataset}, split={split} under {emb_dir}."
    )


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, max_n: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample rows while roughly preserving class balance."""
    if max_n is None or len(y) <= max_n:
        return X, y

    rng = np.random.default_rng(seed)
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
    return X[idx], y[idx]


def standardize_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize features using train-set moments only."""
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def prepare_data(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    emb_dir = Path(args.embedding_dir)

    X_train, y_train = load_embeddings(emb_dir, args.dataset, args.train_split)
    X_test, y_test = load_embeddings(emb_dir, args.dataset, args.test_split)

    print(f"Dataset: {args.dataset}")
    print(
        f"  original train: {X_train.shape[0]}  "
        f"{args.test_split}: {X_test.shape[0]}  "
        f"feature dim: {X_train.shape[1]}"
    )

    X_train, y_train = stratified_subsample(
        X_train, y_train, args.max_train_samples, args.seed
    )
    X_test, y_test = stratified_subsample(
        X_test, y_test, args.max_test_samples, args.seed + 1
    )

    if args.standardize:
        X_train, X_test = standardize_features(X_train, X_test)

    print(
        f"  used train: {X_train.shape[0]}  "
        f"used {args.test_split}: {X_test.shape[0]}  "
        f"feature dim: {X_train.shape[1]}"
    )

    return X_train, y_train, X_test, y_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp1: pyGPs approximate GP classification on embedding datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pcam",
        choices=["pcam", "camelyon17", "embed"],
    )
    parser.add_argument("--embedding-dir", type=str, default="datasets/pcam-hg")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=10000,
        help="Dense pyGPs sanity-check cap for train samples (max 10000).",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=10000,
        help="Dense pyGPs sanity-check cap for test samples (max 10000).",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_false",
        dest="standardize",
        help="Disable train-set feature standardization.",
    )
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(standardize=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pygp = _import_pygp()
    np.random.seed(args.seed)

    if args.max_train_samples is None or args.max_train_samples > SANITY_CHECK_MAX_TRAIN_SAMPLES:
        raise ValueError(
            "exp1_pygp_approx_gpc_embeddings.py is only meant for small sanity checks. "
            f"Set --max-train-samples to at most {SANITY_CHECK_MAX_TRAIN_SAMPLES}."
        )
    if args.max_test_samples is not None and args.max_test_samples > SANITY_CHECK_MAX_TEST_SAMPLES:
        raise ValueError(
            "exp1_pygp_approx_gpc_embeddings.py is only meant for small sanity checks. "
            f"Set --max-test-samples to at most {SANITY_CHECK_MAX_TEST_SAMPLES}."
        )

    X_train, y_train, X_test, y_test = prepare_data(args)
    methods = ["LA", "EP"]

    if args.max_train_samples is None and X_train.shape[0] > 20_000:
        raise ValueError(
            "pyGPs LA/EP is a dense GP method and is not suitable for this many "
            "training points without subsampling. Pass --max-train-samples."
        )

    all_results: dict[str, dict[str, Any]] = {}
    all_metrics: dict[str, dict[str, Any]] = {}

    print("Running pyGPs methods:", ", ".join(methods))
    for method in methods:
        fit_model, fit_time = fit_method(pygp, X_train, y_train, method)
        pred_test = predict_with_model(fit_model, X_test)
        test_metrics = evaluate_binary_probabilistic_predictions(
            y_true=y_test,
            p_pred=pred_test["prob"],
            threshold=0.5,
            n_bins=15,
        )
        test_metrics["fit_time_sec"] = round(fit_time, 3)
        test_metrics["n_train"] = int(X_train.shape[0])
        test_metrics["n_test"] = int(X_test.shape[0])
        test_metrics["feature_dim"] = int(X_train.shape[1])

        all_results[method] = {
            "fit_time": fit_time,
            **pred_test,
        }
        all_metrics[method] = test_metrics

        print(f"  {method}: fit={fit_time:.3f}s")
        print_metric_table(test_metrics, title=f"pyGPs {method} test metrics")

    prefix = f"exp1_pygp_{args.dataset}_{args.train_split}_to_{args.test_split}"
    metrics_path = out_dir / f"{prefix}_metrics.json"
    results_path = out_dir / f"{prefix}_results.npy"

    with open(metrics_path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "embedding_dir": args.embedding_dir,
                "train_split": args.train_split,
                "test_split": args.test_split,
                "max_train_samples": args.max_train_samples,
                "max_test_samples": args.max_test_samples,
                "standardize": args.standardize,
                "methods": all_metrics,
            },
            f,
            indent=2,
        )

    np.save(
        results_path,
        {
            "dataset": args.dataset,
            "train_split": args.train_split,
            "test_split": args.test_split,
            "methods": all_results,
            "y_test": y_test,
        },
        allow_pickle=True,
    )

    print("Saved:")
    print(f"- {metrics_path}")
    print(f"- {results_path}")
    print("Skipped 2D contour plot because embedding features are not 2-dimensional.")


if __name__ == "__main__":
    main()
