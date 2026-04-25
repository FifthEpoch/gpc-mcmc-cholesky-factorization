"""
Experiment 1 (GPyTorch): SVGP binary GP classification on frozen embeddings.

This version is designed for the real embedding datasets used elsewhere in the
repo, including the Hugging Face export layout such as:

    datasets/pcam-hg/{train,valid,test}/embeddings/projected_512.npy

Key properties:
- Bernoulli likelihood
- Sparse variational GP (SVGP) with inducing points
- RBF kernel + constant mean
- Minibatched training so large datasets such as PCam can run on GPU

Typical usage:
    python experiments/exp1_gpytorch_svgp_gpc.py \
        --dataset pcam \
        --embedding-dir datasets/pcam-hg \
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from predictive_metrics import (
    evaluate_binary_probabilistic_predictions,
    print_metric_table,
    print_posterior_statistics,
)
from my_cholesky.eval_metrics import plot_reliability_diagram


def _import_torch_stack():
    """Import torch + gpytorch with a helpful error message."""
    try:
        torch = importlib.import_module("torch")
        gpytorch = importlib.import_module("gpytorch")
        return torch, gpytorch
    except Exception as err:
        raise ImportError(
            "Could not import torch/gpytorch. Install them first, e.g. "
            "'pip install torch gpytorch'"
        ) from err


def _split_aliases(split: str) -> list[str]:
    split = split.lower()
    if split == "val":
        return ["val", "valid"]
    if split == "valid":
        return ["valid", "val"]
    return [split]


def _load_labels_from_csv(labels_csv: Path) -> np.ndarray:
    """Load binary labels from a CSV file, using the last column as fallback."""
    with labels_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            raise ValueError(f"No rows found in {labels_csv}")

        fieldnames = reader.fieldnames or []
        preferred = ["label", "labels", "y", "target", "tumor"]
        label_col = next((name for name in preferred if name in fieldnames), None)
        if label_col is None:
            label_col = fieldnames[-1]

        labels = np.asarray([float(row[label_col]) for row in rows], dtype=np.float32)
        return labels.reshape(-1)


def load_embeddings(
    emb_dir: Path, dataset: str, split: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings for flat or HG-style layouts.

    Preference order for HG-style layouts:
    1. projected_512.npy
    2. embeddings.npy
    """
    split_names = _split_aliases(split)

    for split_name in split_names:
        flat_feat = emb_dir / f"{dataset}_{split_name}_embeddings.npy"
        flat_lbl = emb_dir / f"{dataset}_{split_name}_labels.npy"
        if flat_feat.exists() and flat_lbl.exists():
            emb = np.asarray(np.load(flat_feat), dtype=np.float32)
            lbl = np.asarray(np.load(flat_lbl), dtype=np.float32).reshape(-1)
            print(f"[load_embeddings] {dataset}:{split} features <- {flat_feat}")
            print(f"[load_embeddings] {dataset}:{split} labels   <- {flat_lbl}")
            return emb, lbl

    roots = [
        emb_dir / f"{dataset}-hg",
        emb_dir / dataset,
        emb_dir,
    ]
    feature_names = ["projected_512.npy", "embeddings.npy"]
    label_names = ["y_embeddings.npy", "labels.npy"]

    for root in roots:
        for split_name in split_names:
            split_dir = root / split_name / "embeddings"
            if not split_dir.exists():
                continue

            feature_path = next(
                (split_dir / name for name in feature_names if (split_dir / name).exists()),
                None,
            )
            if feature_path is None:
                continue

            label_path = next(
                (split_dir / name for name in label_names if (split_dir / name).exists()),
                None,
            )
            labels_csv = root / split_name / "labels.csv"

            emb = np.asarray(np.load(feature_path), dtype=np.float32)
            if label_path is not None:
                lbl = np.asarray(np.load(label_path), dtype=np.float32).reshape(-1)
                print(f"[load_embeddings] {dataset}:{split} features <- {feature_path}")
                print(f"[load_embeddings] {dataset}:{split} labels   <- {label_path}")
                return emb, lbl
            if labels_csv.exists():
                lbl = _load_labels_from_csv(labels_csv)
                print(f"[load_embeddings] {dataset}:{split} features <- {feature_path}")
                print(f"[load_embeddings] {dataset}:{split} labels   <- {labels_csv}")
                return emb, lbl

    raise FileNotFoundError(
        f"Could not find embeddings for dataset={dataset!r}, split={split!r} under {emb_dir}"
    )


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subsample to at most *max_n* rows."""
    if max_n <= 0 or len(y) <= max_n:
        return X, y

    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    pos_ratio = len(pos) / max(len(y), 1)
    n_pos = max(1, int(round(max_n * pos_ratio)))
    n_neg = max(1, max_n - n_pos)

    pos_take = min(n_pos, len(pos))
    neg_take = min(n_neg, len(neg))

    idx = np.concatenate(
        [
            rng.choice(pos, size=pos_take, replace=False),
            rng.choice(neg, size=neg_take, replace=False),
        ]
    )
    rng.shuffle(idx)
    return X[idx], y[idx]


def standardize_splits(
    train_X: np.ndarray,
    val_X: np.ndarray,
    test_X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize each feature using train-set mean/std."""
    mean = train_X.mean(axis=0, keepdims=True)
    std = train_X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    train_X = ((train_X - mean) / std).astype(np.float32, copy=False)
    val_X = ((val_X - mean) / std).astype(np.float32, copy=False)
    test_X = ((test_X - mean) / std).astype(np.float32, copy=False)
    return train_X, val_X, test_X


def make_loader(
    torch,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device,
):
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


class SVGPBinaryClassifier:  # type: ignore[misc]
    """Thin wrapper so gpytorch stays injected instead of global."""

    def __new__(cls, gpytorch, inducing_points):
        class _Model(gpytorch.models.ApproximateGP):
            def __init__(self, inducing_points_):
                variational_distribution = (
                    gpytorch.variational.CholeskyVariationalDistribution(
                        inducing_points_.size(0)
                    )
                )
                variational_strategy = gpytorch.variational.VariationalStrategy(
                    self,
                    inducing_points_,
                    variational_distribution,
                    learn_inducing_locations=True,
                )
                super().__init__(variational_strategy)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        return _Model(inducing_points)


def train_one_epoch(
    torch,
    model,
    likelihood,
    loader,
    optimizer,
    mll,
    device,
) -> float:
    model.train()
    likelihood.train()
    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=device.type == "cuda")
        yb = yb.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad()
        output = model(xb)
        loss = -mll(output, yb)
        loss.backward()
        optimizer.step()

        batch_n = len(yb)
        total_loss += float(loss.item()) * batch_n
        total_n += batch_n

    return total_loss / max(total_n, 1)


def predict_posterior(
    torch,
    model,
    likelihood,
    loader,
    device,
    gpytorch,
) -> dict[str, np.ndarray]:
    model.eval()
    likelihood.eval()

    probs = []
    latent_mean = []
    latent_var = []
    labels = []

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            q_f = model(xb)
            p_y = likelihood(q_f)
            probs.append(p_y.probs.detach().cpu().numpy())
            latent_mean.append(q_f.mean.detach().cpu().numpy())
            latent_var.append(q_f.variance.detach().cpu().numpy())
            labels.append(yb.cpu().numpy())

    return {
        "prob": np.concatenate(probs),
        "latent_mean": np.concatenate(latent_mean),
        "latent_var": np.concatenate(latent_var),
        "y_true": np.concatenate(labels),
    }


def safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp1: GPyTorch SVGP binary GP classification on embeddings."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pcam",
        choices=["pcam", "camelyon17", "embed"],
    )
    parser.add_argument("--embedding-dir", type=str, default="datasets/pcam-hg")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inducing", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--predict-batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--disable-standardize",
        action="store_true",
        help="Disable train-set feature standardization.",
    )
    return parser.parse_args()


def main() -> None:
    torch, gpytorch = _import_torch_stack()
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    emb_dir = Path(args.embedding_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    ds = args.dataset
    train_X, train_y = load_embeddings(emb_dir, ds, "train")
    val_X, val_y = load_embeddings(emb_dir, ds, "val")
    test_X, test_y = load_embeddings(emb_dir, ds, "test")

    if args.max_train_samples > 0:
        train_X, train_y = stratified_subsample(
            train_X, train_y, args.max_train_samples, args.seed
        )
    if args.max_val_samples > 0:
        val_X, val_y = stratified_subsample(
            val_X, val_y, args.max_val_samples, args.seed + 1
        )
    if args.max_test_samples > 0:
        test_X, test_y = stratified_subsample(
            test_X, test_y, args.max_test_samples, args.seed + 2
        )

    if not args.disable_standardize:
        train_X, val_X, test_X = standardize_splits(train_X, val_X, test_X)
        print("Feature standardization: enabled (train mean/std applied to all splits)")
    else:
        print("Feature standardization: disabled")

    input_dim = train_X.shape[1]
    num_inducing = min(args.num_inducing, len(train_X))
    rng = np.random.default_rng(args.seed)
    inducing_idx = rng.choice(len(train_X), size=num_inducing, replace=False)
    inducing_points = torch.from_numpy(train_X[inducing_idx]).float().to(device)

    print(f"Dataset: {ds}")
    print(
        f"  train: {train_X.shape[0]}  val: {val_X.shape[0]}  test: {test_X.shape[0]}"
    )
    print(f"  feature dim: {input_dim}")
    print(f"  num inducing: {num_inducing}")
    print(f"  train batch size: {args.batch_size}")
    print(f"  predict batch size: {args.predict_batch_size}")

    train_loader = make_loader(
        torch, train_X, train_y, args.batch_size, shuffle=True, device=device
    )
    val_loader = make_loader(
        torch, val_X, val_y, args.predict_batch_size, shuffle=False, device=device
    )
    test_loader = make_loader(
        torch, test_X, test_y, args.predict_batch_size, shuffle=False, device=device
    )

    model = SVGPBinaryClassifier(gpytorch, inducing_points).to(device)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=args.learning_rate,
    )
    mll = gpytorch.mlls.VariationalELBO(
        likelihood, model, num_data=train_X.shape[0]
    )

    best_val_auroc = -np.inf
    best_model_state = None
    best_likelihood_state = None
    patience_counter = 0
    epoch_losses: list[float] = []

    experiment_start = datetime.now().astimezone()
    print(
        f"\nTraining for up to {args.epochs} epochs (patience={args.patience})..."
    )
    print(f"Experiment start time: {experiment_start.isoformat(timespec='seconds')}")

    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        mean_loss = train_one_epoch(
            torch, model, likelihood, train_loader, optimizer, mll, device
        )
        epoch_losses.append(mean_loss)

        val_pred = predict_posterior(
            torch, model, likelihood, val_loader, device, gpytorch
        )
        val_auroc = safe_auroc(val_pred["y_true"], val_pred["prob"])

        print(f"  epoch {epoch:3d}  loss={mean_loss:.4f}  val_auroc={val_auroc:.4f}")

        if np.isfinite(val_auroc) and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model_state = deepcopy(model.state_dict())
            best_likelihood_state = deepcopy(likelihood.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    "  Early stopping at epoch "
                    f"{epoch} (best val AUROC: {best_val_auroc:.4f})"
                )
                break

    train_time = time.perf_counter() - train_start
    print(f"Training time: {train_time:.2f}s")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_likelihood_state is not None:
        likelihood.load_state_dict(best_likelihood_state)

    infer_start = time.perf_counter()
    test_pred = predict_posterior(
        torch, model, likelihood, test_loader, device, gpytorch
    )
    infer_time = time.perf_counter() - infer_start
    print(f"Inference time ({len(test_pred['y_true'])} samples): {infer_time:.2f}s")

    print_posterior_statistics(
        latent_mean=test_pred["latent_mean"],
        latent_var=test_pred["latent_var"],
        prob_mean=test_pred["prob"],
        title="GPyTorch SVGP test set posterior statistics",
    )

    metrics = evaluate_binary_probabilistic_predictions(
        y_true=test_pred["y_true"],
        p_pred=test_pred["prob"],
        threshold=0.5,
        n_bins=15,
    )

    experiment_end = datetime.now().astimezone()
    total_runtime = (experiment_end - experiment_start).total_seconds()
    metrics.update(
        {
            "train_time_sec": round(train_time, 3),
            "inference_time_sec": round(infer_time, 3),
            "best_val_auroc": round(float(best_val_auroc), 6),
            "n_train": int(train_X.shape[0]),
            "n_val": int(val_X.shape[0]),
            "n_test": int(test_X.shape[0]),
            "feature_dim": int(input_dim),
            "num_inducing": int(num_inducing),
            "batch_size": int(args.batch_size),
            "predict_batch_size": int(args.predict_batch_size),
            "experiment_start_time": experiment_start.isoformat(timespec="seconds"),
            "experiment_end_time": experiment_end.isoformat(timespec="seconds"),
            "total_runtime_sec": round(total_runtime, 3),
        }
    )

    print_metric_table(metrics, title="GPyTorch SVGP test metrics")

    results_json = out_dir / f"exp1_gpytorch_svgp_{ds}_results.json"
    with results_json.open("w") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"\nSaved: {results_json}")

    results_npz = out_dir / f"exp1_gpytorch_svgp_{ds}_posterior.npz"
    np.savez(
        results_npz,
        epoch_losses=np.asarray(epoch_losses, dtype=np.float32),
        y_test=test_pred["y_true"],
        prob_test=test_pred["prob"],
        latent_mean_test=test_pred["latent_mean"],
        latent_var_test=test_pred["latent_var"],
    )
    print(f"Saved: {results_npz}")

    fig, _ = plot_reliability_diagram(
        test_pred["y_true"],
        test_pred["prob"],
        n_bins=15,
        title=f"Exp1 GPyTorch SVGP Reliability Diagram ({ds})",
    )
    cal_path = out_dir / f"exp1_gpytorch_svgp_{ds}_calibration.png"
    fig.savefig(cal_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {cal_path}")

    fpr, tpr, _ = roc_curve(test_pred["y_true"], test_pred["prob"])
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(fpr, tpr, label=f"AUROC = {metrics['auroc']:.4f}")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title(f"Exp1 GPyTorch SVGP ROC Curve ({ds})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    roc_path = out_dir / f"exp1_gpytorch_svgp_{ds}_roc.png"
    fig2.savefig(roc_path, dpi=160)
    plt.close(fig2)
    print(f"Saved: {roc_path}")

    print(f"Experiment end time:   {experiment_end.isoformat(timespec='seconds')}")
    print(f"Total runtime:         {total_runtime:.2f}s")


if __name__ == "__main__":
    main()
