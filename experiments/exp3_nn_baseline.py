"""
Experiment 3: Deterministic neural network baseline.

Trains a 2-layer MLP on frozen embeddings extracted by extract_embeddings.py
and evaluates AUROC, AUPRC, ECE, Brier score, sensitivity, and FNR.

Usage:
    python experiments/exp3_nn_baseline.py \
        --dataset pcam \
        --embedding-dir data/embeddings \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from my_cholesky.eval_metrics import (
    compute_all_metrics,
    compute_auroc,
    plot_reliability_diagram,
)

HG_DATASET_ROOTS = {
    "pcam": "pcam-hg",
    "camelyon17": "camelyon17-hg",
    "embed": "embed",
}


class MLPClassifier(nn.Module):
    """2-layer MLP head for binary classification on frozen embeddings."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_embeddings(
    emb_dir: Path, dataset: str, split: str
) -> tuple[np.ndarray, np.ndarray]:
    flat_emb = emb_dir / f"{dataset}_{split}_embeddings.npy"
    flat_lbl = emb_dir / f"{dataset}_{split}_labels.npy"
    if flat_emb.exists() and flat_lbl.exists():
        emb = np.load(flat_emb)
        lbl = np.asarray(np.load(flat_lbl)).reshape(-1)
        if emb.shape[0] != lbl.shape[0]:
            raise ValueError(
                f"Mismatched rows for {flat_emb.name} ({emb.shape[0]}) and "
                f"{flat_lbl.name} ({lbl.shape[0]})."
            )
        return emb, lbl

    hg_root_name = HG_DATASET_ROOTS.get(dataset, dataset)
    split_dir_name = {"train": "train", "val": "valid", "test": "test"}[split]

    hg_root_candidates = [emb_dir]
    nested_candidate = emb_dir / hg_root_name
    if nested_candidate not in hg_root_candidates:
        hg_root_candidates.append(nested_candidate)

    for hg_root in hg_root_candidates:
        split_emb_dir = hg_root / split_dir_name / "embeddings"
        raw_emb = split_emb_dir / "embeddings.npy"
        projected_emb = split_emb_dir / "projected_512.npy"
        lbl_path = split_emb_dir / "y_embeddings.npy"

        # Prefer the 512-d projected embeddings when available.
        feature_path = projected_emb if projected_emb.exists() else raw_emb
        if feature_path.exists() and lbl_path.exists():
            emb = np.load(feature_path)
            lbl = np.asarray(np.load(lbl_path)).reshape(-1)
            if emb.shape[0] != lbl.shape[0]:
                raise ValueError(
                    f"Mismatched rows for {feature_path} ({emb.shape[0]}) and "
                    f"{lbl_path} ({lbl.shape[0]})."
                )
            return emb, lbl

    raise FileNotFoundError(
        "Could not find embeddings for "
        f"dataset={dataset}, split={split} under {emb_dir}. "
        "Supported layouts are flat files like "
        f"{dataset}_{split}_embeddings.npy and Hugging Face export layouts like "
        f"{hg_root_name}/{split_dir_name}/embeddings/(embeddings.npy|projected_512.npy) "
        "with y_embeddings.npy."
    )


def make_loader(
    embeddings: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(embeddings).float(),
        torch.from_numpy(labels).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    for (x, _) in loader:
        logits = model(x.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def run_experiment(args: argparse.Namespace) -> dict:
    experiment_start = datetime.now().astimezone()
    experiment_start_ts = time()

    device = torch.device(args.device)
    emb_dir = Path(args.embedding_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = args.dataset

    train_emb, train_lbl = load_embeddings(emb_dir, ds, "train")
    val_emb, val_lbl = load_embeddings(emb_dir, ds, "val")
    test_emb, test_lbl = load_embeddings(emb_dir, ds, "test")

    input_dim = train_emb.shape[1]
    print(f"Dataset: {ds}")
    print(f"  train: {train_emb.shape[0]}  val: {val_emb.shape[0]}  test: {test_emb.shape[0]}")
    print(f"  feature dim: {input_dim}")

    train_loader = make_loader(train_emb, train_lbl, args.batch_size, shuffle=True)
    val_loader = make_loader(val_emb, val_lbl, args.batch_size, shuffle=False)
    test_loader = make_loader(test_emb, test_lbl, args.batch_size, shuffle=False)

    model = MLPClassifier(input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auroc = -1.0
    patience_counter = 0
    best_state = None

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")
    print(f"Experiment start time: {experiment_start.isoformat(timespec='seconds')}")
    train_start = time()

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_probs = predict(model, val_loader, device)
        val_auroc = compute_auroc(val_lbl, val_probs)

        if epoch % 5 == 1 or epoch == args.epochs:
            print(f"  epoch {epoch:3d}  loss={loss:.4f}  val_auroc={val_auroc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch} (best val AUROC: {best_val_auroc:.4f})")
                break

    train_time = time() - train_start
    print(f"Training time: {train_time:.2f}s")

    model.load_state_dict(best_state)
    model.to(device)

    infer_start = time()
    test_probs = predict(model, test_loader, device)
    infer_time = time() - infer_start

    metrics = compute_all_metrics(test_lbl, test_probs, threshold=0.5, n_bins=15)
    metrics["train_time_sec"] = round(train_time, 3)
    metrics["inference_time_sec"] = round(infer_time, 3)
    metrics["best_val_auroc"] = round(best_val_auroc, 6)
    metrics["n_train"] = int(len(train_lbl))
    metrics["n_val"] = int(len(val_lbl))
    metrics["n_test"] = int(len(test_lbl))

    results_path = out_dir / f"exp3_{ds}_results.json"
    fig, _ = plot_reliability_diagram(
        test_lbl, test_probs, n_bins=15, title=f"Exp3 Reliability Diagram ({ds})"
    )
    cal_path = out_dir / f"exp3_{ds}_calibration.png"
    fig.savefig(cal_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {cal_path}")

    fpr, tpr, _ = roc_curve(test_lbl, test_probs)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(fpr, tpr, label=f"AUROC = {metrics['auroc']:.4f}")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title(f"Exp3 ROC Curve ({ds})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    roc_path = out_dir / f"exp3_{ds}_roc.png"
    fig2.savefig(roc_path, dpi=160)
    plt.close(fig2)
    print(f"Saved: {roc_path}")

    if ds == "camelyon17":
        hospitals_path = emb_dir / f"camelyon17_test_hospitals.npy"
        if hospitals_path.exists():
            hospitals = np.load(hospitals_path)
            print("\nPer-hospital test metrics:")
            per_hospital = {}
            for h_id in np.unique(hospitals):
                mask = hospitals == h_id
                if np.sum(mask) < 10 or len(np.unique(test_lbl[mask])) < 2:
                    continue
                h_metrics = compute_all_metrics(test_lbl[mask], test_probs[mask])
                per_hospital[int(h_id)] = h_metrics
                print(f"  Hospital {int(h_id)}: AUROC={h_metrics['auroc']:.4f}  FNR={h_metrics['false_negative_rate']:.4f}")
            ph_path = out_dir / f"exp3_{ds}_per_hospital.json"
            with open(ph_path, "w") as f:
                json.dump(per_hospital, f, indent=2)
            print(f"Saved: {ph_path}")

    experiment_end = datetime.now().astimezone()
    total_runtime = time() - experiment_start_ts
    metrics["experiment_start_time"] = experiment_start.isoformat(timespec="seconds")
    metrics["experiment_end_time"] = experiment_end.isoformat(timespec="seconds")
    metrics["total_runtime_sec"] = round(total_runtime, 3)

    print(f"\nTest metrics ({ds}):")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")

    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {results_path}")
    print(f"Experiment end time:   {experiment_end.isoformat(timespec='seconds')}")
    print(f"Total runtime:         {total_runtime:.2f}s")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp3: deterministic NN baseline.")
    parser.add_argument("--dataset", type=str, required=True, choices=["pcam", "camelyon17", "embed"])
    parser.add_argument("--embedding-dir", type=str, default="data/embeddings")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_experiment(args)


if __name__ == "__main__":
    main()
