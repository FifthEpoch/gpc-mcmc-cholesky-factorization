"""
Experiment 4: TabPFN tabular foundation model baseline.

Uses TabPFN (Hollmann et al., 2025) as a classifier on the same frozen
embeddings produced by extract_embeddings.py.  TabPFN is a pre-trained
transformer for tabular data that requires no gradient-based training --
it fits via in-context learning in a single forward pass.

TabPFN works best on datasets with up to ~50k samples.  For larger
training sets the script subsamples to --max-train-samples (default 50000).

Usage:
    python experiments/exp4_tabpfn_baseline.py \
        --dataset pcam \
        --embedding-dir data/embeddings \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from my_cholesky.eval_metrics import (
    compute_all_metrics,
    plot_reliability_diagram,
)


def load_embeddings(
    emb_dir: Path, dataset: str, split: str
) -> tuple[np.ndarray, np.ndarray]:
    emb = np.load(emb_dir / f"{dataset}_{split}_embeddings.npy")
    lbl = np.load(emb_dir / f"{dataset}_{split}_labels.npy")
    return emb, lbl


def subsample(
    X: np.ndarray, y: np.ndarray, max_n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subsample to at most *max_n* rows."""
    if len(y) <= max_n:
        return X, y
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    ratio = len(pos) / len(y)
    n_pos = max(1, int(round(max_n * ratio)))
    n_neg = max_n - n_pos
    idx = np.concatenate([
        rng.choice(pos, size=min(n_pos, len(pos)), replace=False),
        rng.choice(neg, size=min(n_neg, len(neg)), replace=False),
    ])
    rng.shuffle(idx)
    return X[idx], y[idx]


def predict_in_chunks(
    clf, X: np.ndarray, chunk_size: int = 1000
) -> np.ndarray:
    """Call predict_proba in chunks to avoid recomputing the training set."""
    all_probs = []
    for i in range(0, len(X), chunk_size):
        chunk = X[i : i + chunk_size]
        proba = clf.predict_proba(chunk)
        all_probs.append(proba[:, 1])
    return np.concatenate(all_probs)


def run_experiment(args: argparse.Namespace) -> dict:
    from tabpfn import TabPFNClassifier

    emb_dir = Path(args.embedding_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = args.dataset

    train_emb, train_lbl = load_embeddings(emb_dir, ds, "train")
    val_emb, val_lbl = load_embeddings(emb_dir, ds, "val")
    test_emb, test_lbl = load_embeddings(emb_dir, ds, "test")

    print(f"Dataset: {ds}")
    print(f"  train: {train_emb.shape[0]}  val: {val_emb.shape[0]}  test: {test_emb.shape[0]}")
    print(f"  feature dim: {train_emb.shape[1]}")

    if train_emb.shape[0] > args.max_train_samples:
        print(f"  Subsampling train from {train_emb.shape[0]} to {args.max_train_samples}")
        train_emb, train_lbl = subsample(
            train_emb, train_lbl, args.max_train_samples, args.seed
        )

    clf_kwargs = dict(device=args.device, random_state=args.seed)
    if train_emb.shape[0] > 10_000:
        clf_kwargs["ignore_pretraining_limits"] = True

    print(f"\nFitting TabPFN on {train_emb.shape[0]} samples ...")
    clf = TabPFNClassifier(**clf_kwargs)
    fit_start = time()
    clf.fit(train_emb, train_lbl.astype(int))
    fit_time = time() - fit_start
    print(f"Fit time: {fit_time:.2f}s")

    infer_start = time()
    test_probs = predict_in_chunks(clf, test_emb, chunk_size=args.predict_chunk_size)
    infer_time = time() - infer_start
    print(f"Inference time ({len(test_lbl)} samples): {infer_time:.2f}s")

    metrics = compute_all_metrics(test_lbl, test_probs, threshold=0.5, n_bins=15)
    metrics["fit_time_sec"] = round(fit_time, 3)
    metrics["inference_time_sec"] = round(infer_time, 3)
    metrics["n_train"] = int(len(train_lbl))
    metrics["n_test"] = int(len(test_lbl))

    print(f"\nTest metrics ({ds}):")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")

    results_path = out_dir / f"exp4_{ds}_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {results_path}")

    fig, _ = plot_reliability_diagram(
        test_lbl, test_probs, n_bins=15, title=f"Exp4 TabPFN Reliability Diagram ({ds})"
    )
    cal_path = out_dir / f"exp4_{ds}_calibration.png"
    fig.savefig(cal_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {cal_path}")

    fpr, tpr, _ = roc_curve(test_lbl, test_probs)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(fpr, tpr, label=f"AUROC = {metrics['auroc']:.4f}")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title(f"Exp4 TabPFN ROC Curve ({ds})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    roc_path = out_dir / f"exp4_{ds}_roc.png"
    fig2.savefig(roc_path, dpi=160)
    plt.close(fig2)
    print(f"Saved: {roc_path}")

    if ds == "camelyon17":
        hospitals_path = emb_dir / "camelyon17_test_hospitals.npy"
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
            ph_path = out_dir / f"exp4_{ds}_per_hospital.json"
            with open(ph_path, "w") as f:
                json.dump(per_hospital, f, indent=2)
            print(f"Saved: {ph_path}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp4: TabPFN tabular model baseline.")
    parser.add_argument("--dataset", type=str, required=True, choices=["pcam", "camelyon17", "embed"])
    parser.add_argument("--embedding-dir", type=str, default="data/embeddings")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--max-train-samples", type=int, default=50000,
                        help="Subsample train set to this size (TabPFN limit)")
    parser.add_argument("--predict-chunk-size", type=int, default=1000,
                        help="Chunk size for batched predict_proba calls")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    run_experiment(args)


if __name__ == "__main__":
    main()
