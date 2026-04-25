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
import csv
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
    # Format 1 (default project format):
    #   <emb_dir>/<dataset>_<split>_embeddings.npy
    #   <emb_dir>/<dataset>_<split>_labels.npy
    emb_path = emb_dir / f"{dataset}_{split}_embeddings.npy"
    lbl_path = emb_dir / f"{dataset}_{split}_labels.npy"
    if emb_path.exists() and lbl_path.exists():
        return np.load(emb_path), np.load(lbl_path)

    # Format 2 (partner HG export layout), either:
    #   <emb_dir>/<dataset>-hg/<split_dir>/embeddings/...
    # or
    #   <emb_dir>/<split_dir>/embeddings/...
    split_dir = "valid" if split == "val" else split
    dataset_roots = [
        emb_dir / f"{dataset}-hg",
        emb_dir / dataset,
        emb_dir,
    ]
    candidate_emb_files = ["projected_512.npy", "embeddings.npy"]
    candidate_lbl_files = ["y_embeddings.npy", "labels.npy"]

    for ds_root in dataset_roots:
        split_root = ds_root / split_dir
        emb_root = split_root / "embeddings"
        if not emb_root.exists():
            continue

        emb_file = next((emb_root / name for name in candidate_emb_files if (emb_root / name).exists()), None)
        if emb_file is None:
            continue

        lbl_file = next((emb_root / name for name in candidate_lbl_files if (emb_root / name).exists()), None)
        if lbl_file is not None:
            return np.load(emb_file), np.load(lbl_file)

        csv_path = split_root / "labels.csv"
        if csv_path.exists():
            labels = _load_labels_from_csv(csv_path)
            return np.load(emb_file), labels

    raise FileNotFoundError(
        "Could not find embeddings for dataset="
        f"{dataset!r}, split={split!r} under {emb_dir}. "
        "Expected either standard files "
        f"({dataset}_{split}_embeddings.npy / {dataset}_{split}_labels.npy) "
        "or partner HG layout under <root>/<dataset>-hg/<split>/embeddings/."
    )


def _load_labels_from_csv(csv_path: Path) -> np.ndarray:
    """Read labels from split-level labels.csv (expects a 'label' column)."""
    labels: list[int] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"labels.csv has no header: {csv_path}")
        if "label" in reader.fieldnames:
            key = "label"
        elif "y" in reader.fieldnames:
            key = "y"
        else:
            # Fallback to first column if header names differ.
            key = reader.fieldnames[0]
        for row in reader:
            labels.append(int(row[key]))
    return np.asarray(labels, dtype=np.int64)


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


def confusion_counts_rates(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Return TP/TN/FP/FN counts and common classification rates."""
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    tpr = tp / max(tp + fn, 1)  # sensitivity / recall
    tnr = tn / max(tn + fp, 1)  # specificity
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "true_positive_rate": float(tpr),
        "true_negative_rate": float(tnr),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
    }


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
    metrics.update(confusion_counts_rates(test_lbl, test_probs, threshold=0.5))
    metrics["fit_time_sec"] = round(fit_time, 3)
    metrics["inference_time_sec"] = round(infer_time, 3)
    metrics["n_train"] = int(len(train_lbl))
    metrics["n_test"] = int(len(test_lbl))

    print(f"\nTest metrics ({ds}):")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")
    print("\nConfusion details (@ threshold=0.5):")
    print(f"  TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print(
        "  TPR={:.4f}  TNR={:.4f}  FPR={:.4f}  FNR={:.4f}".format(
            metrics["true_positive_rate"],
            metrics["true_negative_rate"],
            metrics["false_positive_rate"],
            metrics["false_negative_rate"],
        )
    )

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
