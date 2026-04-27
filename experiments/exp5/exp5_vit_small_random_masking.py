"""
Experiment 5: ViT-Small classifier trained jointly on PCam + CAMELYON17.

By default this trains *one* ViT-Small model on the union of the two patch
datasets and writes a single shared checkpoint. After training, the best-
validation checkpoint is reloaded and evaluated on each test split
(``pcam``, ``camelyon17``) and on the combined test set, producing a full
metric report and ROC / calibration plots per split.

Datasets are loaded from:
- ``datasets/pcam-hg/{train,valid,test}``
- ``datasets/camelyon17-hg/{train,valid,test}``

Each split must contain ``images/`` and ``labels.csv``.

Random patch masking is available as an optional training-time knob via
``--mask-ratio``; it is disabled by default.

Example:
    python experiments/exp5/exp5_vit_small_random_masking.py \\
        --dataset combined --data-root datasets --output-root data/exp5_vit \\
        --epochs 10 --batch-size 64 \\
        --wandb --wandb-project ML-Final_project --wandb-entity a-salt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models import ViTSmallRandomMaskClassifier
from my_cholesky.eval_metrics import plot_reliability_diagram
from my_cholesky.result_logging import append_result_row
from predictive_metrics import evaluate_binary_probabilistic_predictions


HG_DATASET_ROOTS = {
    "pcam": "pcam-hg",
    "camelyon17": "camelyon17-hg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a ViT-Small classifier jointly on PCam + CAMELYON17 "
            "(or a single dataset) and evaluate on each test split."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["pcam", "camelyon17", "combined"],
        default="combined",
        help=(
            "'combined' (default) trains one model on PCam + CAMELYON17 jointly "
            "and tests on each split. 'pcam' or 'camelyon17' restrict to one dataset."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(PROJECT_ROOT) / "datasets",
        help="Directory containing the pcam-hg / camelyon17-hg exports.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(PROJECT_ROOT) / "data" / "exp5_vit",
        help="Directory to save the shared checkpoint, history, metrics, and plots.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="AdamW learning rate."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="AdamW weight decay."
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.0,
        help="Fraction of patch tokens to drop randomly during training. 0 (default) disables masking.",
    )
    parser.add_argument(
        "--patch-size", type=int, default=16, help="ViT patch size; must divide --image-size."
    )
    parser.add_argument(
        "--embed-dim", type=int, default=384, help="Transformer embedding dimension (ViT-S default 384)."
    )
    parser.add_argument(
        "--depth", type=int, default=12, help="Number of transformer blocks (ViT-S default 12)."
    )
    parser.add_argument(
        "--num-heads", type=int, default=6, help="Number of attention heads (ViT-S default 6)."
    )
    parser.add_argument(
        "--mlp-ratio", type=float, default=4.0, help="MLP hidden ratio."
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout in MLP/positional path.")
    parser.add_argument("--attn-dropout", type=float, default=0.0, help="Attention dropout.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=96,
        help="Resize images to this square resolution before training.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on train samples *per dataset source* for quick runs.",
    )
    parser.add_argument(
        "--max-valid-samples",
        type=int,
        default=None,
        help="Optional cap on valid samples per dataset source.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap on test samples per dataset source.",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device string: auto, cpu, cuda, cuda:0."
    )

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="ML-Final_project")
    parser.add_argument("--wandb-entity", type=str, default="a-salt")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    parser.add_argument(
        "--wandb-watch", action="store_true",
        help="If set, log gradients/parameters with wandb.watch(model).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(sanitize_for_json(payload), f, indent=2)


def maybe_init_wandb(
    args: argparse.Namespace,
    sources: list[str],
    output_dir: Path,
    n_train: int,
    n_valid: int,
    n_test_per_source: dict[str, int],
    device: torch.device,
):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Install dependencies first (pip install -r requirements.txt)."
        ) from exc

    sources_label = "+".join(sources)
    if args.wandb_run_name:
        run_name = args.wandb_run_name
    elif args.mask_ratio > 0.0:
        run_name = f"exp5-vit-mask{args.mask_ratio:g}-{sources_label}"
    else:
        run_name = f"exp5-vit-{sources_label}"

    architecture_label = (
        f"ViT-Small (random patch masking, mask_ratio={args.mask_ratio:g})"
        if args.mask_ratio > 0.0
        else "ViT-Small"
    )
    config = sanitize_for_json(
        {
            **vars(args),
            "dataset_sources": sources,
            "device": str(device),
            "train_samples": n_train,
            "valid_samples": n_valid,
            "test_samples_per_source": n_test_per_source,
            "test_samples_total": int(sum(n_test_per_source.values())),
            "output_dir": str(output_dir),
            "model_name": "ViTSmallRandomMaskClassifier",
            "architecture": architecture_label,
            "masking_enabled": args.mask_ratio > 0.0,
        }
    )
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        config=config,
        dir=str(output_dir),
        reinit=True,
    )


def load_split_records(split_dir: Path, source: str) -> list[dict[str, Any]]:
    """Read labels.csv and tag each record with its dataset source."""
    labels_path = split_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Expected labels.csv at {labels_path}")

    records: list[dict[str, Any]] = []
    missing_images = 0
    with labels_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = split_dir / row["image"]
            if not image_path.exists():
                missing_images += 1
                continue
            record = {
                "image_path": image_path,
                "image_rel": row["image"],
                "label": int(row["label"]),
                "dataset_source": source,
            }
            for key, value in row.items():
                if key not in {"image", "label"}:
                    record[key] = value
            records.append(record)

    if missing_images:
        print(
            f"  WARNING: skipped {missing_images} rows in {labels_path} "
            "because images were missing."
        )
    if not records:
        raise RuntimeError(f"No usable records found in {split_dir}")
    return records


def maybe_subsample_records(
    records: list[dict[str, Any]], max_samples: int | None, seed: int
) -> list[dict[str, Any]]:
    if max_samples is None or max_samples >= len(records):
        return records
    labels = np.array([r["label"] for r in records], dtype=int)
    indices = np.arange(len(records))
    try:
        selected, _ = train_test_split(
            indices, train_size=max_samples, random_state=seed, stratify=labels
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        selected = rng.choice(indices, size=max_samples, replace=False)
    selected = np.sort(selected)
    return [records[i] for i in selected]


def build_records_for_source(
    source: str, data_root: Path, args: argparse.Namespace
) -> dict[str, list[dict[str, Any]]]:
    directory_name = HG_DATASET_ROOTS[source]
    dataset_root = data_root / directory_name
    if not dataset_root.exists():
        raise FileNotFoundError(f"Expected dataset directory at {dataset_root}")

    train = load_split_records(dataset_root / "train", source)
    valid = load_split_records(dataset_root / "valid", source)
    test = load_split_records(dataset_root / "test", source)
    train = maybe_subsample_records(train, args.max_train_samples, seed=args.seed)
    valid = maybe_subsample_records(valid, args.max_valid_samples, seed=args.seed + 1)
    test = maybe_subsample_records(test, args.max_test_samples, seed=args.seed + 2)
    return {"train": train, "valid": valid, "test": test}


class PatchClassificationDataset(Dataset):
    """Image + binary label dataset, returning (tensor, label, image_rel, dataset_source)."""

    def __init__(self, records: list[dict[str, Any]], image_size: int) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - 0.5) / 0.5
        label = torch.tensor(float(record["label"]), dtype=torch.float32)
        return tensor, label, str(record["image_rel"]), str(record["dataset_source"])


def compute_pos_weight(records: list[dict[str, Any]]) -> torch.Tensor:
    labels = np.array([r["label"] for r in records], dtype=int)
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    if pos == 0 or neg == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(neg / pos, dtype=torch.float32)


def make_loader(
    records: list[dict[str, Any]],
    image_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = PatchClassificationDataset(records, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def confusion_counts_rates(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
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


def epoch_metrics(
    labels: np.ndarray, probs: np.ndarray, avg_loss: float
) -> dict[str, float | None]:
    """Lightweight per-epoch summary used during training."""
    preds = (probs >= 0.5).astype(int)
    metrics = evaluate_binary_probabilistic_predictions(
        labels, probs, threshold=0.5, n_bins=15
    )
    metrics["loss"] = float(avg_loss)
    metrics["precision"] = float(precision_score(labels, preds, zero_division=0))
    metrics["recall"] = float(recall_score(labels, preds, zero_division=0))
    metrics["positive_rate"] = float(np.mean(preds))
    metrics["target_positive_rate"] = float(np.mean(labels))
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> dict[str, float | None]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for images, labels, _, _ in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)
        bsz = labels.shape[0]
        total_loss += float(loss.item()) * bsz
        total_examples += bsz
        all_labels.append(labels.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

        progress.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            avg_loss=f"{total_loss / max(total_examples, 1):.4f}",
        )

    return epoch_metrics(
        np.concatenate(all_labels),
        np.concatenate(all_probs),
        total_loss / max(total_examples, 1),
    )


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Eval",
) -> tuple[dict[str, float | None], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    predictions: list[dict[str, Any]] = []

    progress = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for images, labels, image_rels, sources in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images, apply_masking=False)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            bsz = labels.shape[0]
            total_loss += float(loss.item()) * bsz
            total_examples += bsz
            labels_cpu = labels.detach().cpu().numpy()
            probs_cpu = probs.detach().cpu().numpy()
            all_labels.append(labels_cpu)
            all_probs.append(probs_cpu)
            for image_rel, source, label, prob in zip(
                image_rels, sources, labels_cpu, probs_cpu, strict=True
            ):
                predictions.append({
                    "dataset_source": source,
                    "image": image_rel,
                    "label": int(label),
                    "probability": float(prob),
                    "prediction": int(prob >= 0.5),
                })

            progress.set_postfix(
                avg_loss=f"{total_loss / max(total_examples, 1):.4f}"
            )

    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)
    metrics = epoch_metrics(labels_np, probs_np, total_loss / max(total_examples, 1))
    return metrics, labels_np, probs_np, predictions


def save_predictions_csv(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset_source", "image", "label", "probability", "prediction"]
        )
        writer.writeheader()
        writer.writerows(predictions)


def save_roc_plot(
    path: Path, y_true: np.ndarray, y_prob: np.ndarray, auroc: float | None, title: str
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    label = (
        f"AUROC = {auroc:.4f}"
        if auroc is not None and math.isfinite(auroc)
        else "AUROC = n/a"
    )
    ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_split(
    label: str,
    title_label: str,
    records: list[dict[str, Any]],
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
    timing_ctx: dict[str, float],
    base_metrics_extra: dict[str, Any],
    wandb_run,
) -> dict[str, Any]:
    """Run test evaluation on a split, save plots/JSON/CSV, return metrics dict."""
    loader = make_loader(
        records,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    infer_start = perf_counter()
    eval_metrics, y_true, y_prob, predictions = evaluate_loader(
        model, loader, criterion, device, desc=f"Test {title_label}"
    )
    inference_time = perf_counter() - infer_start

    evaluation_start = perf_counter()
    metrics = evaluate_binary_probabilistic_predictions(
        y_true, y_prob, threshold=0.5, n_bins=15
    )
    metrics.update(confusion_counts_rates(y_true, y_prob, threshold=0.5))
    metrics["loss"] = eval_metrics["loss"]
    metrics["precision"] = eval_metrics["precision"]
    metrics["recall"] = eval_metrics["recall"]
    metrics["sensitivity"] = metrics.get("sensitivity_TPR")
    metrics["positive_rate"] = eval_metrics["positive_rate"]
    metrics["target_positive_rate"] = eval_metrics["target_positive_rate"]
    metrics["negative_log_loss"] = metrics.get("negative_log_likelihood_mean")
    metrics["timing_scope"] = "data_loading, training, test_inference, evaluation_plots"
    metrics["data_loading_time_sec"] = round(timing_ctx["data_loading_time_sec"], 3)
    metrics["train_time_sec"] = round(timing_ctx["train_time_sec"], 3)
    metrics["fit_or_train_time_sec"] = round(timing_ctx["train_time_sec"], 3)
    metrics["inference_time_sec"] = round(inference_time, 3)
    metrics["n_test"] = int(len(y_true))
    metrics.update(base_metrics_extra)

    cal_path = output_dir / f"exp5_{label}_calibration.png"
    fig, _ = plot_reliability_diagram(
        y_true, y_prob, n_bins=15, title=f"Exp5 ViT-Small Reliability Diagram ({title_label})"
    )
    fig.savefig(cal_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {cal_path}")

    roc_path = output_dir / f"exp5_{label}_roc.png"
    save_roc_plot(roc_path, y_true, y_prob, metrics.get("auroc"),
                  title=f"Exp5 ViT-Small ROC Curve ({title_label})")
    print(f"Saved: {roc_path}")

    pred_path = output_dir / f"exp5_{label}_test_predictions.csv"
    save_predictions_csv(pred_path, predictions)
    print(f"Saved: {pred_path}")

    evaluation_time = perf_counter() - evaluation_start
    metrics["evaluation_time_sec"] = round(evaluation_time, 3)

    print(f"\nTest metrics ({title_label}):")
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

    results_path = output_dir / f"exp5_{label}_results.json"
    save_json(results_path, metrics)
    print(f"Saved: {results_path}")

    if wandb_run is not None:
        wandb_run.log(
            {
                f"test_{label}/loss": metrics["loss"],
                f"test_{label}/accuracy": metrics["accuracy"],
                f"test_{label}/auroc": metrics["auroc"],
                f"test_{label}/auprc": metrics["auprc"],
                f"test_{label}/brier": metrics["brier"],
                f"test_{label}/ece": metrics["ece"],
                f"test_{label}/negative_log_loss": metrics["negative_log_loss"],
                f"test_{label}/precision": metrics["precision"],
                f"test_{label}/recall": metrics["recall"],
                f"test_{label}/sensitivity_TPR": metrics["sensitivity_TPR"],
                f"test_{label}/specificity_TNR": metrics["specificity_TNR"],
                f"test_{label}/false_positive_rate": metrics["false_positive_rate"],
                f"test_{label}/false_negative_rate": metrics["false_negative_rate"],
            },
            step=metrics.get("epochs_ran") or args.epochs,
        )
        for k, v in metrics.items():
            wandb_run.summary[f"test_{label}/{k}"] = v
        wandb_run.summary[f"artifact_{label}_results"] = str(results_path)
        wandb_run.summary[f"artifact_{label}_roc"] = str(roc_path)
        wandb_run.summary[f"artifact_{label}_calibration"] = str(cal_path)

    return metrics


def run_experiment(args: argparse.Namespace, device: torch.device) -> None:
    experiment_start = datetime.now().astimezone()
    experiment_start_ts = perf_counter()

    output_dir = args.output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "combined":
        sources = ["pcam", "camelyon17"]
    else:
        sources = [args.dataset]

    data_loading_start = perf_counter()
    bundles = {s: build_records_for_source(s, args.data_root, args) for s in sources}
    train_records = [r for s in sources for r in bundles[s]["train"]]
    valid_records = [r for s in sources for r in bundles[s]["valid"]]
    rng = random.Random(args.seed)
    rng.shuffle(train_records)
    data_loading_time = perf_counter() - data_loading_start

    n_test_per_source = {s: len(bundles[s]["test"]) for s in sources}

    print("")
    print("==============================================================================")
    masking_label = (
        f"random masking (ratio={args.mask_ratio:g})"
        if args.mask_ratio > 0.0 else "masking disabled"
    )
    print(f"Experiment 5: ViT-Small on {'+'.join(sources)} ({masking_label})")
    print("==============================================================================")
    print(f"Output dir       : {output_dir}")
    for s in sources:
        b = bundles[s]
        print(f"  {s:11s}: train={len(b['train'])}  valid={len(b['valid'])}  test={len(b['test'])}")
    print(f"Combined train   : {len(train_records)}")
    print(f"Combined valid   : {len(valid_records)}")
    print(f"Image size       : {args.image_size}  patch size: {args.patch_size}")
    print(f"Embed/depth/heads: {args.embed_dim}/{args.depth}/{args.num_heads}")
    print(f"Mask ratio       : {args.mask_ratio}")
    print(f"Device           : {device}")
    print("==============================================================================")

    wandb_run = maybe_init_wandb(
        args, sources, output_dir,
        n_train=len(train_records), n_valid=len(valid_records),
        n_test_per_source=n_test_per_source, device=device,
    )

    train_loader = make_loader(
        train_records, image_size=args.image_size, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, device=device,
    )
    valid_loader = make_loader(
        valid_records, image_size=args.image_size, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, device=device,
    )

    model = ViTSmallRandomMaskClassifier(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        mask_ratio=args.mask_ratio,
    ).to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    pos_weight = compute_pos_weight(train_records).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    if wandb_run is not None and args.wandb_watch:
        wandb_run.watch(model, log="all", log_freq=50)

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_val_auroc = float("-inf")
    best_epoch = -1
    epochs_ran = 0
    best_checkpoint_path = output_dir / "best_model.pt"

    train_phase_start = perf_counter()
    epoch_bar = tqdm(
        range(1, args.epochs + 1),
        desc="Training",
        unit="epoch",
        total=args.epochs,
    )
    for epoch in epoch_bar:
        epoch_start = perf_counter()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, total_epochs=args.epochs,
        )
        valid_metrics, _, _, _ = evaluate_loader(
            model, valid_loader, criterion, device,
            desc=f"Valid {epoch}/{args.epochs}",
        )
        epoch_time = perf_counter() - epoch_start
        epochs_ran = epoch
        epoch_bar.set_postfix(
            train_loss=f"{train_metrics['loss']:.4f}",
            valid_loss=f"{valid_metrics['loss']:.4f}",
            valid_auroc=(
                f"{valid_metrics['auroc']:.4f}"
                if valid_metrics["auroc"] is not None else "n/a"
            ),
        )

        history.append({
            "epoch": epoch,
            "seconds": epoch_time,
            "train": train_metrics,
            "valid": valid_metrics,
        })

        valid_auroc = valid_metrics.get("auroc")
        valid_score = (
            float(valid_auroc) if valid_auroc is not None and math.isfinite(float(valid_auroc))
            else -float(valid_metrics["loss"])
        )
        if valid_score > best_score:
            best_score = valid_score
            best_epoch = epoch
            if valid_auroc is not None and math.isfinite(float(valid_auroc)):
                best_val_auroc = float(valid_auroc)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "dataset_sources": sources,
                    "args": sanitize_for_json(vars(args)),
                    "valid_metrics": sanitize_for_json(valid_metrics),
                },
                best_checkpoint_path,
            )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "epoch_seconds": epoch_time,
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/auroc": train_metrics["auroc"],
                    "train/auprc": train_metrics["auprc"],
                    "train/brier": train_metrics["brier"],
                    "valid/loss": valid_metrics["loss"],
                    "valid/accuracy": valid_metrics["accuracy"],
                    "valid/auroc": valid_metrics["auroc"],
                    "valid/auprc": valid_metrics["auprc"],
                    "valid/brier": valid_metrics["brier"],
                    "best_val_auroc": best_val_auroc,
                    "best_epoch_so_far": best_epoch,
                },
                step=epoch,
            )

        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_acc={valid_metrics['accuracy']:.4f} "
            f"valid_auroc={valid_metrics['auroc'] if valid_metrics['auroc'] is not None else 'n/a'} "
            f"time={epoch_time:.1f}s"
        )

    train_time = perf_counter() - train_phase_start

    save_json(
        output_dir / "history.json",
        {
            "dataset_sources": sources,
            "best_epoch": best_epoch,
            "best_val_auroc": best_val_auroc if math.isfinite(best_val_auroc) else None,
            "history": history,
        },
    )

    print(f"\nLoading best checkpoint from epoch {best_epoch}: {best_checkpoint_path}")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    base_extra = {
        "best_val_auroc": (
            round(best_val_auroc, 6) if math.isfinite(best_val_auroc) else None
        ),
        "model_architecture": "vit_small",
        "patch_size": int(args.patch_size),
        "image_size": int(args.image_size),
        "embed_dim": int(args.embed_dim),
        "depth": int(args.depth),
        "num_heads": int(args.num_heads),
        "mlp_ratio": float(args.mlp_ratio),
        "mask_ratio": float(args.mask_ratio),
        "dropout": float(args.dropout),
        "attn_dropout": float(args.attn_dropout),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "epochs_requested": int(args.epochs),
        "epochs_ran": int(epochs_ran),
        "trainable_parameters": int(n_parameters),
        "n_train": int(len(train_records)),
        "n_val": int(len(valid_records)),
        "dataset_sources": sources,
    }
    timing_ctx = {
        "data_loading_time_sec": data_loading_time,
        "train_time_sec": train_time,
    }

    per_split_results: dict[str, dict[str, Any]] = {}
    for source in sources:
        per_split_results[source] = evaluate_split(
            label=source,
            title_label=source,
            records=bundles[source]["test"],
            model=model,
            criterion=criterion,
            device=device,
            args=args,
            output_dir=output_dir,
            timing_ctx=timing_ctx,
            base_metrics_extra=base_extra,
            wandb_run=wandb_run,
        )

    if len(sources) > 1:
        combined_test = [r for s in sources for r in bundles[s]["test"]]
        per_split_results["combined"] = evaluate_split(
            label="combined",
            title_label="combined (" + "+".join(sources) + ")",
            records=combined_test,
            model=model,
            criterion=criterion,
            device=device,
            args=args,
            output_dir=output_dir,
            timing_ctx=timing_ctx,
            base_metrics_extra=base_extra,
            wandb_run=wandb_run,
        )

    experiment_end = datetime.now().astimezone()
    total_runtime = perf_counter() - experiment_start_ts
    for label, metrics in per_split_results.items():
        metrics["experiment_start_time"] = experiment_start.isoformat(timespec="seconds")
        metrics["experiment_end_time"] = experiment_end.isoformat(timespec="seconds")
        metrics["total_runtime_sec"] = round(total_runtime, 3)
        metrics["total_pipeline_time_sec"] = round(total_runtime, 3)
        save_json(output_dir / f"exp5_{label}_results.json", metrics)

    summary_payload = {
        "dataset_sources": sources,
        "best_epoch": best_epoch,
        "best_val_auroc": (
            round(best_val_auroc, 6) if math.isfinite(best_val_auroc) else None
        ),
        "best_checkpoint_path": str(best_checkpoint_path),
        "n_train": len(train_records),
        "n_val": len(valid_records),
        "n_test_per_source": n_test_per_source,
        "n_test_combined": int(sum(n_test_per_source.values())) if len(sources) > 1 else None,
        "data_loading_time_sec": round(data_loading_time, 3),
        "train_time_sec": round(train_time, 3),
        "total_runtime_sec": round(total_runtime, 3),
        "experiment_start_time": experiment_start.isoformat(timespec="seconds"),
        "experiment_end_time": experiment_end.isoformat(timespec="seconds"),
        "trainable_parameters": int(n_parameters),
        "epochs_requested": int(args.epochs),
        "epochs_ran": int(epochs_ran),
        "results": per_split_results,
    }
    summary_path = output_dir / "summary.json"
    save_json(summary_path, summary_payload)
    print(f"\nSaved: {summary_path}")

    csv_artifacts: list[str] = [str(best_checkpoint_path), str(output_dir / "history.json"),
                                str(summary_path)]
    for label in per_split_results:
        for suffix in ("results.json", "roc.png", "calibration.png", "test_predictions.csv"):
            csv_artifacts.append(str(output_dir / f"exp5_{label}_{suffix}"))

    for label, metrics in per_split_results.items():
        csv_path = append_result_row(
            {
                "experiment": "exp5",
                "script_path": "experiments/exp5/exp5_vit_small_random_masking.py",
                "artifacts": json.dumps(csv_artifacts),
                "dataset": label,
                "dataset_sources": json.dumps(sources),
                "training_mode": "joint" if len(sources) > 1 else "single",
                "seed": args.seed,
                "method_name": (
                    "vit_small_random_mask"
                    if args.mask_ratio > 0.0 else "vit_small"
                ),
                "device": str(device),
                **metrics,
            }
        )
    print(f"Appended CSV metrics to {csv_path}")

    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["best_val_auroc"] = (
            best_val_auroc if math.isfinite(best_val_auroc) else None
        )
        wandb_run.summary["checkpoint_path"] = str(best_checkpoint_path)
        wandb_run.summary["summary_path"] = str(summary_path)
        wandb_run.finish()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_experiment(args, device)


if __name__ == "__main__":
    main()
