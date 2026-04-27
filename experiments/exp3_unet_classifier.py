"""
Experiment 3: U-Net based binary classification on pathology patch datasets.

Supports:
- PCam exported as datasets/pcam-hg/{train,valid,test}
- CAMELYON17 exported as datasets/camelyon17-hg/{train,valid,test}

Each split is expected to contain:
- images/
- labels.csv

The model is a U-Net style encoder-decoder that produces a dense logit map and
then averages it spatially to obtain an image-level binary logit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


# Allow direct script execution without package install.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models import UNetBinaryClassifier
from my_cholesky.result_logging import append_result_row
from predictive_metrics import evaluate_binary_probabilistic_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a U-Net style binary classifier on pathology patch datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=["pcam", "camelyon17"],
        default="pcam",
        help="Which dataset export to train on.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(PROJECT_ROOT) / "datasets",
        help="Directory containing pcam-hg and camelyon17-hg exports.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(PROJECT_ROOT) / "data" / "exp3_unet",
        help="Directory to save checkpoints, histories, and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Adam learning rate."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Adam weight decay."
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="Base width of the U-Net encoder/decoder.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=96,
        help="Resize images to this square resolution before training.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="PyTorch DataLoader workers."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on train samples for quick experiments.",
    )
    parser.add_argument(
        "--max-valid-samples",
        type=int,
        default=None,
        help="Optional cap on validation samples for quick experiments.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap on test samples for quick experiments.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string, for example auto, cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging for Experiment 3.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="ML-Final_project",
        help="W&B project name when --wandb is enabled.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="a-salt",
        help="Optional W&B entity/team name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run name prefix.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Optional W&B group name, useful when running both datasets.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B mode when --wandb is enabled.",
    )
    parser.add_argument(
        "--wandb-watch",
        action="store_true",
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
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(sanitize_for_json(payload), f, indent=2)


def maybe_init_wandb(
    args: argparse.Namespace,
    dataset_name: str,
    output_dir: Path,
    split_records: dict[str, list[dict[str, Any]]],
    device: torch.device,
):
    if not args.wandb or args.wandb_mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Install dependencies first "
            "(pip install -r requirements.txt)."
        ) from exc

    run_name = (
        f"{args.wandb_run_name}-{dataset_name}"
        if args.wandb_run_name
        else f"exp3-{dataset_name}"
    )
    config = sanitize_for_json(
        {
            **vars(args),
            "dataset_name": dataset_name,
            "device": str(device),
            "train_samples": len(split_records["train"]),
            "valid_samples": len(split_records["valid"]),
            "test_samples": len(split_records["test"]),
            "output_dir": str(output_dir),
            "model_name": "UNetBinaryClassifier",
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


def load_split_records(split_dir: Path) -> list[dict[str, Any]]:
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
            }
            for key, value in row.items():
                if key not in {"image", "label"}:
                    record[key] = value
            records.append(record)

    if missing_images:
        print(
            f"  WARNING: skipped {missing_images} rows in {labels_path} because images were missing."
        )
    if not records:
        raise RuntimeError(f"No usable records found in {split_dir}")
    return records


def maybe_subsample_records(
    records: list[dict[str, Any]], max_samples: int | None, seed: int
) -> list[dict[str, Any]]:
    if max_samples is None or max_samples >= len(records):
        return records

    labels = np.array([record["label"] for record in records], dtype=int)
    indices = np.arange(len(records))
    try:
        selected, _ = train_test_split(
            indices,
            train_size=max_samples,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        selected = rng.choice(indices, size=max_samples, replace=False)

    selected = np.sort(selected)
    return [records[index] for index in selected]


class PatchClassificationDataset(Dataset):
    """Simple image + binary label dataset backed by labels.csv."""

    def __init__(self, records: list[dict[str, Any]], image_size: int) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - 0.5) / 0.5

        label = torch.tensor(float(record["label"]), dtype=torch.float32)
        return tensor, label, str(record["image_rel"])


def compute_pos_weight(records: list[dict[str, Any]]) -> torch.Tensor:
    labels = np.array([record["label"] for record in records], dtype=int)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(negatives / positives, dtype=torch.float32)


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


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    avg_loss: float,
) -> dict[str, float | None]:
    preds = (probs >= 0.5).astype(int)
    metrics: dict[str, float | None] = evaluate_binary_probabilistic_predictions(
        labels,
        probs,
        threshold=0.5,
        n_bins=15,
    )
    metrics.update({
        "loss": float(avg_loss),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "positive_rate": float(np.mean(preds)),
        "target_positive_rate": float(np.mean(labels)),
    })
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int | None = None,
    total_epochs: int | None = None,
    dataset_name: str | None = None,
) -> dict[str, float | None]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    if epoch is not None and total_epochs is not None:
        prefix = f"Epoch {epoch}/{total_epochs}"
    else:
        prefix = "Train"
    if dataset_name:
        prefix = f"[{dataset_name}] {prefix}"

    progress = tqdm(loader, desc=prefix, leave=False)

    for images, labels, _ in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)
        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        all_labels.append(labels.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

        progress.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            avg_loss=f"{total_loss / max(total_examples, 1):.4f}",
        )

    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)
    return compute_metrics(labels_np, probs_np, total_loss / max(total_examples, 1))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    predictions: list[dict[str, Any]] = []

    with torch.no_grad():
        for images, labels, image_paths in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            labels_cpu = labels.detach().cpu().numpy()
            probs_cpu = probs.detach().cpu().numpy()
            all_labels.append(labels_cpu)
            all_probs.append(probs_cpu)

            for image_path, label, prob in zip(image_paths, labels_cpu, probs_cpu, strict=True):
                predictions.append(
                    {
                        "image": image_path,
                        "label": int(label),
                        "probability": float(prob),
                        "prediction": int(prob >= 0.5),
                    }
                )

    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)
    metrics = compute_metrics(labels_np, probs_np, total_loss / max(total_examples, 1))
    return metrics, predictions


def save_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "label", "probability", "prediction"]
        )
        writer.writeheader()
        writer.writerows(predictions)


def build_dataset_bundle(
    dataset_name: str, data_root: Path, args: argparse.Namespace
) -> dict[str, list[dict[str, Any]]]:
    directory_name = {"pcam": "pcam-hg", "camelyon17": "camelyon17-hg"}[dataset_name]
    dataset_root = data_root / directory_name
    if not dataset_root.exists():
        raise FileNotFoundError(f"Expected dataset directory at {dataset_root}")

    split_records = {
        "train": load_split_records(dataset_root / "train"),
        "valid": load_split_records(dataset_root / "valid"),
        "test": load_split_records(dataset_root / "test"),
    }
    split_records["train"] = maybe_subsample_records(
        split_records["train"], args.max_train_samples, seed=args.seed
    )
    split_records["valid"] = maybe_subsample_records(
        split_records["valid"], args.max_valid_samples, seed=args.seed + 1
    )
    split_records["test"] = maybe_subsample_records(
        split_records["test"], args.max_test_samples, seed=args.seed + 2
    )
    return split_records


def train_dataset(
    dataset_name: str,
    split_records: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    output_dir = args.output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("==============================================================================")
    print(f"Experiment 3: U-Net classifier on {dataset_name}")
    print("==============================================================================")
    print(f"Output dir      : {output_dir}")
    print(f"Train samples   : {len(split_records['train'])}")
    print(f"Valid samples   : {len(split_records['valid'])}")
    print(f"Test samples    : {len(split_records['test'])}")
    print(f"Device          : {device}")
    print("==============================================================================")

    wandb_run = maybe_init_wandb(args, dataset_name, output_dir, split_records, device)

    train_loader = make_loader(
        split_records["train"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    valid_loader = make_loader(
        split_records["valid"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )
    test_loader = make_loader(
        split_records["test"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    model = UNetBinaryClassifier(base_channels=args.base_channels).to(device)
    pos_weight = compute_pos_weight(split_records["train"]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if wandb_run is not None and args.wandb_watch:
        wandb_run.watch(model, log="all", log_freq=50)

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = -1
    best_checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=args.epochs,
            dataset_name=dataset_name,
        )
        valid_metrics, _ = evaluate(model, valid_loader, criterion, device)
        epoch_time = time.perf_counter() - epoch_start

        history_entry = {
            "epoch": epoch,
            "seconds": epoch_time,
            "train": train_metrics,
            "valid": valid_metrics,
        }
        history.append(history_entry)

        valid_score = (
            valid_metrics["auroc"]
            if valid_metrics["auroc"] is not None
            else -float(valid_metrics["loss"])
        )
        if valid_score > best_score:
            best_score = valid_score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "dataset": dataset_name,
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
                    "train/precision": train_metrics["precision"],
                    "train/recall": train_metrics["recall"],
                    "train/brier": train_metrics["brier"],
                    "train/positive_rate": train_metrics["positive_rate"],
                    "train/target_positive_rate": train_metrics["target_positive_rate"],
                    "train/auroc": train_metrics["auroc"],
                    "train/auprc": train_metrics["auprc"],
                    "valid/loss": valid_metrics["loss"],
                    "valid/accuracy": valid_metrics["accuracy"],
                    "valid/precision": valid_metrics["precision"],
                    "valid/recall": valid_metrics["recall"],
                    "valid/brier": valid_metrics["brier"],
                    "valid/positive_rate": valid_metrics["positive_rate"],
                    "valid/target_positive_rate": valid_metrics["target_positive_rate"],
                    "valid/auroc": valid_metrics["auroc"],
                    "valid/auprc": valid_metrics["auprc"],
                    "best_valid_score": best_score,
                    "best_epoch_so_far": best_epoch,
                },
                step=epoch,
            )

        print(
            f"[{dataset_name}] epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_acc={valid_metrics['accuracy']:.4f} "
            f"valid_auroc={valid_metrics['auroc'] if valid_metrics['auroc'] is not None else 'n/a'} "
            f"time={epoch_time:.1f}s"
        )

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_predictions = evaluate(model, test_loader, criterion, device)

    save_predictions(output_dir / "test_predictions.csv", test_predictions)
    save_json(
        output_dir / "history.json",
        {
            "dataset": dataset_name,
            "best_epoch": best_epoch,
            "history": history,
        },
    )
    save_json(
        output_dir / "test_metrics.json",
        {
            "dataset": dataset_name,
            "best_epoch": best_epoch,
            "test": test_metrics,
        },
    )

    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["checkpoint_path"] = str(best_checkpoint_path)
        wandb_run.summary["test_metrics_path"] = str(output_dir / "test_metrics.json")
        for metric_name, metric_value in test_metrics.items():
            wandb_run.summary[f"test/{metric_name}"] = metric_value
        wandb_run.log(
            {
                "test/loss": test_metrics["loss"],
                "test/accuracy": test_metrics["accuracy"],
                "test/precision": test_metrics["precision"],
                "test/recall": test_metrics["recall"],
                "test/brier": test_metrics["brier"],
                "test/positive_rate": test_metrics["positive_rate"],
                "test/target_positive_rate": test_metrics["target_positive_rate"],
                "test/auroc": test_metrics["auroc"],
                "test/auprc": test_metrics["auprc"],
            },
            step=max(args.epochs, best_epoch),
        )
        wandb_run.finish()

    print(
        f"[{dataset_name}] best_epoch={best_epoch} "
        f"test_loss={test_metrics['loss']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_auroc={test_metrics['auroc'] if test_metrics['auroc'] is not None else 'n/a'}"
    )
    print(f"[{dataset_name}] saved checkpoint: {best_checkpoint_path}")
    print(f"[{dataset_name}] saved metrics   : {output_dir / 'test_metrics.json'}")
    train_time = float(sum(float(entry["seconds"]) for entry in history))
    csv_path = append_result_row(
        {
            "experiment": "exp3",
            "script_path": "experiments/exp3_unet_classifier.py",
            "artifacts": json.dumps(
                [
                    str(best_checkpoint_path),
                    str(output_dir / "history.json"),
                    str(output_dir / "test_metrics.json"),
                    str(output_dir / "test_predictions.csv"),
                ]
            ),
            "dataset": dataset_name,
            "seed": args.seed,
            "method_name": "unet_classifier",
            "model_architecture": "unet",
            "base_channels": args.base_channels,
            "image_size": args.image_size,
            "num_workers": args.num_workers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": str(device),
            "n_train": len(split_records["train"]),
            "n_val": len(split_records["valid"]),
            "n_test": len(split_records["test"]),
            "fit_or_train_time_sec": train_time,
            "total_pipeline_time_sec": train_time,
            **test_metrics,
        }
    )
    print(f"[{dataset_name}] appended CSV metrics: {csv_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    split_records = build_dataset_bundle(args.dataset, args.data_root, args)
    train_dataset(args.dataset, split_records, args, device)


if __name__ == "__main__":
    main()
