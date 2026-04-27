"""
Experiment 5: ViT-Small classifier on PCam and CAMELYON17.

Trains a ViT-Small (patch=16, embed_dim=384, depth=12, heads=6) image-level
binary classifier on:
- ``datasets/pcam-hg/{train,valid,test}``
- ``datasets/camelyon17-hg/{train,valid,test}``

Random patch masking is available as an optional training-time knob via
``--mask-ratio`` but is disabled by default.

Each split is expected to contain:
- images/
- labels.csv

After training, the best-validation checkpoint is reloaded and evaluated on the
test split; metrics, per-epoch history, predictions, and the checkpoint are
written under ``data/exp5_vit/<dataset>/``. Pass ``--wandb`` to log
training/validation/test metrics to Weights & Biases. Pass
``--dataset all`` to train and test on PCam followed by CAMELYON17 in one run.
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
from my_cholesky.result_logging import append_result_row
from predictive_metrics import evaluate_binary_probabilistic_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ViT-Small classifier with random patch masking on pcam-hg."
    )
    parser.add_argument(
        "--dataset",
        choices=["pcam", "camelyon17", "all"],
        default="pcam",
        help="Which dataset export to train on. Use 'all' to run pcam followed by camelyon17.",
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
        help="Directory to save checkpoints, histories, and metrics.",
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
        "--patch-size",
        type=int,
        default=16,
        help="ViT patch size. Must divide --image-size.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=384,
        help="Transformer embedding dimension (ViT-Small default 384).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=12,
        help="Number of transformer blocks (ViT-Small default 12).",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=6,
        help="Number of attention heads (ViT-Small default 6).",
    )
    parser.add_argument(
        "--mlp-ratio", type=float, default=4.0, help="MLP hidden ratio in transformer blocks."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout in MLP/positional path."
    )
    parser.add_argument(
        "--attn-dropout", type=float, default=0.0, help="Attention dropout."
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
        help="Enable Weights & Biases logging for Experiment 5.",
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
        help="Optional W&B group name.",
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
    if isinstance(value, Path):
        return str(value)
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

    if args.wandb_run_name:
        run_name = f"{args.wandb_run_name}-{dataset_name}"
    elif args.mask_ratio > 0.0:
        run_name = f"exp5-vit-mask{args.mask_ratio:g}-{dataset_name}"
    else:
        run_name = f"exp5-vit-{dataset_name}"
    architecture_label = (
        f"ViT-Small (random patch masking, mask_ratio={args.mask_ratio:g})"
        if args.mask_ratio > 0.0
        else "ViT-Small"
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
    """Image + binary label dataset backed by labels.csv (matches exp3 format)."""

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

            logits = model(images, apply_masking=False)
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
) -> dict[str, Any]:
    output_dir = args.output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    masking_label = (
        f"random masking (ratio={args.mask_ratio:g})"
        if args.mask_ratio > 0.0
        else "masking disabled"
    )
    print("")
    print("==============================================================================")
    print(f"Experiment 5: ViT-Small on {dataset_name} ({masking_label})")
    print("==============================================================================")
    print(f"Output dir      : {output_dir}")
    print(f"Train samples   : {len(split_records['train'])}")
    print(f"Valid samples   : {len(split_records['valid'])}")
    print(f"Test samples    : {len(split_records['test'])}")
    print(f"Image size      : {args.image_size}  patch size: {args.patch_size}")
    print(f"Embed/depth/heads: {args.embed_dim}/{args.depth}/{args.num_heads}")
    print(f"Mask ratio      : {args.mask_ratio}")
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
    pos_weight = compute_pos_weight(split_records["train"]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
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
            "experiment": "exp5",
            "script_path": "experiments/exp5/exp5_vit_small_random_masking.py",
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
            "method_name": "vit_small_random_mask",
            "model_architecture": "vit_small",
            "patch_size": args.patch_size,
            "embed_dim": args.embed_dim,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "mlp_ratio": args.mlp_ratio,
            "mask_ratio": args.mask_ratio,
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

    return {
        "dataset": dataset_name,
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(best_checkpoint_path),
        "n_train": len(split_records["train"]),
        "n_valid": len(split_records["valid"]),
        "n_test": len(split_records["test"]),
        "train_time_sec": train_time,
        "test_metrics": test_metrics,
    }


SUMMARY_METRIC_KEYS: list[str] = [
    "loss",
    "accuracy",
    "auroc",
    "auprc",
    "brier",
    "ece",
    "log_likelihood_mean",
    "negative_log_likelihood_mean",
    "elpd",
    "elpd_mean",
    "predictive_likelihood",
    "precision",
    "recall",
    "sensitivity_TPR",
    "specificity_TNR",
    "FPR",
    "FNR",
    "positive_rate",
    "target_positive_rate",
    "number_errors",
    "TP",
    "FP",
    "TN",
    "FN",
]


def _format_metric_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not math.isfinite(v):
            return "n/a"
        return f"{v:.6f}"
    return str(value)


def print_test_summary(results: list[dict[str, Any]]) -> None:
    print("")
    print("==============================================================================")
    print("Experiment 5 — Test set summary (best-validation checkpoint per dataset)")
    print("==============================================================================")
    for result in results:
        ds = result["dataset"]
        m = result["test_metrics"]
        print(
            f"\n[{ds}] best_epoch={result['best_epoch']}  "
            f"n_train={result['n_train']}  n_valid={result['n_valid']}  n_test={result['n_test']}  "
            f"train_time_sec={result['train_time_sec']:.1f}"
        )
        print(f"  checkpoint: {result['best_checkpoint_path']}")
        for key in SUMMARY_METRIC_KEYS:
            if key in m:
                print(f"  {key:32s}: {_format_metric_value(m[key])}")
        extra_keys = sorted(set(m.keys()) - set(SUMMARY_METRIC_KEYS))
        if extra_keys:
            print("  --- additional metrics ---")
            for key in extra_keys:
                print(f"  {key:32s}: {_format_metric_value(m[key])}")
    print("==============================================================================")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.dataset == "all":
        dataset_names = ["pcam", "camelyon17"]
    else:
        dataset_names = [args.dataset]

    results: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        split_records = build_dataset_bundle(dataset_name, args.data_root, args)
        result = train_dataset(dataset_name, split_records, args, device)
        results.append(result)

    print_test_summary(results)

    args.output_root.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_root / "summary.json",
        {
            "datasets": [r["dataset"] for r in results],
            "results": [
                {
                    "dataset": r["dataset"],
                    "best_epoch": r["best_epoch"],
                    "best_checkpoint_path": r["best_checkpoint_path"],
                    "n_train": r["n_train"],
                    "n_valid": r["n_valid"],
                    "n_test": r["n_test"],
                    "train_time_sec": r["train_time_sec"],
                    "test_metrics": r["test_metrics"],
                }
                for r in results
            ],
        },
    )
    print(f"Saved consolidated summary: {args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
