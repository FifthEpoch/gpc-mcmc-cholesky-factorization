"""
Embedding extraction for Experiments 1--3.

Passes raw images through a frozen pretrained encoder and saves the resulting
feature vectors as .npy files.  Supported encoders:

  - densenet121  : DenseNet-121 (ImageNet, 1024-dim)   [WILDS default]
  - dinov2_vitl14: DINOv2 ViT-L/14 (1024-dim)          [optional]

Supported datasets: pcam, camelyon17, embed.

Usage:
    python experiments/extract_embeddings.py \
        --dataset pcam \
        --data-root datasets \
        --encoder densenet121 \
        --batch-size 256 \
        --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


ENCODERS = {
    "densenet121": {
        "dim": 1024,
        "input_size": 224,
    },
    "dinov2_vitl14": {
        "dim": 1024,
        "input_size": 224,
    },
}


def build_encoder(name: str, device: torch.device) -> tuple[nn.Module, int, int]:
    """Return (model, feature_dim, input_size) with all params frozen."""
    cfg = ENCODERS[name]

    if name == "densenet121":
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        base.classifier = nn.Identity()
        model = base

    elif name == "dinov2_vitl14":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")

    else:
        raise ValueError(f"Unknown encoder: {name}")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, cfg["dim"], cfg["input_size"]


def get_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class NumpyImageDataset(Dataset):
    """Dataset wrapper that applies transforms per image on demand."""

    def __init__(self, images: np.ndarray, transform: transforms.Compose):
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float() / 255.0
        return self.transform(img)


@torch.no_grad()
def extract(
    model: nn.Module,
    images: np.ndarray,
    transform: transforms.Compose,
    batch_size: int,
    device: torch.device,
    feat_dim: int,
) -> np.ndarray:
    """Run *images* (N, H, W, C uint8) through *model* and return (N, D) features."""
    # Apply transforms lazily per image to avoid allocating a huge N x C x H x W tensor.
    dataset = NumpyImageDataset(images, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_feats = np.empty((len(images), feat_dim), dtype=np.float32)
    idx = 0
    for batch in tqdm(loader, desc="Extracting", unit="batch"):
        batch = batch.to(device)
        feats = model(batch)
        if feats.dim() > 2:
            feats = feats.mean(dim=(-2, -1))
        bs = feats.shape[0]
        all_feats[idx : idx + bs] = feats.cpu().numpy()
        idx += bs

    return all_feats


HF_PCAM_REPO = "1aurent/PatchCamelyon"

_PCAM_SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}


def load_pcam_split(data_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a PCam split from the HuggingFace mirror (1aurent/PatchCamelyon)."""
    from datasets import load_dataset

    hf_split = _PCAM_SPLIT_MAP.get(split, split)
    cache_dir = str(data_root / "pcam" / "hf_cache")
    print(f"[PCam] Loading {hf_split} split from HuggingFace ({HF_PCAM_REPO}) ...")

    ds = load_dataset(HF_PCAM_REPO, split=hf_split, cache_dir=cache_dir)

    images_list = []
    labels_list = []

    for row in tqdm(ds, desc=f"Loading PCam {split}", unit="img"):
        img = row["image"]
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        images_list.append(np.array(img, dtype=np.uint8))
        labels_list.append(int(row["label"]))

    images = np.stack(images_list)
    labels = np.array(labels_list, dtype=np.int64)
    return images, labels


HF_CAMELYON17_REPO = "wltjr1007/Camelyon17-WILDS"

_HF_SPLIT_MAP = {"train": "train", "val": "validation", "test": "test"}


def load_camelyon17_split(
    data_root: Path, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a CAMELYON17-WILDS split from the HuggingFace mirror.

    Returns (images, labels, hospital_ids).  The original CodaLab source
    used by the ``wilds`` library has been broken since June 2025, so we
    load from ``wltjr1007/Camelyon17-WILDS`` on HuggingFace instead.
    """
    from datasets import load_dataset

    hf_split = _HF_SPLIT_MAP.get(split, split)
    cache_dir = str(data_root / "camelyon17_wilds" / "hf_cache")
    print(f"[CAMELYON17] Loading {hf_split} split from HuggingFace ({HF_CAMELYON17_REPO}) ...")

    ds = load_dataset(HF_CAMELYON17_REPO, split=hf_split, cache_dir=cache_dir)

    images_list = []
    labels_list = []
    hospitals_list = []

    has_hospital = "center" in ds.column_names or "hospital" in ds.column_names
    hospital_col = "center" if "center" in ds.column_names else "hospital"

    for row in tqdm(ds, desc=f"Loading CAMELYON17 {split}", unit="img"):
        img = row["image"]
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        images_list.append(np.array(img, dtype=np.uint8))
        labels_list.append(int(row["label"]))
        if has_hospital:
            hospitals_list.append(int(row[hospital_col]))
        else:
            hospitals_list.append(0)

    images = np.stack(images_list)
    labels = np.array(labels_list, dtype=np.int64)
    hospitals = np.array(hospitals_list, dtype=np.int64)
    return images, labels, hospitals


def run_pcam(args: argparse.Namespace, model, transform, feat_dim, device) -> None:
    out_dir = Path(args.output_dir)
    for split in ["train", "val", "test"]:
        emb_path = out_dir / f"pcam_{split}_embeddings.npy"
        lbl_path = out_dir / f"pcam_{split}_labels.npy"
        if emb_path.exists() and lbl_path.exists() and not args.overwrite:
            print(f"[PCam] Skipping {split} (already exists). Use --overwrite to redo.")
            continue

        images, labels = load_pcam_split(Path(args.data_root), split)
        print(f"[PCam] {split}: {images.shape[0]} images")
        embeddings = extract(model, images, transform, args.batch_size, device, feat_dim)
        np.save(emb_path, embeddings)
        np.save(lbl_path, labels)
        print(f"[PCam] Saved {emb_path}  ({embeddings.shape})")


def run_camelyon17(args: argparse.Namespace, model, transform, feat_dim, device) -> None:
    out_dir = Path(args.output_dir)
    for split in ["train", "val", "test"]:
        emb_path = out_dir / f"camelyon17_{split}_embeddings.npy"
        lbl_path = out_dir / f"camelyon17_{split}_labels.npy"
        meta_path = out_dir / f"camelyon17_{split}_hospitals.npy"
        if emb_path.exists() and lbl_path.exists() and not args.overwrite:
            print(f"[CAMELYON17] Skipping {split} (already exists). Use --overwrite to redo.")
            continue

        images, labels, hospitals = load_camelyon17_split(Path(args.data_root), split)
        print(f"[CAMELYON17] {split}: {images.shape[0]} images")
        embeddings = extract(model, images, transform, args.batch_size, device, feat_dim)
        np.save(emb_path, embeddings)
        np.save(lbl_path, labels)
        np.save(meta_path, hospitals)
        print(f"[CAMELYON17] Saved {emb_path}  ({embeddings.shape})")


def run_embed(args: argparse.Namespace, model, transform, feat_dim, device) -> None:
    embed_dir = Path(args.data_root) / "embed"
    if not embed_dir.exists():
        print("[EMBED] Dataset directory not found.")
        print("[EMBED] Access must be requested: https://forms.gle/6YVFKTz7ucEJKEWw8")
        print("[EMBED] After approval, download and rerun.")
        return
    print("[EMBED] EMBED extraction is dataset-specific and depends on your approved data layout.")
    print("[EMBED] Implement the DICOM/PNG loading logic for your institution's copy here.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen-encoder embeddings.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["pcam", "camelyon17", "embed"],
    )
    parser.add_argument("--data-root", type=str, default="datasets")
    parser.add_argument("--output-dir", type=str, default="data/embeddings")
    parser.add_argument(
        "--encoder",
        type=str,
        default="densenet121",
        choices=list(ENCODERS.keys()),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Encoder: {args.encoder}  |  Device: {device}")

    model, feat_dim, input_size = build_encoder(args.encoder, device)
    transform = get_transform(input_size)

    if args.dataset == "pcam":
        run_pcam(args, model, transform, feat_dim, device)
    elif args.dataset == "camelyon17":
        run_camelyon17(args, model, transform, feat_dim, device)
    elif args.dataset == "embed":
        run_embed(args, model, transform, feat_dim, device)


if __name__ == "__main__":
    main()
