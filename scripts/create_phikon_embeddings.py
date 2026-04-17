#!/usr/bin/env python3
"""
Create Phikon embeddings for exported pathology image datasets.

Expected split layout:
    <dataset-root>/<split>/
        images/
        labels.csv

This script adds:
    <dataset-root>/<split>/
        embeddings/
            embeddings.npy
            metadata.json
            progress.json

Row alignment guarantee:
    embeddings.npy[i] corresponds to row i+1 in labels.csv (excluding header).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image
from tqdm.auto import tqdm

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Phikon image embeddings for PCam and CAMELYON17 Hugging Face "
            "exports and save them beside each split."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["pcam", "camelyon17", "all"],
        required=True,
        help="Which dataset export to process.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "datasets",
        help="Directory containing pcam-hg and camelyon17-hg.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="owkin/phikon",
        help="Hugging Face model name to use for feature extraction.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of images to encode per forward pass.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Storage dtype for embeddings.npy.",
    )
    parser.add_argument(
        "--feature-pooling",
        choices=["cls", "mean"],
        default="cls",
        help="How to reduce token features into one vector per image.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "valid", "test"],
        default=["train", "valid", "test"],
        help="Which splits to process.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory override.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate existing embedding outputs from scratch.",
    )
    parser.add_argument(
        "--project-dim",
        type=int,
        default=None,
        help=(
            "Optional projected embedding size. If set, fit a linear PCA "
            "projection on the train split and apply it to train/valid/test."
        ),
    )
    parser.add_argument(
        "--projection-batch-size",
        type=int,
        default=2048,
        help="Batch size to use while fitting and applying the PCA projection.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def json_default(value: Any) -> Any:
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
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def dataset_root_name(dataset_name: str) -> str:
    return {"pcam": "pcam-hg", "camelyon17": "camelyon17-hg"}[dataset_name]


def selected_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return ["pcam", "camelyon17"]
    return [dataset_arg]


def load_split_records(split_dir: Path) -> list[dict[str, Any]]:
    labels_path = split_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Expected labels.csv at {labels_path}")

    records: list[dict[str, Any]] = []
    with labels_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "image" not in (reader.fieldnames or []):
            raise ValueError(f"{labels_path} is missing an 'image' column.")
        for row_index, row in enumerate(reader):
            image_rel = row["image"]
            image_path = split_dir / image_rel
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Expected image for row {row_index + 1} at {image_path}"
                )
            records.append(
                {
                    "row_index": row_index,
                    "image_rel": image_rel,
                    "image_path": image_path,
                }
            )

    if not records:
        raise RuntimeError(f"No rows found in {labels_path}")
    return records


def load_processor_and_model(model_name: str, cache_dir: Path | None):
    try:
        from transformers import AutoImageProcessor, ViTModel
    except Exception as exc:
        raise RuntimeError(
            "Failed to import the Hugging Face vision stack. Make sure "
            "`transformers`, `torch`, and `torchvision` are installed in the same "
            "environment and have compatible versions."
        ) from exc

    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = ViTModel.from_pretrained(
        model_name,
        add_pooling_layer=False,
        cache_dir=cache_dir,
    )
    return processor, model


def get_embedding_dtype(dtype_name: str) -> np.dtype:
    return np.float16 if dtype_name == "float16" else np.float32


def embedding_paths(split_dir: Path) -> dict[str, Path]:
    embedding_dir = split_dir / "embeddings"
    return {
        "dir": embedding_dir,
        "array": embedding_dir / "embeddings.npy",
        "metadata": embedding_dir / "metadata.json",
        "progress": embedding_dir / "progress.json",
    }


def projected_embedding_paths(split_dir: Path, projected_dim: int) -> dict[str, Path]:
    embedding_dir = split_dir / "embeddings"
    prefix = f"projected_{projected_dim}"
    return {
        "array": embedding_dir / f"{prefix}.npy",
        "metadata": embedding_dir / f"{prefix}_metadata.json",
        "progress": embedding_dir / f"{prefix}_progress.json",
    }


def projection_paths(dataset_root: Path, projected_dim: int) -> dict[str, Path]:
    projection_dir = dataset_root / "embedding_projection"
    projection_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"pca_{projected_dim}"
    return {
        "dir": projection_dir,
        "matrix": projection_dir / f"{prefix}.npz",
        "metadata": projection_dir / f"{prefix}_metadata.json",
    }


def validate_or_write_metadata(
    metadata_path: Path,
    payload: dict[str, Any],
    overwrite: bool,
) -> None:
    if metadata_path.exists() and not overwrite:
        existing = load_json(metadata_path)
        for key in [
            "model_name",
            "feature_pooling",
            "embedding_dim",
            "dtype",
            "rows_total",
            "alignment",
        ]:
            if existing.get(key) != payload.get(key):
                raise RuntimeError(
                    f"Existing metadata at {metadata_path} does not match current "
                    f"run for key '{key}'. Use --overwrite to rebuild."
                )
        return
    save_json(metadata_path, payload)


def prepare_embedding_array(
    array_path: Path,
    rows_total: int,
    embedding_dim: int,
    dtype: np.dtype,
    overwrite: bool,
) -> np.memmap:
    mode = "r+"
    if overwrite or not array_path.exists():
        mode = "w+"
    return open_memmap(
        array_path,
        mode=mode,
        dtype=dtype,
        shape=(rows_total, embedding_dim),
    )


def initial_completed_rows(
    progress_path: Path,
    rows_total: int,
    overwrite: bool,
) -> int:
    if overwrite or not progress_path.exists():
        return 0
    progress = load_json(progress_path)
    completed_rows = int(progress.get("rows_completed", 0))
    return max(0, min(completed_rows, rows_total))


def inspect_existing_base_embeddings(
    split_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    records = load_split_records(split_dir)
    rows_total = len(records)
    paths = embedding_paths(split_dir)

    status: dict[str, Any] = {
        "rows_total": rows_total,
        "ready": False,
        "embedding_dim": None,
        "paths": paths,
    }
    if args.overwrite:
        return status
    if not all(paths[key].exists() for key in ["array", "metadata", "progress"]):
        return status

    metadata = load_json(paths["metadata"])
    completed_rows = initial_completed_rows(
        paths["progress"], rows_total=rows_total, overwrite=False
    )
    try:
        array = np.load(paths["array"], mmap_mode="r")
    except Exception:
        return status

    metadata_matches = (
        metadata.get("model_name") == args.model_name
        and metadata.get("feature_pooling") == args.feature_pooling
        and metadata.get("dtype") == args.dtype
        and int(metadata.get("rows_total", -1)) == rows_total
    )
    shape_matches = (
        getattr(array, "ndim", 0) == 2
        and int(array.shape[0]) == rows_total
        and int(array.shape[1]) > 0
    )
    ready = metadata_matches and shape_matches and completed_rows >= rows_total

    status["ready"] = ready
    status["embedding_dim"] = int(array.shape[1]) if shape_matches else None
    return status


def save_progress(
    progress_path: Path,
    *,
    rows_completed: int,
    rows_total: int,
    dataset_name: str,
    split_name: str,
    model_name: str,
) -> None:
    save_json(
        progress_path,
        {
            "dataset": dataset_name,
            "split": split_name,
            "model_name": model_name,
            "rows_completed": rows_completed,
            "rows_total": rows_total,
        },
    )


def load_batch_images(records: list[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for record in records:
        with Image.open(record["image_path"]) as image:
            images.append(image.convert("RGB"))
    return images


def extract_batch_embeddings(
    *,
    images: list[Image.Image],
    processor,
    model,
    device: torch.device,
    feature_pooling: str,
) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        token_features = outputs.last_hidden_state
        if feature_pooling == "cls":
            batch_embeddings = token_features[:, 0, :]
        else:
            if token_features.shape[1] > 1:
                batch_embeddings = token_features[:, 1:, :].mean(dim=1)
            else:
                batch_embeddings = token_features[:, 0, :]
    return batch_embeddings.detach().cpu().numpy()


def process_split(
    *,
    dataset_name: str,
    split_name: str,
    split_dir: Path,
    processor,
    model,
    embedding_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    records = load_split_records(split_dir)
    paths = embedding_paths(split_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset": dataset_name,
        "split": split_name,
        "model_name": args.model_name,
        "feature_pooling": args.feature_pooling,
        "embedding_dim": embedding_dim,
        "dtype": args.dtype,
        "rows_total": len(records),
        "alignment": "embeddings.npy row i matches labels.csv data row i (0-indexed).",
        "labels_file": "labels.csv",
        "image_column": "image",
    }
    validate_or_write_metadata(paths["metadata"], metadata, overwrite=args.overwrite)

    embeddings = prepare_embedding_array(
        paths["array"],
        rows_total=len(records),
        embedding_dim=embedding_dim,
        dtype=get_embedding_dtype(args.dtype),
        overwrite=args.overwrite,
    )
    completed_rows = initial_completed_rows(
        paths["progress"], rows_total=len(records), overwrite=args.overwrite
    )

    if completed_rows >= len(records):
        print(
            f"[{dataset_name}/{split_name}] embeddings already complete at {paths['array']}"
        )
        return

    if completed_rows > 0:
        print(
            f"[{dataset_name}/{split_name}] resuming from row {completed_rows + 1} / {len(records)}"
        )
    else:
        print(f"[{dataset_name}/{split_name}] starting from row 1 / {len(records)}")

    progress = tqdm(
        total=len(records),
        initial=completed_rows,
        desc=f"{dataset_name}:{split_name}",
        unit="img",
    )

    for start in range(completed_rows, len(records), args.batch_size):
        stop = min(start + args.batch_size, len(records))
        batch_records = records[start:stop]
        batch_images = load_batch_images(batch_records)
        batch_embeddings = extract_batch_embeddings(
            images=batch_images,
            processor=processor,
            model=model,
            device=device,
            feature_pooling=args.feature_pooling,
        ).astype(get_embedding_dtype(args.dtype), copy=False)

        embeddings[start:stop] = batch_embeddings
        embeddings.flush()

        save_progress(
            paths["progress"],
            rows_completed=stop,
            rows_total=len(records),
            dataset_name=dataset_name,
            split_name=split_name,
            model_name=args.model_name,
        )
        progress.update(stop - start)
        progress.set_postfix(saved=f"{stop}/{len(records)}")

    progress.close()
    print(f"[{dataset_name}/{split_name}] saved embeddings to {paths['array']}")


def fit_or_load_projection(
    *,
    dataset_name: str,
    dataset_root: Path,
    train_array_path: Path,
    args: argparse.Namespace,
) -> dict[str, np.ndarray | int | str]:
    if args.project_dim is None:
        raise ValueError("project_dim must be set before fitting a projection.")

    paths = projection_paths(dataset_root, args.project_dim)
    if paths["matrix"].exists() and paths["metadata"].exists() and not args.overwrite:
        metadata = load_json(paths["metadata"])
        projection = np.load(paths["matrix"])
        return {
            "method": metadata["method"],
            "fitted_on_split": metadata["fitted_on_split"],
            "source_dim": int(metadata["source_dim"]),
            "projected_dim": int(metadata["projected_dim"]),
            "components": projection["components"],
            "mean": projection["mean"],
            "explained_variance_ratio": projection["explained_variance_ratio"],
        }

    from sklearn.decomposition import IncrementalPCA

    train_embeddings = np.load(train_array_path, mmap_mode="r")
    source_dim = int(train_embeddings.shape[1])
    if args.project_dim <= 0 or args.project_dim > source_dim:
        raise ValueError(
            f"--project-dim must be between 1 and {source_dim}, got {args.project_dim}."
        )

    print(
        f"[{dataset_name}] fitting PCA projection {source_dim} -> {args.project_dim} "
        f"using train split embeddings"
    )
    pca = IncrementalPCA(
        n_components=args.project_dim,
        batch_size=max(args.projection_batch_size, args.project_dim),
    )
    fit_batch_size = max(args.projection_batch_size, args.project_dim)
    total_rows = int(train_embeddings.shape[0])
    if total_rows < args.project_dim:
        raise ValueError(
            f"Train split has only {total_rows} rows, which is fewer than "
            f"--project-dim={args.project_dim}."
        )

    progress = tqdm(
        total=total_rows,
        desc=f"{dataset_name}:fit-pca",
        unit="row",
    )
    start = 0
    while start < total_rows:
        stop = min(start + fit_batch_size, total_rows)
        remaining = total_rows - stop
        if 0 < remaining < args.project_dim:
            stop = total_rows
        batch = np.asarray(train_embeddings[start:stop], dtype=np.float32)
        pca.partial_fit(batch)
        progress.update(stop - start)
        start = stop
    progress.close()

    np.savez(
        paths["matrix"],
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )
    save_json(
        paths["metadata"],
        {
            "dataset": dataset_name,
            "method": "pca",
            "fitted_on_split": "train",
            "source_dim": source_dim,
            "projected_dim": args.project_dim,
            "projection_batch_size": args.projection_batch_size,
        },
    )
    return {
        "method": "pca",
        "fitted_on_split": "train",
        "source_dim": source_dim,
        "projected_dim": args.project_dim,
        "components": pca.components_.astype(np.float32),
        "mean": pca.mean_.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
    }


def apply_projection_to_split(
    *,
    dataset_name: str,
    split_name: str,
    split_dir: Path,
    projection: dict[str, np.ndarray | int | str],
    args: argparse.Namespace,
) -> None:
    projected_dim = int(projection["projected_dim"])
    split_paths = embedding_paths(split_dir)
    if not split_paths["array"].exists():
        raise FileNotFoundError(
            f"Expected base embeddings at {split_paths['array']} before projection."
        )

    base_embeddings = np.load(split_paths["array"], mmap_mode="r")
    output_paths = projected_embedding_paths(split_dir, projected_dim)
    metadata_payload = {
        "dataset": dataset_name,
        "split": split_name,
        "method": projection["method"],
        "source_array": str(split_paths["array"].name),
        "source_dim": int(projection["source_dim"]),
        "projected_dim": projected_dim,
        "dtype": args.dtype,
        "rows_total": int(base_embeddings.shape[0]),
        "fitted_on_split": projection["fitted_on_split"],
        "alignment": (
            f"{output_paths['array'].name} row i matches embeddings.npy row i and "
            "labels.csv data row i (0-indexed)."
        ),
    }
    validate_or_write_metadata(
        output_paths["metadata"], metadata_payload, overwrite=args.overwrite
    )

    projected_embeddings = prepare_embedding_array(
        output_paths["array"],
        rows_total=int(base_embeddings.shape[0]),
        embedding_dim=projected_dim,
        dtype=get_embedding_dtype(args.dtype),
        overwrite=args.overwrite,
    )
    completed_rows = initial_completed_rows(
        output_paths["progress"],
        rows_total=int(base_embeddings.shape[0]),
        overwrite=args.overwrite,
    )

    if completed_rows >= int(base_embeddings.shape[0]):
        print(
            f"[{dataset_name}/{split_name}] projected embeddings already complete at "
            f"{output_paths['array']}"
        )
        return

    if completed_rows > 0:
        print(
            f"[{dataset_name}/{split_name}] resuming projected embeddings from row "
            f"{completed_rows + 1} / {int(base_embeddings.shape[0])}"
        )
    else:
        print(
            f"[{dataset_name}/{split_name}] creating projected embeddings "
            f"{int(projection['source_dim'])} -> {projected_dim}"
        )

    components = np.asarray(projection["components"], dtype=np.float32)
    mean = np.asarray(projection["mean"], dtype=np.float32)

    progress = tqdm(
        total=int(base_embeddings.shape[0]),
        initial=completed_rows,
        desc=f"{dataset_name}:{split_name}:project",
        unit="row",
    )
    for start in range(completed_rows, int(base_embeddings.shape[0]), args.projection_batch_size):
        stop = min(
            start + args.projection_batch_size, int(base_embeddings.shape[0])
        )
        batch = np.asarray(base_embeddings[start:stop], dtype=np.float32)
        projected_batch = (batch - mean) @ components.T
        projected_embeddings[start:stop] = projected_batch.astype(
            get_embedding_dtype(args.dtype), copy=False
        )
        projected_embeddings.flush()
        save_progress(
            output_paths["progress"],
            rows_completed=stop,
            rows_total=int(base_embeddings.shape[0]),
            dataset_name=dataset_name,
            split_name=split_name,
            model_name=f"{args.model_name}+{projection['method']}_{projected_dim}",
        )
        progress.update(stop - start)
        progress.set_postfix(saved=f"{stop}/{int(base_embeddings.shape[0])}")
    progress.close()

    print(
        f"[{dataset_name}/{split_name}] saved projected embeddings to "
        f"{output_paths['array']}"
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    base_statuses: dict[str, dict[str, dict[str, Any]]] = {}
    selected = selected_datasets(args.dataset)
    need_base_embeddings = False
    known_embedding_dims: list[int] = []

    for dataset_name in selected:
        dataset_root = args.data_root / dataset_root_name(dataset_name)
        if not dataset_root.exists():
            raise FileNotFoundError(f"Expected dataset directory at {dataset_root}")
        base_statuses[dataset_name] = {}
        for split_name in args.splits:
            status = inspect_existing_base_embeddings(dataset_root / split_name, args)
            base_statuses[dataset_name][split_name] = status
            if status["ready"]:
                known_embedding_dims.append(int(status["embedding_dim"]))
            else:
                need_base_embeddings = True

    processor = None
    model = None
    if len(set(known_embedding_dims)) > 1:
        raise RuntimeError(
            f"Found inconsistent existing embedding dimensions: {sorted(set(known_embedding_dims))}"
        )
    embedding_dim = known_embedding_dims[0] if known_embedding_dims else None
    if need_base_embeddings:
        processor, model = load_processor_and_model(args.model_name, args.cache_dir)
        model = model.to(device)
        model.eval()
        embedding_dim = int(model.config.hidden_size)
    elif embedding_dim is None:
        raise RuntimeError("Could not determine the existing embedding dimension.")

    print("==============================================================================")
    print("Phikon Embedding Extraction")
    print("==============================================================================")
    print(f"Model           : {args.model_name}")
    print(f"Feature pooling : {args.feature_pooling}")
    print(f"Embedding dim   : {embedding_dim}")
    print(f"Storage dtype   : {args.dtype}")
    print(
        "Projected dim   : "
        f"{args.project_dim if args.project_dim is not None else 'disabled'}"
    )
    print(f"Device          : {device}")
    print(f"Reuse existing  : {'yes' if not need_base_embeddings else 'partial/no'}")
    print("==============================================================================")

    for dataset_name in selected:
        dataset_root = args.data_root / dataset_root_name(dataset_name)
        print("")
        print(f"[{dataset_name}] dataset root: {dataset_root}")
        for split_name in args.splits:
            status = base_statuses[dataset_name][split_name]
            if status["ready"]:
                print(
                    f"[{dataset_name}/{split_name}] reusing existing embeddings at "
                    f"{status['paths']['array']}"
                )
            else:
                process_split(
                    dataset_name=dataset_name,
                    split_name=split_name,
                    split_dir=dataset_root / split_name,
                    processor=processor,
                    model=model,
                    embedding_dim=embedding_dim,
                    args=args,
                    device=device,
                )
        if args.project_dim is not None:
            projection = fit_or_load_projection(
                dataset_name=dataset_name,
                dataset_root=dataset_root,
                train_array_path=embedding_paths(dataset_root / "train")["array"],
                args=args,
            )
            for split_name in args.splits:
                apply_projection_to_split(
                    dataset_name=dataset_name,
                    split_name=split_name,
                    split_dir=dataset_root / split_name,
                    projection=projection,
                    args=args,
                )


if __name__ == "__main__":
    main()
