#!/usr/bin/env python3
"""
Create aligned label arrays for exported pathology image datasets.

Expected split layout:
    <dataset-root>/<split>/
        images/
        labels.csv

This script adds:
    <dataset-root>/<split>/
        embeddings/
            y_embeddings.npy
            y_embeddings_metadata.json

Row alignment guarantee:
    y_embeddings.npy[i, 0] corresponds to row i+1 in labels.csv (excluding
    the header) and to the image path from that same labels.csv row.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract aligned label arrays from labels.csv for PCam and "
            "CAMELYON17 Hugging Face exports."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["pcam", "pcam-hg", "camelyon17", "camelyon17-hg", "all"],
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
        "--splits",
        nargs="+",
        choices=["train", "valid", "test"],
        default=["train", "valid", "test"],
        help="Which splits to process.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Column in labels.csv to store in y_embeddings.npy.",
    )
    parser.add_argument(
        "--dtype",
        choices=["int64", "int32", "float32"],
        default="int64",
        help="Storage dtype for y_embeddings.npy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate existing outputs from scratch.",
    )
    return parser.parse_args()


def dataset_root_name(dataset_name: str) -> str:
    return {
        "pcam": "pcam-hg",
        "pcam-hg": "pcam-hg",
        "camelyon17": "camelyon17-hg-2",
        "camelyon17-hg": "camelyon17-hg-2",
    }[dataset_name]


def selected_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return ["pcam-hg", "camelyon17-hg"]
    return [dataset_arg]


def output_paths(split_dir: Path) -> dict[str, Path]:
    embedding_dir = split_dir / "embeddings"
    return {
        "dir": embedding_dir,
        "array": embedding_dir / "y_embeddings.npy",
        "metadata": embedding_dir / "y_embeddings_metadata.json",
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def maybe_parse_label(value: str, dtype_name: str) -> int | float:
    if dtype_name == "float32":
        return float(value)
    return int(value)


def label_dtype(dtype_name: str) -> np.dtype:
    return {
        "int64": np.int64,
        "int32": np.int32,
        "float32": np.float32,
    }[dtype_name]


def count_label_rows(labels_path: Path) -> int:
    with labels_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def load_labels(
    split_dir: Path,
    *,
    dataset_name: str,
    split_name: str,
    label_column: str,
    dtype_name: str,
) -> np.ndarray:
    labels_path = split_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Expected labels.csv at {labels_path}")

    total_rows = count_label_rows(labels_path)
    labels: list[int | float] = []
    repaired_paths = 0
    trimmed_at_row: int | None = None
    progress = tqdm(
        total=total_rows,
        desc=f"{dataset_name}:{split_name}:labels",
        unit="row",
        file=sys.stdout,
        dynamic_ncols=True,
    )
    with labels_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "image" not in fieldnames:
            raise ValueError(f"{labels_path} is missing an 'image' column.")
        if label_column not in fieldnames:
            raise ValueError(
                f"{labels_path} is missing the requested '{label_column}' column."
            )

        for row_index, row in enumerate(reader):
            image_rel = row["image"]
            image_path = split_dir / image_rel
            if not image_path.exists():
                fallback_rel = f"images/{row_index + 1}.png"
                fallback_path = split_dir / fallback_rel
                if fallback_path.exists():
                    repaired_paths += 1
                else:
                    trimmed_at_row = row_index + 1
                    print(
                        f"[load_labels] WARNING: missing image for row "
                        f"{trimmed_at_row} at {image_path}; using only the first "
                        f"{row_index} contiguous rows from {labels_path}"
                    )
                    break
            labels.append(maybe_parse_label(row[label_column], dtype_name))
            progress.update(1)
            progress.set_postfix(saved=f"{len(labels)}/{total_rows}")
    progress.close()

    if not labels:
        raise RuntimeError(f"No rows found in {labels_path}")
    if repaired_paths > 0:
        print(
            f"[load_labels] WARNING: repaired {repaired_paths} image path(s) in "
            f"{labels_path} using contiguous images/<row>.png fallback"
        )
    if trimmed_at_row is not None:
        print(
            f"[load_labels] NOTICE: y_embeddings will align to the first "
            f"{len(labels)} usable rows of {labels_path}"
        )
    return np.asarray(labels, dtype=label_dtype(dtype_name)).reshape(-1, 1)


def write_split_labels(
    *,
    dataset_name: str,
    split_name: str,
    split_dir: Path,
    args: argparse.Namespace,
) -> None:
    paths = output_paths(split_dir)
    if paths["array"].exists() and not args.overwrite:
        print(
            f"[{dataset_name}/{split_name}] reusing existing labels at {paths['array']}"
        )
        return

    labels = load_labels(
        split_dir,
        dataset_name=dataset_name,
        split_name=split_name,
        label_column=args.label_column,
        dtype_name=args.dtype,
    )

    paths["dir"].mkdir(parents=True, exist_ok=True)
    np.save(paths["array"], labels)
    save_json(
        paths["metadata"],
        {
            "dataset": dataset_name,
            "split": split_name,
            "source_file": "labels.csv",
            "label_column": args.label_column,
            "dtype": args.dtype,
            "shape": [int(labels.shape[0]), int(labels.shape[1])],
            "alignment": (
                "y_embeddings.npy row i matches labels.csv data row i "
                "(0-indexed) and the image path from that same row."
            ),
        },
    )
    print(
        f"[{dataset_name}/{split_name}] saved labels with shape {labels.shape} "
        f"to {paths['array']}"
    )


def main() -> None:
    args = parse_args()
    selected = selected_datasets(args.dataset)

    print("==============================================================================")
    print("Label Array Extraction")
    print("==============================================================================")
    print(f"Label column    : {args.label_column}")
    print(f"Storage dtype   : {args.dtype}")
    print(f"Overwrite       : {'yes' if args.overwrite else 'no'}")
    print("==============================================================================")

    for dataset_name in selected:
        dataset_root = args.data_root / dataset_root_name(dataset_name)
        print("")
        print(f"[{dataset_name}] dataset root: {dataset_root}")
        for split_name in args.splits:
            write_split_labels(
                dataset_name=dataset_name,
                split_name=split_name,
                split_dir=dataset_root / split_name,
                args=args,
            )


if __name__ == "__main__":
    main()
