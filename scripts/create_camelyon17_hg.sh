#!/bin/bash
# Download Camelyon17-WILDS from Hugging Face and materialize it as:
#   <output>/train/images/1.png
#   <output>/valid/images/1.png
#   <output>/test/images/1.png
# plus labels.csv for each split.

## bash scripts/create_camelyon17_hg.sh \
##  /scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets/camelyon17-hg


set -euo pipefail

usage() {
  cat <<'EOF'
Download Camelyon17-WILDS from Hugging Face as real image files.

Usage:
  bash scripts/create_camelyon17_hg.sh [output-dir]

Examples:
  bash scripts/create_camelyon17_hg.sh
  bash scripts/create_camelyon17_hg.sh /scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets/camelyon17-hg

Output structure:
  <output-dir>/train/images/1.png
  <output-dir>/train/labels.csv
  <output-dir>/valid/images/1.png
  <output-dir>/valid/labels.csv
  <output-dir>/test/images/1.png
  <output-dir>/test/labels.csv
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${1:-${PROJECT_ROOT}/datasets/camelyon17-hg}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "=================================="
echo "Camelyon17 Hugging Face Export"
echo "=================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Python:       ${PYTHON_BIN}"
echo "HF dataset:   wltjr1007/Camelyon17-WILDS"
echo "=================================="
echo ""

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
import csv
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "ERROR: The 'datasets' package is required. Run: pip install -r requirements.txt"
    ) from exc

repo_id = "wltjr1007/Camelyon17-WILDS"
output_dir = Path(sys.argv[1]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

print(f"[Camelyon17-HG] Output directory: {output_dir}")
print(f"[Camelyon17-HG] Downloading from Hugging Face: {repo_id}")


def count_label_rows(labels_path: Path) -> int:
    if not labels_path.exists():
        return 0

    with labels_path.open("r", newline="") as labels_file:
        reader = csv.reader(labels_file)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def count_contiguous_images(images_dir: Path) -> int:
    count = 0
    while (images_dir / f"{count + 1}.png").exists():
        count += 1
    return count


def trim_labels_to_count(labels_path: Path, keep_rows: int) -> None:
    if not labels_path.exists():
        return

    with labels_path.open("r", newline="") as labels_file:
        rows = list(csv.reader(labels_file))

    if not rows:
        return

    header = rows[0]
    data_rows = rows[1 : keep_rows + 1]
    with labels_path.open("w", newline="") as labels_file:
        writer = csv.writer(labels_file)
        writer.writerow(header)
        writer.writerows(data_rows)


split_map = {
    "train": "train",
    "validation": "valid",
    "test": "test",
}

fieldnames = [
    "image",
    "label",
    "center",
    "image_id",
    "patient",
    "node",
    "x_coord",
    "y_coord",
    "slide",
]

for hf_split, output_split in split_map.items():
    print(f"[Camelyon17-HG] Loading split: {hf_split}")
    dataset = load_dataset(repo_id, split=hf_split)
    total_examples = len(dataset)

    split_dir = output_dir / output_split
    images_dir = split_dir / "images"
    labels_path = split_dir / "labels.csv"

    images_dir.mkdir(parents=True, exist_ok=True)

    label_rows = count_label_rows(labels_path)
    image_rows = count_contiguous_images(images_dir)
    if label_rows > image_rows:
        print(
            f"[Camelyon17-HG] {output_split}: labels.csv has {label_rows} rows but only "
            f"{image_rows} contiguous images exist. Trimming labels.csv to {image_rows}.",
            flush=True,
        )
        trim_labels_to_count(labels_path, image_rows)
        label_rows = image_rows

    completed_rows = min(label_rows, image_rows)
    start_idx = completed_rows + 1
    if completed_rows > 0:
        print(
            f"[Camelyon17-HG] {output_split}: resuming from image {start_idx} "
            f"(labels={label_rows}, images={image_rows}, using {completed_rows})"
        )
    else:
        print(f"[Camelyon17-HG] {output_split}: starting fresh")

    saved = 0
    skipped = 0
    labels_mode = "a" if labels_path.exists() and label_rows > 0 else "w"
    with labels_path.open(labels_mode, newline="") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=fieldnames)
        if labels_mode == "w":
            writer.writeheader()
            labels_file.flush()

        for idx in range(start_idx, total_examples + 1):
            example = dataset[idx - 1]
            image_name = f"{idx}.png"
            image_path = images_dir / image_name

            if image_path.exists():
                skipped += 1
                print(
                    f"[Camelyon17-HG] {output_split}: {idx}/{total_examples} "
                    f"already exists: {image_path}",
                    flush=True,
                )
            else:
                example["image"].save(image_path)
                saved += 1
                print(
                    f"[Camelyon17-HG] {output_split}: {idx}/{total_examples} "
                    f"downloaded: {image_path}",
                    flush=True,
                )

            writer.writerow(
                {
                    "image": f"images/{image_name}",
                    "label": int(example["label"]),
                    "center": int(example["center"]),
                    "image_id": int(example["image_id"]),
                    "patient": int(example["patient"]),
                    "node": int(example["node"]),
                    "x_coord": int(example["x_coord"]),
                    "y_coord": int(example["y_coord"]),
                    "slide": int(example["slide"]),
                }
            )
            labels_file.flush()

    total = completed_rows + saved + skipped
    print(
        f"[Camelyon17-HG] Completed {output_split}: {total} images "
        f"({saved} saved this run, {skipped} already present this run). "
        f"Labels: {labels_path}"
    )

print("[Camelyon17-HG] Done.")
PY

echo ""
echo "Camelyon17 Hugging Face export complete."
echo "Created:"
echo "  ${OUTPUT_DIR}/train/images"
echo "  ${OUTPUT_DIR}/valid/images"
echo "  ${OUTPUT_DIR}/test/images"
