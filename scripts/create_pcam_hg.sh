#!/bin/bash
# Download PatchCamelyon from Hugging Face and materialize it as:
#   <output>/train/images/1.png
#   <output>/valid/images/1.png
#   <output>/test/images/1.png
# plus labels.csv for each split.

# bash scripts/create_pcam_hg.sh \
#  /scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets/pcam-hg


set -euo pipefail

usage() {
  cat <<'EOF'
Download PatchCamelyon from Hugging Face as real image files.

Usage:
  bash scripts/create_pcam_hg.sh [output-dir]

Examples:
  bash scripts/create_pcam_hg.sh
  bash scripts/create_pcam_hg.sh /scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets/pcam-hg

Output structure:
  <output-dir>/train/images/1.png
  <output-dir>/train/images/2.png
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
OUTPUT_DIR="${1:-${PROJECT_ROOT}/datasets/pcam-hg}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "=================================="
echo "PCam Hugging Face Export"
echo "=================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Python:       ${PYTHON_BIN}"
echo "HF dataset:   1aurent/PatchCamelyon"
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

repo_id = "1aurent/PatchCamelyon"
output_dir = Path(sys.argv[1]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

print(f"[PCam-HG] Output directory: {output_dir}")
print(f"[PCam-HG] Downloading from Hugging Face: {repo_id}")


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

for split in ("train", "valid", "test"):
    print(f"[PCam-HG] Loading split: {split}")
    dataset = load_dataset(repo_id, split=split)
    total_examples = len(dataset)

    split_dir = output_dir / split
    images_dir = split_dir / "images"
    labels_path = split_dir / "labels.csv"

    images_dir.mkdir(parents=True, exist_ok=True)

    label_rows = count_label_rows(labels_path)
    image_rows = count_contiguous_images(images_dir)
    if label_rows > image_rows:
        print(
            f"[PCam-HG] {split}: labels.csv has {label_rows} rows but only "
            f"{image_rows} contiguous images exist. Trimming labels.csv to {image_rows}.",
            flush=True,
        )
        trim_labels_to_count(labels_path, image_rows)
        label_rows = image_rows

    completed_rows = min(label_rows, image_rows)
    start_idx = completed_rows + 1
    if completed_rows > 0:
        print(
            f"[PCam-HG] {split}: resuming from image {start_idx} "
            f"(labels={label_rows}, images={image_rows}, using {completed_rows})"
        )
    else:
        print(f"[PCam-HG] {split}: starting fresh")

    saved = 0
    skipped = 0
    labels_mode = "a" if labels_path.exists() and label_rows > 0 else "w"
    with labels_path.open(labels_mode, newline="") as labels_file:
        writer = csv.writer(labels_file)
        if labels_mode == "w":
            writer.writerow(["image", "label"])
            labels_file.flush()

        for idx in range(start_idx, total_examples + 1):
            example = dataset[idx - 1]
            image_name = f"{idx}.png"
            image_path = images_dir / image_name

            if image_path.exists():
                skipped += 1
                print(
                    f"[PCam-HG] {split}: {idx}/{total_examples} already exists: {image_path}",
                    flush=True,
                )
            else:
                example["image"].save(image_path)
                saved += 1
                print(
                    f"[PCam-HG] {split}: {idx}/{total_examples} downloaded: {image_path}",
                    flush=True,
                )

            writer.writerow([f"images/{image_name}", int(bool(example["label"]))])
            labels_file.flush()

    total = completed_rows + saved + skipped
    print(
        f"[PCam-HG] Completed {split}: {total} images "
        f"({saved} saved this run, {skipped} already present this run). "
        f"Labels: {labels_path}"
    )

print("[PCam-HG] Done.")
PY

echo ""
echo "PCam Hugging Face export complete."
echo "Created:"
echo "  ${OUTPUT_DIR}/train/images"
echo "  ${OUTPUT_DIR}/valid/images"
echo "  ${OUTPUT_DIR}/test/images"
