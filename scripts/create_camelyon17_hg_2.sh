#!/bin/bash
# Download Camelyon17-WILDS from Hugging Face and materialize it as:
#   <output>/train/images/1.png
#   <output>/valid/images/1.png
#   <output>/test/images/1.png
# plus labels.csv for each split. Logs every image as it is added.

## bash scripts/create_camelyon17_hg.sh \
##  /scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets/camelyon17-hg-2

set -euo pipefail

usage() {
  cat <<'EOF'
Download Camelyon17-WILDS from Hugging Face as real image files,
with per-image logging to stdout and to <output-dir>/download.log.

Usage:
  bash scripts/create_camelyon17_hg.sh [output-dir]

Output structure:
  <output-dir>/train/images/1.png
  <output-dir>/train/labels.csv
  <output-dir>/valid/images/1.png
  <output-dir>/valid/labels.csv
  <output-dir>/test/images/1.png
  <output-dir>/test/labels.csv
  <output-dir>/download.log
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage; exit 0
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
import logging
import sys
import time
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "ERROR: The 'datasets' package is required. Run: pip install datasets pillow"
    ) from exc

# ---------------------------------------------------------------------------
# Logging: console + persistent file in the output dir
# ---------------------------------------------------------------------------
output_dir = Path(sys.argv[1]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
log_path = output_dir / "download.log"

log = logging.getLogger("camelyon17-hg")
log.setLevel(logging.INFO)
log.handlers.clear()

fmt = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%H:%M:%S",
)
sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)
fh = logging.FileHandler(log_path, mode="a"); fh.setFormatter(fmt); log.addHandler(fh)

repo_id = "wltjr1007/Camelyon17-WILDS"
log.info("Output directory: %s", output_dir)
log.info("Log file:         %s", log_path)
log.info("HF dataset:       %s", repo_id)


# ---------------------------------------------------------------------------
# Helpers (same resume logic as before — see the audit script for caveats)
# ---------------------------------------------------------------------------
def count_label_rows(labels_path: Path) -> int:
    if not labels_path.exists():
        return 0
    with labels_path.open("r", newline="") as f:
        reader = csv.reader(f)
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
    with labels_path.open("r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header, data = rows[0], rows[1 : keep_rows + 1]
    with labels_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header); w.writerows(data)


def fmt_rate(n: int, elapsed: float) -> str:
    return f"{(n / elapsed):.1f} img/s" if elapsed > 0 else "—"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
split_map = {"train": "train", "validation": "valid", "test": "test"}

fieldnames = [
    "image", "label", "center", "image_id",
    "patient", "node", "x_coord", "y_coord", "slide",
]

# How often to log every image vs. progress lines. Set LOG_EVERY=1 to log
# every single image (very chatty for 456k rows). Default logs every image
# but compactly; a heartbeat summary is printed every HEARTBEAT images.
LOG_EVERY = 1
HEARTBEAT = 1000

# Per-split totals collected for the final summary table
totals = []

for hf_split, output_split in split_map.items():
    log.info("=" * 60)
    log.info("Loading split: %s -> %s", hf_split, output_split)
    dataset = load_dataset(repo_id, split=hf_split)
    total = len(dataset)

    split_dir   = output_dir / output_split
    images_dir  = split_dir / "images"
    labels_path = split_dir / "labels.csv"
    images_dir.mkdir(parents=True, exist_ok=True)

    label_rows = count_label_rows(labels_path)
    image_rows = count_contiguous_images(images_dir)
    if label_rows > image_rows:
        log.warning(
            "%s: labels.csv has %d rows but only %d contiguous images. "
            "Trimming labels.csv to %d.",
            output_split, label_rows, image_rows, image_rows,
        )
        trim_labels_to_count(labels_path, image_rows)
        label_rows = image_rows

    completed = min(label_rows, image_rows)
    start_idx = completed + 1
    if completed > 0:
        log.info(
            "%s: resuming from image %d (labels=%d, images=%d, using=%d)",
            output_split, start_idx, label_rows, image_rows, completed,
        )
    else:
        log.info("%s: starting fresh (total=%d)", output_split, total)

    saved = skipped = failed = 0
    t0 = time.time()
    labels_mode = "a" if labels_path.exists() and label_rows > 0 else "w"

    with labels_path.open(labels_mode, newline="") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=fieldnames)
        if labels_mode == "w":
            writer.writeheader(); labels_file.flush()

        for idx in range(start_idx, total + 1):
            example = dataset[idx - 1]
            image_name = f"{idx}.png"
            image_path = images_dir / image_name

            try:
                if image_path.exists():
                    skipped += 1
                    status = "SKIP "
                else:
                    example["image"].save(image_path)
                    saved += 1
                    status = "ADDED"

                writer.writerow({
                    "image":    f"images/{image_name}",
                    "label":    int(example["label"]),
                    "center":   int(example["center"]),
                    "image_id": int(example["image_id"]),
                    "patient":  int(example["patient"]),
                    "node":     int(example["node"]),
                    "x_coord":  int(example["x_coord"]),
                    "y_coord":  int(example["y_coord"]),
                    "slide":    int(example["slide"]),
                })
                labels_file.flush()

                done = idx
                pct = 100.0 * done / total
                if (done - start_idx + 1) % LOG_EVERY == 0:
                    log.info(
                        "[%s] [%s] %6d/%-6d (%5.1f%%) %s",
                        output_split, status, done, total, pct,
                        image_path.relative_to(output_dir),
                    )

                if (done - start_idx + 1) % HEARTBEAT == 0:
                    elapsed = time.time() - t0
                    rate = fmt_rate(saved + skipped, elapsed)
                    remaining = total - done
                    eta = (remaining / max(1, saved + skipped)) * elapsed
                    log.info(
                        "[%s] heartbeat: done=%d saved=%d skipped=%d failed=%d "
                        "rate=%s eta=%.0fs",
                        output_split, done, saved, skipped, failed, rate, eta,
                    )

            except Exception as e:
                failed += 1
                log.error(
                    "[%s] [FAIL ] %6d/%-6d %s — %s",
                    output_split, idx, total, image_path.name, e,
                )

    elapsed = time.time() - t0
    on_disk = completed + saved + skipped
    log.info(
        "Completed %s: total_on_disk=%d, saved=%d, skipped=%d, failed=%d, "
        "elapsed=%.1fs (%s)",
        output_split, on_disk,
        saved, skipped, failed, elapsed, fmt_rate(saved + skipped, elapsed),
    )
    totals.append({
        "split":    output_split,
        "expected": total,
        "on_disk":  on_disk,
        "added":    saved,
        "skipped":  skipped,
        "failed":   failed,
        "elapsed":  elapsed,
    })

# ---------------------------------------------------------------------------
# Final cross-split summary
# ---------------------------------------------------------------------------
log.info("=" * 60)
log.info("FINAL SUMMARY")
log.info("=" * 60)
header = f"{'split':<8} {'expected':>10} {'on_disk':>10} {'added':>10} {'skipped':>10} {'failed':>8}"
log.info(header)
log.info("-" * len(header))
gt = {"expected": 0, "on_disk": 0, "added": 0, "skipped": 0, "failed": 0}
for t in totals:
    log.info(
        f"{t['split']:<8} {t['expected']:>10} {t['on_disk']:>10} "
        f"{t['added']:>10} {t['skipped']:>10} {t['failed']:>8}"
    )
    for k in gt:
        gt[k] += t[k]
log.info("-" * len(header))
log.info(
    f"{'TOTAL':<8} {gt['expected']:>10} {gt['on_disk']:>10} "
    f"{gt['added']:>10} {gt['skipped']:>10} {gt['failed']:>8}"
)
missing = gt["expected"] - gt["on_disk"]
if missing:
    log.warning("%d image(s) expected but not on disk — re-run to resume.", missing)
else:
    log.info("All %d expected images are present on disk.", gt["expected"])

log.info("Done. Full log: %s", log_path)
PY

echo ""
echo "Camelyon17 Hugging Face export complete."
echo "Created:"
echo "  ${OUTPUT_DIR}/train/images"
echo "  ${OUTPUT_DIR}/valid/images"
echo "  ${OUTPUT_DIR}/test/images"
echo "  ${OUTPUT_DIR}/download.log"