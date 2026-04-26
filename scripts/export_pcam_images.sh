#!/bin/bash
# Export already-downloaded PatchCamelyon archives into:
#   pcam/train/images
#   pcam/valid/images
#   pcam/test/images
#
# Usage:
#   bash scripts/export_pcam_images.sh /scratch/<netid>/gpc-mcmc-cholesky-factorization/datasets
#   bash scripts/export_pcam_images.sh /scratch/<netid>/gpc-mcmc-cholesky-factorization/datasets/pcam
#   bash scripts/export_pcam_images.sh /scratch/<netid>/gpc-mcmc-cholesky-factorization/datasets png

set -euo pipefail

usage() {
  cat <<'EOF'
Export already-downloaded PCam .h5.gz files into split image folders.

Usage:
  bash scripts/export_pcam_images.sh <datasets-root-or-pcam-dir> [png|jpg|jpeg]

Examples:
  bash scripts/export_pcam_images.sh /scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets
  bash scripts/export_pcam_images.sh /scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets/pcam
  bash scripts/export_pcam_images.sh /scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets png

This script expects the original PCam files to already exist under:
  <datasets-root>/pcam/

It will create:
  <datasets-root>/pcam/train/images
  <datasets-root>/pcam/valid/images
  <datasets-root>/pcam/test/images
  <datasets-root>/pcam/{train,valid,test}/labels.csv
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_PATH="${1:-${PROJECT_ROOT}/datasets}"
IMAGE_FORMAT="${2:-png}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "${IMAGE_FORMAT}" in
  png|jpg|jpeg) ;;
  *)
    echo "ERROR: Unsupported image format '${IMAGE_FORMAT}'. Use png, jpg, or jpeg." >&2
    exit 1
    ;;
esac

if [ -d "${INPUT_PATH}/pcam" ]; then
  DATASETS_ROOT="${INPUT_PATH}"
  PCAM_DIR="${INPUT_PATH}/pcam"
elif [ -d "${INPUT_PATH}" ] && [ "$(basename "${INPUT_PATH}")" = "pcam" ]; then
  DATASETS_ROOT="$(cd "${INPUT_PATH}/.." && pwd)"
  PCAM_DIR="${INPUT_PATH}"
else
  echo "ERROR: Could not locate a PCam directory from '${INPUT_PATH}'." >&2
  echo "Pass either the datasets root or the pcam directory itself." >&2
  exit 1
fi

if [ ! -d "${PCAM_DIR}" ]; then
  echo "ERROR: PCam directory not found: ${PCAM_DIR}" >&2
  exit 1
fi

echo "=================================="
echo "PCam Image Export"
echo "=================================="
echo "Project root:  ${PROJECT_ROOT}"
echo "Datasets root: ${DATASETS_ROOT}"
echo "PCam dir:      ${PCAM_DIR}"
echo "Image format:  ${IMAGE_FORMAT}"
echo "Python:        ${PYTHON_BIN}"
echo "=================================="
echo ""

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/download_datasets.py" \
  --datasets pcam \
  --root "${DATASETS_ROOT}" \
  --pcam-source existing \
  --prepare-pcam \
  --export-pcam-images \
  --pcam-image-format "${IMAGE_FORMAT}"

echo ""
echo "PCam export complete."
echo "Images are under:"
echo "  ${PCAM_DIR}/train/images"
echo "  ${PCAM_DIR}/valid/images"
echo "  ${PCAM_DIR}/test/images"
