#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/setup_env.sh
#   ./scripts/setup_env.sh --with-datasets
#
# This script creates a local virtual environment, installs base dependencies,
# and optionally installs dataset tooling dependencies used by scripts/download_datasets.py.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
WITH_DATASETS=0

for arg in "$@"; do
  case "$arg" in
    --with-datasets)
      WITH_DATASETS=1
      ;;
    *)
      echo "Unknown argument: ${arg}"
      echo "Usage: ./scripts/setup_env.sh [--with-datasets]"
      exit 1
      ;;
  esac
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3.10+ and retry."
  exit 1
fi

echo "Creating virtual environment at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"

echo "Activating virtual environment"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "Installing base dependencies from requirements.txt"
python -m pip install -r "${ROOT_DIR}/requirements.txt"

if [[ "${WITH_DATASETS}" -eq 1 ]]; then
  echo "Installing optional dataset dependencies (WILDS + PyTorch)"
  python -m pip install wilds torch torchvision
fi

echo
echo "Environment setup complete."
echo "Activate with:"
echo "  source .venv/bin/activate"
