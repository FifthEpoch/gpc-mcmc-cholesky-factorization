#!/bin/bash
# Create conda environment and install project dependencies.
#
# This script is designed for the cluster where /home has a small quota.
# All packages and caches are stored under /scratch/<NETID>.
#
# Usage (on login node):
#   NETID=ab1234 bash scripts/setup_env.sh
#   NETID=ab1234 bash scripts/setup_env.sh --with-datasets
#
# The conda env will be created at /scratch/<NETID>/conda-envs/gpc

set -euo pipefail

# ============================================================================
# Load scratch environment variables
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/set_scratch_env.sh" ]; then
  echo "Loading scratch environment configuration..."
  source "${SCRIPT_DIR}/set_scratch_env.sh"
else
  echo "ERROR: ${SCRIPT_DIR}/set_scratch_env.sh not found!" >&2
  exit 1
fi

WITH_DATASETS=0
for arg in "$@"; do
  case "$arg" in
    --with-datasets) WITH_DATASETS=1 ;;
    *) echo "Unknown argument: ${arg}"; echo "Usage: NETID=ab1234 bash scripts/setup_env.sh [--with-datasets]"; exit 1 ;;
  esac
done

# ============================================================================
# Load cluster anaconda module
# ============================================================================
module purge || true
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# Configure conda to use scratch directories
conda config --add envs_dirs "${SCRATCH_BASE}/conda-envs" 2>/dev/null || true
conda config --add pkgs_dirs "${SCRATCH_BASE}/conda-pkgs" 2>/dev/null || true

# ============================================================================
# Create environment
# ============================================================================
ENV_PATH="${SCRATCH_BASE}/conda-envs/gpc"

if [ -d "${ENV_PATH}" ]; then
  echo "Conda env already exists at ${ENV_PATH}, reusing it."
else
  echo "Creating conda env at ${ENV_PATH}..."
  conda create -y -p "${ENV_PATH}" python=3.12
fi

conda activate "${ENV_PATH}"
unset PYTHONHOME
unset PYTHONPATH

PYTHON="${ENV_PATH}/bin/python"
echo "Using: ${PYTHON} ($(${PYTHON} --version 2>&1))"

python -m pip install -U pip setuptools wheel

# ============================================================================
# Install project dependencies
# ============================================================================
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Installing base dependencies from requirements.txt..."
pip install -r "${PROJECT_ROOT}/requirements.txt"

if [ "${WITH_DATASETS}" -eq 1 ]; then
  echo "Installing optional dataset dependencies (WILDS + PyTorch)..."
  pip install wilds torch torchvision
fi

# ============================================================================
# Verification
# ============================================================================
echo ""
echo "============================================"
echo "Environment Setup Complete - Version Check"
echo "============================================"
"${PYTHON}" -c "
import sys
import numpy as np
import scipy
import sklearn
import matplotlib
import emcee

print(f'Python:       {sys.version.split()[0]}')
print(f'NumPy:        {np.__version__}')
print(f'SciPy:        {scipy.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print(f'matplotlib:   {matplotlib.__version__}')
print(f'emcee:        {emcee.__version__}')
print()
print('All core packages imported successfully.')
"
echo "============================================"
echo ""
echo "Environment is ready at: ${ENV_PATH}"
echo ""
echo "To activate in the future:"
echo "  module load anaconda3/2025.06"
echo "  source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_PATH}"
echo "  unset PYTHONHOME PYTHONPATH"
echo ""
echo "To run Experiment 0 on the cluster:"
echo "  sbatch --export=ALL,NETID=${NETID} scripts/exp0_algorithm_verification.sbatch"
