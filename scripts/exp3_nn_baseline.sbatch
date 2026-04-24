#!/bin/bash
#SBATCH --job-name=exp3_nn_baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

# =============================================================================
# User-configurable paths
# =============================================================================
# Submit with:
#   sbatch --export=ALL,NETID=ab1234 scripts/exp3_nn_baseline.sbatch
#
# Optional overrides:
#   DATASET       pcam | camelyon17 | embed          (default: pcam)
#   ENCODER       densenet121 | dinov2_vitl14        (default: densenet121)
#   CONDA_ENV     /path/to/env                       (auto-detected)
#   PROJECT_ROOT  /path/to/project                   (auto-detected)
#   SKIP_EMBED    1                                  (skip embedding extraction)

NETID="${NETID:-<netid>}"
if [ "${NETID}" = "<netid>" ]; then
  echo "ERROR: NETID is required. Submit with --export=ALL,NETID=<your-netid>." >&2
  exit 1
fi

DATASET="${DATASET:-pcam}"
ENCODER="${ENCODER:-densenet121}"
SKIP_EMBED="${SKIP_EMBED:-0}"

SCRATCH_BASE="/scratch/${NETID}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/gpc-mcmc-cholesky-factorization}"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"

# =============================================================================
# Load scratch environment (caches, tmp dirs, etc.)
# =============================================================================
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
if [ -f "${SCRIPT_DIR}/set_scratch_env.sh" ]; then
  source "${SCRIPT_DIR}/set_scratch_env.sh"
else
  echo "WARN: set_scratch_env.sh not found, continuing without it." >&2
fi

# =============================================================================
# Conda activation (cluster-specific, CRITICAL sequence)
# =============================================================================
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# Avoid occasional conda hook issues under strict nounset shells on this cluster.
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
set +u
conda activate "${CONDA_ENV}"
set -u

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "${CONDA_PREFIX}" ]; then
  echo "ERROR: Failed to activate conda environment: ${CONDA_ENV}" >&2
  echo "Pass the correct env with --export=ALL,CONDA_ENV=/scratch/${NETID}/conda-envs/<env>" >&2
  exit 1
fi
echo "Activated conda env: ${CONDA_PREFIX}"

# Ensure the activated env's Python wins over module-level Anaconda paths.
export PATH="${CONDA_PREFIX}/bin:${PATH}"

unset PYTHONHOME
unset PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python not found at ${PYTHON}" >&2
  exit 1
fi
echo "Using: ${PYTHON} ($(${PYTHON} --version 2>&1))"

"${PYTHON}" -c "import torch; print(f'PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}')" || {
  echo "ERROR: PyTorch import check failed." >&2
  exit 1
}

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# =============================================================================
# Run Experiment 3
# =============================================================================
cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
mkdir -p data/embeddings

echo "=============================================================================="
echo "Experiment 3: Deterministic Neural Network Baseline"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time   : $(date)"
echo "Project root : ${PROJECT_ROOT}"
echo "Conda env    : ${CONDA_ENV}"
echo "Dataset      : ${DATASET}"
echo "Encoder      : ${ENCODER}"
echo "=============================================================================="

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# --- Step 1: Extract embeddings (GPU) ----------------------------------------
if [ "${SKIP_EMBED}" != "1" ]; then
  echo ""
  echo "[Step 1] Extracting embeddings: ${DATASET} with ${ENCODER}"
  "${PYTHON}" experiments/extract_embeddings.py \
      --dataset "${DATASET}" \
      --data-root datasets \
      --output-dir data/embeddings \
      --encoder "${ENCODER}" \
      --batch-size 256 \
      --device cuda
else
  echo ""
  echo "[Step 1] Skipped (SKIP_EMBED=1). Using existing embeddings in data/embeddings/"
fi

# --- Step 2: Train MLP + evaluate --------------------------------------------
echo ""
echo "[Step 2] Training MLP head and evaluating on ${DATASET}"
"${PYTHON}" experiments/exp3_nn_baseline.py \
    --dataset "${DATASET}" \
    --embedding-dir data/embeddings \
    --output-dir data \
    --device cuda

echo "=============================================================================="
echo "Done: $(date)"
echo "Expected outputs in ${PROJECT_ROOT}/data:"
echo "  - exp3_${DATASET}_results.json"
echo "  - exp3_${DATASET}_calibration.png"
echo "  - exp3_${DATASET}_roc.png"
echo "=============================================================================="
