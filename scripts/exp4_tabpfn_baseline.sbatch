#!/bin/bash
#SBATCH --job-name=exp4_tabpfn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
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
#   sbatch --export=ALL,NETID=ab1234 scripts/exp4_tabpfn_baseline.sbatch
#
# Optional overrides:
#   DATASET              pcam | camelyon17 | embed     (default: pcam)
#   MAX_TRAIN_SAMPLES    int                           (default: 50000)
#   CONDA_ENV            /path/to/env                  (auto-detected)
#   PROJECT_ROOT         /path/to/project              (auto-detected)

NETID="${NETID:-<netid>}"
if [ "${NETID}" = "<netid>" ]; then
  echo "ERROR: NETID is required. Submit with --export=ALL,NETID=<your-netid>." >&2
  exit 1
fi

DATASET="${DATASET:-pcam}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-50000}"

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

if [ ! -d "${CONDA_ENV}" ]; then
  echo "ERROR: Conda env not found at ${CONDA_ENV}" >&2
  echo "Create it first on the login node:" >&2
  echo "  NETID=${NETID} bash scripts/setup_env.sh" >&2
  exit 1
fi

conda activate "${CONDA_ENV}"
echo "Activated conda env: ${CONDA_ENV}"

unset PYTHONHOME
unset PYTHONPATH

PYTHON="${CONDA_ENV}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python not found at ${PYTHON}" >&2
  exit 1
fi
echo "Using: ${PYTHON} ($(${PYTHON} --version 2>&1))"

"${PYTHON}" -c "import tabpfn; print(f'TabPFN imported OK')" || {
  echo "ERROR: TabPFN import check failed. Install with: pip install tabpfn" >&2
  exit 1
}

export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# Disable TabPFN telemetry on cluster jobs
export TABPFN_DISABLE_TELEMETRY=1

# =============================================================================
# Run Experiment 4
# =============================================================================
cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
mkdir -p data

echo "=============================================================================="
echo "Experiment 4: TabPFN Tabular Model Baseline"
echo "=============================================================================="
echo "Job ID            : ${SLURM_JOB_ID}"
echo "Node              : $(hostname)"
echo "GPU               : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time        : $(date)"
echo "Project root      : ${PROJECT_ROOT}"
echo "Conda env         : ${CONDA_ENV}"
echo "Dataset           : ${DATASET}"
echo "Max train samples : ${MAX_TRAIN_SAMPLES}"
echo "=============================================================================="

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

"${PYTHON}" experiments/exp4_tabpfn_baseline.py \
    --dataset "${DATASET}" \
    --embedding-dir data/embeddings \
    --output-dir data \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --device cuda

echo "=============================================================================="
echo "Done: $(date)"
echo "Expected outputs in ${PROJECT_ROOT}/data:"
echo "  - exp4_${DATASET}_results.json"
echo "  - exp4_${DATASET}_calibration.png"
echo "  - exp4_${DATASET}_roc.png"
echo "=============================================================================="
