#!/bin/bash
#SBATCH --job-name=exp3_test
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
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

# Run only experiments/exp3_nn_baseline.py.
# This script expects embeddings to already exist under EMBEDDING_DIR.
#
# Example:
#   sbatch --export=ALL,DATASET=pcam test.sbatch

SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SBATCH_DIR}}"
NETID="${NETID:-}"

if [ -z "${NETID}" ]; then
  case "${PROJECT_ROOT}" in
    /scratch/*/*)
      NETID="${PROJECT_ROOT#/scratch/}"
      NETID="${NETID%%/*}"
      ;;
    *)
      NETID="${USER:-}"
      ;;
  esac
fi

if [ -z "${NETID}" ]; then
  echo "ERROR: NETID could not be determined." >&2
  echo "Submit with --export=ALL,NETID=<your-netid> if auto-detection fails." >&2
  exit 1
fi

DATASET="${DATASET:-pcam}"
EMBEDDING_DIR="${EMBEDDING_DIR:-data/embeddings}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
DEVICE="${DEVICE:-cuda}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
DROPOUT="${DROPOUT:-0.3}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SEED="${SEED:-42}"

SCRATCH_BASE="/scratch/${NETID}"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
set +u
conda activate "${CONDA_ENV}"
set -u

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "${CONDA_PREFIX}" ]; then
  echo "ERROR: Failed to activate conda environment: ${CONDA_ENV}" >&2
  exit 1
fi

export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset PYTHONHOME
unset PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python not found at ${PYTHON}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
mkdir -p "${OUTPUT_DIR}"

if [ ! -f "experiments/exp3_nn_baseline.py" ]; then
  echo "ERROR: experiments/exp3_nn_baseline.py not found under ${PROJECT_ROOT}" >&2
  exit 1
fi

if [ ! -f "${EMBEDDING_DIR}/${DATASET}_train_embeddings.npy" ]; then
  echo "ERROR: Missing embeddings for dataset=${DATASET} in ${EMBEDDING_DIR}" >&2
  echo "Expected file: ${EMBEDDING_DIR}/${DATASET}_train_embeddings.npy" >&2
  echo "Run embedding extraction first, or point EMBEDDING_DIR to an existing embedding directory." >&2
  exit 1
fi

echo "=============================================================================="
echo "Experiment 3 Test Job"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Project root : ${PROJECT_ROOT}"
echo "Conda env    : ${CONDA_ENV}"
echo "Dataset      : ${DATASET}"
echo "Embed dir    : ${EMBEDDING_DIR}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Device       : ${DEVICE}"
echo "Epochs       : ${EPOCHS}"
echo "Batch size   : ${BATCH_SIZE}"
echo "=============================================================================="

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

"${PYTHON}" experiments/exp3_nn_baseline.py \
    --dataset "${DATASET}" \
    --embedding-dir "${EMBEDDING_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --dropout "${DROPOUT}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --seed "${SEED}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Expected outputs in ${PROJECT_ROOT}/${OUTPUT_DIR}:"
echo "  - exp3_${DATASET}_results.json"
echo "  - exp3_${DATASET}_calibration.png"
echo "  - exp3_${DATASET}_roc.png"
echo "=============================================================================="
