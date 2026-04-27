#!/bin/bash
#SBATCH --job-name=exp4_tabpfn_test
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

# Submit example:
#   sbatch --export=ALL,NETID=ab1234,DATASET=pcam experiments/exp4_tabpfn.sbatch

SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${SBATCH_DIR}/.." && pwd)}}"
NETID="${NETID:-${USER:-}}"

if [ -z "${NETID}" ]; then
  echo "ERROR: NETID could not be determined." >&2
  exit 1
fi

DATASET="${DATASET:-pcam}"
DATA_ROOT="${DATA_ROOT:-datasets}"
EMBEDDING_DIR="${EMBEDDING_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
DEVICE="${DEVICE:-cuda}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-3500}"
PREDICT_CHUNK_SIZE="${PREDICT_CHUNK_SIZE:-1000}"
N_ESTIMATORS="${N_ESTIMATORS:-4}"
SEED="${SEED:-42}"
IGNORE_PRETRAINING_LIMITS="${IGNORE_PRETRAINING_LIMITS:-0}"

SCRATCH_BASE="/scratch/${NETID}"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"

source "/scratch/${NETID}/miniconda3/etc/profile.d/conda.sh"
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
export TABPFN_DISABLE_TELEMETRY="${TABPFN_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python not found at ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" -c "import tabpfn; print('TabPFN imported OK')" || {
  echo "ERROR: TabPFN import check failed. Install it in ${CONDA_ENV}." >&2
  exit 1
}

hg_dataset_root_name() {
  case "${DATASET}" in
    pcam) echo "pcam-hg" ;;
    camelyon17) echo "camelyon17-hg" ;;
    *) echo "${DATASET}" ;;
  esac
}

default_hg_embedding_dir="${DATA_ROOT}/$(hg_dataset_root_name)"
if [ -z "${EMBEDDING_DIR}" ]; then
  if [ -d "${PROJECT_ROOT}/${default_hg_embedding_dir}/train/embeddings" ]; then
    EMBEDDING_DIR="${default_hg_embedding_dir}"
  else
    EMBEDDING_DIR="data/embeddings"
  fi
fi

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
mkdir -p "${OUTPUT_DIR}"

if [ ! -f "experiments/exp4_tabpfn_test.py" ]; then
  echo "ERROR: experiments/exp4_tabpfn_test.py not found under ${PROJECT_ROOT}" >&2
  exit 1
fi

echo "=============================================================================="
echo "Experiment 4: TabPFN Test Baseline"
echo "=============================================================================="
echo "Job ID            : ${SLURM_JOB_ID:-N/A}"
echo "Node              : $(hostname)"
echo "GPU               : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time        : $(date)"
echo "Project root      : ${PROJECT_ROOT}"
echo "Conda env         : ${CONDA_ENV}"
echo "Dataset           : ${DATASET}"
echo "Embedding dir     : ${EMBEDDING_DIR}"
echo "Output dir        : ${OUTPUT_DIR}"
echo "Device            : ${DEVICE}"
echo "Max train samples : ${MAX_TRAIN_SAMPLES}"
echo "Predict chunk     : ${PREDICT_CHUNK_SIZE}"
echo "N estimators      : ${N_ESTIMATORS}"
echo "Seed              : ${SEED}"
echo "=============================================================================="

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

CMD=(
  "${PYTHON}" "experiments/exp4_tabpfn_test.py"
  "--dataset" "${DATASET}"
  "--embedding-dir" "${EMBEDDING_DIR}"
  "--output-dir" "${OUTPUT_DIR}"
  "--max-train-samples" "${MAX_TRAIN_SAMPLES}"
  "--predict-chunk-size" "${PREDICT_CHUNK_SIZE}"
  "--device" "${DEVICE}"
  "--n-estimators" "${N_ESTIMATORS}"
  "--seed" "${SEED}"
)

if [ "${IGNORE_PRETRAINING_LIMITS}" = "1" ]; then
  CMD+=("--ignore-pretraining-limits")
fi

"${CMD[@]}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Expected outputs in ${PROJECT_ROOT}/${OUTPUT_DIR}:"
echo "  - exp4_${DATASET}_results.json"
echo "  - exp4_${DATASET}_calibration.png"
echo "  - exp4_${DATASET}_roc.png"
echo "=============================================================================="
