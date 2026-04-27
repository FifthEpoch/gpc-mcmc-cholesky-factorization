#!/bin/bash
#SBATCH --job-name=exp1_svgp
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/svgp/%x_%j.out
#SBATCH --error=slurm_logs/svgp/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

SBATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${SBATCH_DIR}/.." && pwd)}}"

infer_netid() {
  case "$1" in
    /scratch/*/*)
      printf '%s\n' "$1" | awk -F/ '{print $3}'
      ;;
    *)
      return 1
      ;;
  esac
}

NETID="${NETID:-}"
if [ -z "${NETID}" ]; then
  NETID="$(infer_netid "${PROJECT_ROOT}" 2>/dev/null || true)"
fi
if [ -z "${NETID}" ] && [ -n "${USER:-}" ]; then
  NETID="${USER}"
fi
if [ -z "${NETID}" ]; then
  echo "ERROR: Could not infer NETID. Submit with --export=ALL,NETID=<your-netid>." >&2
  exit 1
fi

DATASET="${DATASET:-pcam}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
DEVICE="${DEVICE:-cuda}"
NUM_INDUCING="${NUM_INDUCING:-256}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-15}"
PATIENCE="${PATIENCE:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-0}"
DISABLE_STANDARDIZE="${DISABLE_STANDARDIZE:-0}"
CONDA_ENV="${CONDA_ENV:-/scratch/${NETID}/conda-envs/gpc}"
# Set to 0 to skip automatic pip installs (fail fast if anything is missing).
INSTALL_SVGP_DEPS="${INSTALL_SVGP_DEPS:-1}"

default_hg_embedding_dir="datasets/${DATASET}-hg"
if [ -d "${PROJECT_ROOT}/${default_hg_embedding_dir}/train/embeddings" ]; then
  EMBEDDING_DIR="${EMBEDDING_DIR:-${default_hg_embedding_dir}}"
else
  EMBEDDING_DIR="${EMBEDDING_DIR:-data/embeddings}"
fi

have_flat_embeddings() {
  local root="$1"
  local dataset="$2"
  local split="$3"
  [ -f "${root}/${dataset}_${split}_embeddings.npy" ] && [ -f "${root}/${dataset}_${split}_labels.npy" ]
}

have_hg_embeddings() {
  local root="$1"
  local split_dir="$2"
  if [ ! -d "${root}/${split_dir}/embeddings" ]; then
    return 1
  fi

  if [ -f "${root}/${split_dir}/embeddings/projected_512.npy" ] || [ -f "${root}/${split_dir}/embeddings/embeddings.npy" ]; then
    if [ -f "${root}/${split_dir}/embeddings/y_embeddings.npy" ] || [ -f "${root}/${split_dir}/embeddings/labels.npy" ] || [ -f "${root}/${split_dir}/labels.csv" ]; then
      return 0
    fi
  fi
  return 1
}

if ! have_flat_embeddings "${PROJECT_ROOT}/${EMBEDDING_DIR}" "${DATASET}" "train" \
  && ! have_hg_embeddings "${PROJECT_ROOT}/${EMBEDDING_DIR}" "train"; then
  echo "ERROR: Missing embeddings for dataset=${DATASET} in ${EMBEDDING_DIR}" >&2
  echo "Expected either:" >&2
  echo "  - ${EMBEDDING_DIR}/${DATASET}_train_embeddings.npy" >&2
  echo "  - ${EMBEDDING_DIR}/train/embeddings/projected_512.npy" >&2
  echo "  - ${EMBEDDING_DIR}/train/embeddings/embeddings.npy" >&2
  exit 1
fi

if [ -f "${PROJECT_ROOT}/scripts/set_scratch_env.sh" ]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/scripts/set_scratch_env.sh"
fi

load_conda() {
  module purge >/dev/null 2>&1 || true

  local candidates=(
    "/scratch/${NETID}/miniconda3/etc/profile.d/conda.sh"
    "${HOME}/miniconda3/etc/profile.d/conda.sh"
    "/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -f "${candidate}" ]; then
      # shellcheck disable=SC1090
      source "${candidate}"
      return 0
    fi
  done

  echo "ERROR: Could not find conda.sh for NETID=${NETID}." >&2
  return 1
}

load_conda

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
set +u
conda activate "${CONDA_ENV}"
set -u

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "${CONDA_PREFIX}" ]; then
  echo "ERROR: Failed to activate conda environment: ${CONDA_ENV}" >&2
  exit 1
fi

unset PYTHONHOME
unset PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID:-$$}"

mkdir -p "${PROJECT_ROOT}/slurm_logs/svgp"
mkdir -p "${PROJECT_ROOT}/${OUTPUT_DIR}"
mkdir -p "${MPLCONFIGDIR}"

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python not found at ${PYTHON}" >&2
  exit 1
fi

ensure_svgp_python_stack() {
  if ! "${PYTHON}" -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch (torch) is not installed in ${CONDA_ENV}." >&2
    echo "Install a CUDA-appropriate torch build first (see scripts/setup_env.sh)." >&2
    exit 1
  fi

  if "${PYTHON}" -c "import gpytorch, numpy, sklearn, matplotlib" 2>/dev/null; then
    return 0
  fi

  echo "Installing missing SVGP packages into ${CONDA_ENV} (gpytorch, numpy, scikit-learn, matplotlib) ..."
  "${PYTHON}" -m pip install --no-cache-dir gpytorch numpy scikit-learn matplotlib
}

if [ "${INSTALL_SVGP_DEPS}" != "0" ]; then
  ensure_svgp_python_stack
fi

"${PYTHON}" - <<'PY'
import gpytorch
import torch
print(f"torch import OK: {torch.__version__}")
print(f"gpytorch import OK: {gpytorch.__version__}")
PY

cd "${PROJECT_ROOT}"

echo "=============================================================================="
echo "Experiment 1: GPyTorch SVGP Binary GP Classification"
echo "=============================================================================="
echo "Job ID              : ${SLURM_JOB_ID:-N/A}"
echo "Node                : $(hostname)"
echo "Start time          : $(date)"
echo "Project root        : ${PROJECT_ROOT}"
echo "NETID               : ${NETID}"
echo "Conda env           : ${CONDA_ENV}"
echo "Dataset             : ${DATASET}"
echo "Embedding dir       : ${EMBEDDING_DIR}"
echo "Output dir          : ${OUTPUT_DIR}"
echo "Device              : ${DEVICE}"
echo "Num inducing        : ${NUM_INDUCING}"
echo "Batch size          : ${BATCH_SIZE}"
echo "Predict batch size  : ${PREDICT_BATCH_SIZE}"
echo "Epochs              : ${EPOCHS}"
echo "Patience            : ${PATIENCE}"
echo "Learning rate       : ${LEARNING_RATE}"
echo "Max train samples   : ${MAX_TRAIN_SAMPLES}"
echo "Max val samples     : ${MAX_VAL_SAMPLES}"
echo "Max test samples    : ${MAX_TEST_SAMPLES}"
echo "Disable standardize : ${DISABLE_STANDARDIZE}"
echo "=============================================================================="

CMD=(
  "${PYTHON}" "experiments/exp1_gpytorch_svgp_gpc.py"
  "--dataset" "${DATASET}"
  "--embedding-dir" "${EMBEDDING_DIR}"
  "--output-dir" "${OUTPUT_DIR}"
  "--device" "${DEVICE}"
  "--num-inducing" "${NUM_INDUCING}"
  "--batch-size" "${BATCH_SIZE}"
  "--predict-batch-size" "${PREDICT_BATCH_SIZE}"
  "--epochs" "${EPOCHS}"
  "--patience" "${PATIENCE}"
  "--learning-rate" "${LEARNING_RATE}"
  "--max-train-samples" "${MAX_TRAIN_SAMPLES}"
  "--max-val-samples" "${MAX_VAL_SAMPLES}"
  "--max-test-samples" "${MAX_TEST_SAMPLES}"
)

if [ "${DISABLE_STANDARDIZE}" = "1" ]; then
  CMD+=("--disable-standardize")
fi

"${CMD[@]}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Expected outputs in ${PROJECT_ROOT}/${OUTPUT_DIR}:"
echo "  - exp1_gpytorch_svgp_${DATASET}_results.json"
echo "  - exp1_gpytorch_svgp_${DATASET}_posterior.npz"
echo "  - exp1_gpytorch_svgp_${DATASET}_calibration.png"
echo "  - exp1_gpytorch_svgp_${DATASET}_roc.png"
echo "=============================================================================="
