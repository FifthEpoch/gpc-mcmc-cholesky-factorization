#!/bin/bash
#SBATCH --job-name=exp1_full_hmc
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp1_full_hmc_embeddings/%x_%j.out
#SBATCH --error=slurm_logs/exp1_full_hmc_embeddings/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

# Submit with, for example:
#   sbatch --account=csci_ga_2565-2026sp --partition=<partition> \
#          --export=ALL,NETID=sd6701 scripts/exp1_predictive_embeddings_hmc.sbatch
#
# Optional overrides:
#   NETID, PROJECT_ROOT, CONDA_ENV
#   DATASET                  pcam | camelyon17 | embed  (default: pcam)
#   EMBEDDING_DIR            datasets/pcam-hg           (auto-detected)
#   TRAIN_SPLIT              train
#   TEST_SPLIT               test
#   MAX_TRAIN_SAMPLES        2000
#   MAX_TEST_SAMPLES         2000
#   STANDARDIZE              1 | 0
#   KERNEL_BANDWIDTH         20.0
#   N_SAMPLES                1000
#   N_WARMUP                 200
#   HMC_STEP_SIZE            empty means 4/sqrt(n_train)
#   N_LEAPFROG               12
#   N_CONDITIONAL_DRAWS      5
#   OUTPUT_DIR               data/full_hmc_embeddings
#   SEED                     42

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

SBATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${SBATCH_DIR}/.." && pwd)}}"

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

SCRATCH_BASE="/scratch/${NETID}"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"
DATASET="${DATASET:-pcam}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
TEST_SPLIT="${TEST_SPLIT:-test}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-2000}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-2000}"
STANDARDIZE="${STANDARDIZE:-1}"
KERNEL_BANDWIDTH="${KERNEL_BANDWIDTH:-20.0}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_WARMUP="${N_WARMUP:-200}"
HMC_STEP_SIZE="${HMC_STEP_SIZE:-}"
N_LEAPFROG="${N_LEAPFROG:-12}"
N_CONDITIONAL_DRAWS="${N_CONDITIONAL_DRAWS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-data/full_hmc_embeddings}"
SEED="${SEED:-42}"

if [ -z "${EMBEDDING_DIR:-}" ]; then
  if [ -d "${PROJECT_ROOT}/datasets/pcam-hg/train/embeddings" ]; then
    EMBEDDING_DIR="datasets/pcam-hg"
  else
    EMBEDDING_DIR="data/embeddings"
  fi
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

mkdir -p "${PROJECT_ROOT}/slurm_logs/exp1_full_hmc_embeddings"
mkdir -p "${PROJECT_ROOT}/${OUTPUT_DIR}"
mkdir -p "${MPLCONFIGDIR}"

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python not found at ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" -c "import numpy, scipy, emcee; print('Env import check: OK')" || {
  echo "ERROR: Import check failed in ${CONDA_ENV}." >&2
  exit 1
}

cd "${PROJECT_ROOT}"

if [ ! -f "experiments/exp1_predictive_embeddings_hmc.py" ]; then
  echo "ERROR: experiments/exp1_predictive_embeddings_hmc.py not found under ${PROJECT_ROOT}" >&2
  exit 1
fi

echo "=============================================================================="
echo "Experiment 1: full HMC GP classification on embedding subsample"
echo "=============================================================================="
echo "Job ID                 : ${SLURM_JOB_ID:-N/A}"
echo "Node                   : $(hostname)"
echo "Start time             : $(date)"
echo "Project root           : ${PROJECT_ROOT}"
echo "NETID                  : ${NETID}"
echo "Conda env              : ${CONDA_ENV}"
echo "Dataset                : ${DATASET}"
echo "Embedding dir          : ${EMBEDDING_DIR}"
echo "Train split            : ${TRAIN_SPLIT}"
echo "Test split             : ${TEST_SPLIT}"
echo "Max train samples      : ${MAX_TRAIN_SAMPLES}"
echo "Max test samples       : ${MAX_TEST_SAMPLES}"
echo "Standardize            : ${STANDARDIZE}"
echo "Kernel bandwidth       : ${KERNEL_BANDWIDTH}"
echo "HMC samples/warmup     : ${N_SAMPLES} / ${N_WARMUP}"
echo "HMC step size          : ${HMC_STEP_SIZE:-auto}"
echo "HMC leapfrog           : ${N_LEAPFROG}"
echo "Conditional draws      : ${N_CONDITIONAL_DRAWS}"
echo "Output dir             : ${OUTPUT_DIR}"
echo "Seed                   : ${SEED}"
echo "=============================================================================="

CMD=(
  "${PYTHON}" "experiments/exp1_predictive_embeddings_hmc.py"
  "--dataset" "${DATASET}"
  "--embedding-dir" "${EMBEDDING_DIR}"
  "--train-split" "${TRAIN_SPLIT}"
  "--test-split" "${TEST_SPLIT}"
  "--max-train-samples" "${MAX_TRAIN_SAMPLES}"
  "--max-test-samples" "${MAX_TEST_SAMPLES}"
  "--kernel-bandwidth" "${KERNEL_BANDWIDTH}"
  "--n-samples" "${N_SAMPLES}"
  "--n-warmup" "${N_WARMUP}"
  "--n-leapfrog" "${N_LEAPFROG}"
  "--n-conditional-draws" "${N_CONDITIONAL_DRAWS}"
  "--output-dir" "${OUTPUT_DIR}"
  "--seed" "${SEED}"
)

if [ "${STANDARDIZE}" != "1" ]; then
  CMD+=("--no-standardize")
fi
if [ -n "${HMC_STEP_SIZE}" ]; then
  CMD+=("--hmc-step-size" "${HMC_STEP_SIZE}")
fi

"${CMD[@]}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Artifacts written under ${PROJECT_ROOT}/${OUTPUT_DIR}"
echo "=============================================================================="
