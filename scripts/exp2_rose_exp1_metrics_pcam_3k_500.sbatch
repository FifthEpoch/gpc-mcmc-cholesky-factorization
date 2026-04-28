#!/bin/bash
#SBATCH --job-name=exp2_rose_pcam_3k_500
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/hmc_rose/%x_%j.out
#SBATCH --error=slurm_logs/hmc_rose/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

# Submit with:
#   sbatch --export=ALL,NETID=<your-netid> scripts/exp2_rose_exp1_metrics_pcam_3k_500.sbatch
#
# Optional overrides:
#   PROJECT_ROOT, CONDA_ENV, DATASET_ROOT
#   K_VALUE, BANDWIDTH, BLOCK_SIZE, STOPTOL, SEED
#   N_SAMPLES, N_WARMUP, N_LEAPFROG, PREDICTION_BATCH_SIZE
#   FACTOR_DIR, OUTPUT_DIR

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

SCRATCH_BASE="/scratch/${NETID}"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"

DATASET_ROOT="${DATASET_ROOT:-datasets/pcam-hg}"
TRAIN_EMBEDDINGS="${TRAIN_EMBEDDINGS:-${DATASET_ROOT}/train/embeddings/projected_512.npy}"
TRAIN_LABELS="${TRAIN_LABELS:-${DATASET_ROOT}/train/embeddings/y_embeddings.npy}"
TEST_EMBEDDINGS="${TEST_EMBEDDINGS:-${DATASET_ROOT}/test/embeddings/projected_512.npy}"
TEST_LABELS="${TEST_LABELS:-${DATASET_ROOT}/test/embeddings/y_embeddings.npy}"

TRAIN_SUBSAMPLE="${TRAIN_SUBSAMPLE:-3000}"
TEST_SUBSAMPLE="${TEST_SUBSAMPLE:-500}"
K_VALUE="${K_VALUE:-200}"
BANDWIDTH="${BANDWIDTH:-approx_median}"
BLOCK_SIZE="${BLOCK_SIZE:-10}"
STOPTOL="${STOPTOL:-1e-13}"
SEED="${SEED:-42}"

N_SAMPLES="${N_SAMPLES:-2000}"
N_WARMUP="${N_WARMUP:-200}"
N_LEAPFROG="${N_LEAPFROG:-12}"
PREDICTION_BATCH_SIZE="${PREDICTION_BATCH_SIZE:-512}"

FACTOR_DIR="${FACTOR_DIR:-${DATASET_ROOT}/train/embeddings/exp2_rpchol_3k_stratified_standardized}"
OUTPUT_DIR="${OUTPUT_DIR:-data/exp2_rose_exp1_metrics_pcam_3k_500}"

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

PYTHON="${CONDA_PREFIX}/bin/python"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs/hmc_rose "${FACTOR_DIR}" "${OUTPUT_DIR}" "${MPLCONFIGDIR}"

"${PYTHON}" -c "import numpy, scipy, emcee; print('Env import check: OK')" || {
  echo "ERROR: Import check failed in ${CONDA_ENV}." >&2
  exit 1
}

for path in \
  "${TRAIN_EMBEDDINGS}" \
  "${TRAIN_LABELS}" \
  "${TEST_EMBEDDINGS}" \
  "${TEST_LABELS}" \
  "experiments/exp_rpcholesky_embeddings.py" \
  "experiments/exp2_rose_exp1_metrics.py"; do
  if [ ! -f "${path}" ]; then
    echo "ERROR: Missing required file: ${PROJECT_ROOT}/${path}" >&2
    exit 1
  fi
done

echo "=============================================================================="
echo "Experiment 2 Rose, exp1-style PCam metrics"
echo "=============================================================================="
echo "Job ID                 : ${SLURM_JOB_ID:-N/A}"
echo "Node                   : $(hostname)"
echo "Start time             : $(date)"
echo "Project root           : ${PROJECT_ROOT}"
echo "Conda env              : ${CONDA_ENV}"
echo "Train embeddings       : ${TRAIN_EMBEDDINGS}"
echo "Train labels           : ${TRAIN_LABELS}"
echo "Test embeddings        : ${TEST_EMBEDDINGS}"
echo "Test labels            : ${TEST_LABELS}"
echo "Train subsample        : ${TRAIN_SUBSAMPLE} stratified, seed ${SEED}"
echo "Test subsample         : ${TEST_SUBSAMPLE} stratified, seed $((SEED + 1))"
echo "Standardization        : train moments saved in factor dir, applied to test"
echo "Rank k                 : ${K_VALUE}"
echo "Bandwidth              : ${BANDWIDTH}"
echo "Factor dir             : ${FACTOR_DIR}"
echo "Output dir             : ${OUTPUT_DIR}"
echo "HMC samples/warmup     : ${N_SAMPLES}/${N_WARMUP}"
echo "Leapfrog steps         : ${N_LEAPFROG}"
echo "=============================================================================="

echo "[1/2] Building stratified standardized RP-Cholesky factor on 3000 PCam train rows..."
"${PYTHON}" experiments/exp_rpcholesky_embeddings.py \
  --embeddings "${TRAIN_EMBEDDINGS}" \
  --labels "${TRAIN_LABELS}" \
  --subsample "${TRAIN_SUBSAMPLE}" \
  --stratified-subsample \
  --standardize \
  --k-values "${K_VALUE}" \
  --output-dir "${FACTOR_DIR}" \
  --bandwidth "${BANDWIDTH}" \
  --block-size "${BLOCK_SIZE}" \
  --stoptol "${STOPTOL}" \
  --seed "${SEED}"

echo "[2/2] Running exp2_rose_exp1_metrics on 500 stratified PCam test rows..."
"${PYTHON}" experiments/exp2_rose_exp1_metrics.py \
  --factor-dir "${FACTOR_DIR}" \
  --test-embeddings "${TEST_EMBEDDINGS}" \
  --test-labels "${TEST_LABELS}" \
  --k "${K_VALUE}" \
  --subsample-test "${TEST_SUBSAMPLE}" \
  --stratified-subsample-test \
  --standardize-from-factor \
  --n-samples "${N_SAMPLES}" \
  --n-warmup "${N_WARMUP}" \
  --n-leapfrog "${N_LEAPFROG}" \
  --prediction-batch-size "${PREDICTION_BATCH_SIZE}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Artifacts:"
echo "  Factor dir : ${PROJECT_ROOT}/${FACTOR_DIR}"
echo "  Results    : ${PROJECT_ROOT}/${OUTPUT_DIR}"
echo "=============================================================================="
