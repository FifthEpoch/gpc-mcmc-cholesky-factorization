#!/bin/bash
#SBATCH --job-name=exp2_metrics_full
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
#   sbatch --export=ALL,NETID=<your-netid> scripts/exp2_rose_exp1_metrics_pcam_full_precomputed.sbatch
#
# This script does not build or overwrite RP-Cholesky factors. It only runs
# exp2_rose_exp1_metrics.py against a precomputed factor directory.

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

FACTOR_DIR="${FACTOR_DIR:-datasets/pcam-hg/train/embeddings/exp2_rpchol_full}"
TEST_EMBEDDINGS="${TEST_EMBEDDINGS:-datasets/pcam-hg/test/embeddings/projected_512.npy}"
TEST_LABELS="${TEST_LABELS:-datasets/pcam-hg/test/embeddings/y_embeddings.npy}"
K_VALUE="${K_VALUE:-200}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_WARMUP="${N_WARMUP:-1000}"
N_LEAPFROG="${N_LEAPFROG:-25}"
PREDICTION_BATCH_SIZE="${PREDICTION_BATCH_SIZE:-512}"
HMC_STEP_CONSTANT="${HMC_STEP_CONSTANT:-0.05}"
HMC_TARGET_ACCEPT="${HMC_TARGET_ACCEPT:-0.8}"
TEST_SUBSAMPLE="${TEST_SUBSAMPLE:-}"
STRATIFIED_SUBSAMPLE_TEST="${STRATIFIED_SUBSAMPLE_TEST:-0}"
STANDARDIZE_FROM_FACTOR="${STANDARDIZE_FROM_FACTOR:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-data/exp2_rose_exp1_metrics_pcam_full_k200_lf25_s1000_w1000}"
SEED="${SEED:-42}"

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
mkdir -p slurm_logs/hmc_rose "${OUTPUT_DIR}" "${MPLCONFIGDIR}"

"${PYTHON}" -c "import numpy, scipy, emcee; print('Env import check: OK')" || {
  echo "ERROR: Import check failed in ${CONDA_ENV}." >&2
  exit 1
}

for path in \
  "${FACTOR_DIR}/summary.json" \
  "${FACTOR_DIR}/factor_k${K_VALUE}.npy" \
  "${FACTOR_DIR}/pivots_k${K_VALUE}.npy" \
  "${FACTOR_DIR}/kernel_submatrix_k${K_VALUE}.npy" \
  "${FACTOR_DIR}/embeddings.npy" \
  "${FACTOR_DIR}/labels.npy" \
  "${TEST_EMBEDDINGS}" \
  "${TEST_LABELS}" \
  "experiments/exp2_rose_exp1_metrics.py"; do
  if [ ! -f "${path}" ]; then
    echo "ERROR: Missing required file: ${PROJECT_ROOT}/${path}" >&2
    exit 1
  fi
done

echo "=============================================================================="
echo "Experiment 2 Rose exp1 metrics on precomputed full PCam factor"
echo "=============================================================================="
echo "Job ID                 : ${SLURM_JOB_ID:-N/A}"
echo "Node                   : $(hostname)"
echo "Start time             : $(date)"
echo "Project root           : ${PROJECT_ROOT}"
echo "Conda env              : ${CONDA_ENV}"
echo "Factor dir             : ${FACTOR_DIR}"
echo "Test embeddings        : ${TEST_EMBEDDINGS}"
echo "Test labels            : ${TEST_LABELS}"
echo "Rank k                 : ${K_VALUE}"
echo "HMC samples/warmup     : ${N_SAMPLES}/${N_WARMUP}"
echo "Leapfrog steps         : ${N_LEAPFROG}"
echo "Bandwidth              : loaded from ${FACTOR_DIR}/summary.json"
echo "Test subsample         : ${TEST_SUBSAMPLE:-<full test split>}"
echo "Standardize from factor: ${STANDARDIZE_FROM_FACTOR}"
echo "Output dir             : ${OUTPUT_DIR}"
echo "Seed                   : ${SEED}"
echo "=============================================================================="

CMD=(
  "${PYTHON}" "experiments/exp2_rose_exp1_metrics.py"
  "--factor-dir" "${FACTOR_DIR}"
  "--test-embeddings" "${TEST_EMBEDDINGS}"
  "--test-labels" "${TEST_LABELS}"
  "--k" "${K_VALUE}"
  "--n-samples" "${N_SAMPLES}"
  "--n-warmup" "${N_WARMUP}"
  "--n-leapfrog" "${N_LEAPFROG}"
  "--prediction-batch-size" "${PREDICTION_BATCH_SIZE}"
  "--hmc-step-constant" "${HMC_STEP_CONSTANT}"
  "--hmc-target-accept" "${HMC_TARGET_ACCEPT}"
  "--seed" "${SEED}"
  "--output-dir" "${OUTPUT_DIR}"
)

if [ -n "${TEST_SUBSAMPLE}" ]; then
  CMD+=("--subsample-test" "${TEST_SUBSAMPLE}")
fi
if [ "${STRATIFIED_SUBSAMPLE_TEST}" = "1" ]; then
  CMD+=("--stratified-subsample-test")
fi
if [ "${STANDARDIZE_FROM_FACTOR}" = "1" ]; then
  CMD+=("--standardize-from-factor")
fi

"${CMD[@]}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Artifacts written under ${PROJECT_ROOT}/${OUTPUT_DIR}"
echo "=============================================================================="
