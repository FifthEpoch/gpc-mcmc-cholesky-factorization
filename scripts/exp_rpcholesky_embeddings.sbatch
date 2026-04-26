#!/bin/bash
#SBATCH --job-name=rpchol_embed
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --output=slurm_logs/rpcholesky_embeddings/%x_%j.out
#SBATCH --error=slurm_logs/rpcholesky_embeddings/%x_%j.err

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

SCRATCH_BASE="/scratch/${NETID}"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"
DATASET_ROOT="${DATASET_ROOT:-datasets/pcam-hg}"
SPLITS="${SPLITS:-train valid test}"
SUBSAMPLE_SIZES="${SUBSAMPLE_SIZES:-5000 10000 20000 50000 full}"
K_VALUE="${K_VALUE:-200}"
BANDWIDTH="${BANDWIDTH:-approx_median}"
BLOCK_SIZE="${BLOCK_SIZE:-10}"
STOPTOL="${STOPTOL:-1e-13}"
SEED="${SEED:-42}"
EMBEDDING_FILENAME="${EMBEDDING_FILENAME:-projected_512.npy}"
LABEL_FILENAME="${LABEL_FILENAME:-y_embeddings.npy}"

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

mkdir -p "${PROJECT_ROOT}/slurm_logs/rpcholesky_embeddings"
mkdir -p "${MPLCONFIGDIR}"

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python not found at ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" -c "import numpy, scipy; print('Env import check: OK')" || {
  echo "ERROR: Import check failed in ${CONDA_ENV}." >&2
  exit 1
}

cd "${PROJECT_ROOT}"

if [ ! -f "experiments/exp_rpcholesky_embeddings.py" ]; then
  echo "ERROR: experiments/exp_rpcholesky_embeddings.py not found under ${PROJECT_ROOT}" >&2
  exit 1
fi

size_tag() {
  case "$1" in
    5000) echo "5k" ;;
    10000) echo "10k" ;;
    20000) echo "20k" ;;
    50000) echo "50k" ;;
    full) echo "full" ;;
    *) echo "$1" ;;
  esac
}

row_count() {
  "${PYTHON}" -c 'import numpy as np, sys; print(np.load(sys.argv[1], mmap_mode="r").shape[0])' "$1"
}

move_outputs() {
  local tmp_dir="$1"
  local dest_dir="$2"
  local tag="$3"
  local k="$4"

  local factor_src=("${tmp_dir}"/factor_k*.npy)
  local pivots_src=("${tmp_dir}"/pivots_k*.npy)
  local kernel_src=("${tmp_dir}"/kernel_submatrix_k*.npy)

  if [ ${#factor_src[@]} -ne 1 ] || [ ${#pivots_src[@]} -ne 1 ] || [ ${#kernel_src[@]} -ne 1 ]; then
    echo "ERROR: Expected exactly one factor/pivots/kernel_submatrix file in ${tmp_dir}" >&2
    exit 1
  fi

  mv -f "${factor_src[0]}" "${dest_dir}/factor_${tag}_k${k}.npy"
  mv -f "${pivots_src[0]}" "${dest_dir}/pivots_${tag}_k${k}.npy"
  mv -f "${kernel_src[0]}" "${dest_dir}/kernel_submatrix_${tag}_k${k}.npy"
  mv -f "${tmp_dir}/embeddings.npy" "${dest_dir}/embeddings_${tag}_k${k}.npy"
  if [ -f "${tmp_dir}/labels.npy" ]; then
    mv -f "${tmp_dir}/labels.npy" "${dest_dir}/labels_${tag}_k${k}.npy"
  fi
  mv -f "${tmp_dir}/summary.json" "${dest_dir}/summary_${tag}_k${k}.json"
}

echo "=============================================================================="
echo "Experiment: RPCholesky on PCam-HG projected 512-d embeddings"
echo "=============================================================================="
echo "Job ID             : ${SLURM_JOB_ID:-N/A}"
echo "Node               : $(hostname)"
echo "Start time         : $(date)"
echo "Project root       : ${PROJECT_ROOT}"
echo "NETID              : ${NETID}"
echo "Conda env          : ${CONDA_ENV}"
echo "Dataset root       : ${DATASET_ROOT}"
echo "Splits             : ${SPLITS}"
echo "Subsample sizes    : ${SUBSAMPLE_SIZES}"
echo "Embedding file     : ${EMBEDDING_FILENAME}"
echo "Label file         : ${LABEL_FILENAME}"
echo "k                  : ${K_VALUE}"
echo "Bandwidth          : ${BANDWIDTH}"
echo "Block size         : ${BLOCK_SIZE}"
echo "Stop tol           : ${STOPTOL}"
echo "Seed               : ${SEED}"
echo "=============================================================================="

for split in ${SPLITS}; do
  split_dir="${PROJECT_ROOT}/${DATASET_ROOT}/${split}/embeddings"
  emb_path="${split_dir}/${EMBEDDING_FILENAME}"
  lbl_path="${split_dir}/${LABEL_FILENAME}"

  if [ ! -f "${emb_path}" ]; then
    echo "ERROR: Missing embeddings file for split=${split}: ${emb_path}" >&2
    exit 1
  fi
  if [ ! -f "${lbl_path}" ]; then
    echo "ERROR: Missing labels file for split=${split}: ${lbl_path}" >&2
    exit 1
  fi

  n_rows="$(row_count "${emb_path}")"
  echo
  echo "Split ${split}: ${n_rows} rows"

  for requested_size in ${SUBSAMPLE_SIZES}; do
    tag="$(size_tag "${requested_size}")"
    tmp_dir="${TMPDIR:-/tmp}/rpchol_${split}_${tag}_k${K_VALUE}_${SLURM_JOB_ID:-$$}"
    rm -rf "${tmp_dir}"
    mkdir -p "${tmp_dir}"

    cmd=(
      "${PYTHON}" "experiments/exp_rpcholesky_embeddings.py"
      "--embeddings" "${emb_path}"
      "--labels" "${lbl_path}"
      "--k-values" "${K_VALUE}"
      "--output-dir" "${tmp_dir}"
      "--bandwidth" "${BANDWIDTH}"
      "--block-size" "${BLOCK_SIZE}"
      "--stoptol" "${STOPTOL}"
      "--seed" "${SEED}"
    )

    if [ "${requested_size}" = "full" ]; then
      echo "[${split}] running full dataset -> *_full_k${K_VALUE}"
    else
      if [ "${requested_size}" -gt "${n_rows}" ]; then
        echo "[${split}] skipping ${requested_size} rows (split only has ${n_rows})"
        rm -rf "${tmp_dir}"
        continue
      fi
      echo "[${split}] running subsample=${requested_size} -> *_${tag}_k${K_VALUE}"
      cmd+=("--subsample" "${requested_size}")
    fi

    "${cmd[@]}"
    move_outputs "${tmp_dir}" "${split_dir}" "${tag}" "${K_VALUE}"
    rm -rf "${tmp_dir}"

    echo "[${split}] saved:"
    echo "  ${split_dir}/factor_${tag}_k${K_VALUE}.npy"
    echo "  ${split_dir}/pivots_${tag}_k${K_VALUE}.npy"
    echo "  ${split_dir}/kernel_submatrix_${tag}_k${K_VALUE}.npy"
    echo "  ${split_dir}/embeddings_${tag}_k${K_VALUE}.npy"
    echo "  ${split_dir}/labels_${tag}_k${K_VALUE}.npy"
    echo "  ${split_dir}/summary_${tag}_k${K_VALUE}.json"
  done
done

echo "=============================================================================="
echo "Done: $(date)"
echo "Artifacts were written directly into:"
echo "  ${PROJECT_ROOT}/${DATASET_ROOT}/{train,valid,test}/embeddings/"
echo "using names like factor_5k_k${K_VALUE}.npy and summary_full_k${K_VALUE}.json"
echo "=============================================================================="
