#!/bin/bash
#SBATCH --job-name=exp0_rpcholesky
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

# =============================================================================
# User-configurable paths  (EDIT THIS SECTION BEFORE YOUR FIRST SUBMISSION)
# =============================================================================
#
# On this cluster, /home has a very small quota. All large files -- including
# conda environments -- should live under /scratch/<NETID>.
#
# STEP 1: Set your NetID.
#   Either edit the line below, or pass it at submit time:
#     sbatch --export=ALL,NETID=ab1234 scripts/exp0_algorithm_verification.sbatch
#
NETID="${NETID:-<netid>}"
if [ "${NETID}" = "<netid>" ]; then
  echo "ERROR: NETID is required." >&2
  echo "  Option A: edit NETID in this file        (line above)" >&2
  echo "  Option B: sbatch --export=ALL,NETID=ab1234 scripts/exp0_algorithm_verification.sbatch" >&2
  exit 1
fi

SCRATCH_BASE="/scratch/${NETID}"

# STEP 2: Point to your conda environment.
#   Your conda env must live under /scratch/<NETID> (not /home) due to the
#   cluster's disk quota policy. Typical locations:
#     /scratch/<NETID>/conda-envs/gpc
#     /scratch/<NETID>/.conda/envs/gpc
#     /scratch/<NETID>/miniconda3/envs/gpc
#
#   Option A: edit the default path below.
#   Option B: pass it at submit time:
#     sbatch --export=ALL,NETID=ab1234,CONDA_ENV=/scratch/ab1234/my-env \
#            scripts/exp0_algorithm_verification.sbatch
#
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"

# Auto-discovery: if the default path doesn't exist, check common locations.
if [ ! -d "${CONDA_ENV}" ]; then
  echo "WARN: ${CONDA_ENV} not found. Searching common locations..." >&2
  for candidate in \
    "${SCRATCH_BASE}/conda-envs/gpc" \
    "${SCRATCH_BASE}/.conda/envs/gpc" \
    "${SCRATCH_BASE}/miniconda3/envs/gpc" \
    "${SCRATCH_BASE}/anaconda3/envs/gpc"; do
    if [ -d "${candidate}" ]; then
      echo "  Found conda env at: ${candidate}" >&2
      CONDA_ENV="${candidate}"
      break
    fi
  done
fi

# STEP 3 (optional): Override the project root if your clone is somewhere else.
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/gpc-mcmc-cholesky-factorization}"

#
# If you haven't created a conda env yet, run:
#   conda create -p /scratch/<NETID>/conda-envs/gpc python=3.12 -y
#   conda activate /scratch/<NETID>/conda-envs/gpc
#   pip install numpy scipy scikit-learn matplotlib tqdm gdown
#

# =============================================================================
# Conda activation (cluster-specific, CRITICAL sequence)
#
# Steps that MUST happen in this exact order:
#   1. module purge           - clear any stale module state
#   2. module load anaconda3  - load the cluster's base anaconda
#   3. source conda.sh        - make `conda activate` available in bash
#   4. conda activate <env>   - switch into user env
#   5. unset PYTHONHOME/PATH  - the module load leaks these, breaking imports
# =============================================================================
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

if [ ! -d "${CONDA_ENV}" ]; then
  echo "ERROR: Conda env not found at ${CONDA_ENV}" >&2
  exit 1
fi

conda activate "${CONDA_ENV}"
echo "Activated conda env: ${CONDA_ENV}"

# CRITICAL: module load anaconda3 sets PYTHONHOME and PYTHONPATH to the
# system anaconda, which poisons the user env's sys.path and causes
# "No module named ..." errors. We MUST unset them after conda activate.
unset PYTHONHOME
unset PYTHONPATH

PYTHON="${CONDA_ENV}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python not found at ${PYTHON}" >&2
  exit 1
fi
echo "Using: ${PYTHON} ($(${PYTHON} --version 2>&1))"

"${PYTHON}" -c "import numpy, scipy, sklearn, matplotlib; print('Env import check: OK')" || {
  echo "ERROR: import check failed -- conda activation is likely broken." >&2
  exit 1
}

export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# =============================================================================
# Run Experiment 0
# =============================================================================
cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
mkdir -p data

echo "=============================================================================="
echo "Experiment 0: RPCholesky algorithm verification"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "Project root : ${PROJECT_ROOT}"
echo "Conda env    : ${CONDA_ENV}"
echo "=============================================================================="

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

"${PYTHON}" experiments/exp0_algorithm_verification.py

echo "=============================================================================="
echo "Done: $(date)"
echo "Expected outputs in ${PROJECT_ROOT}/data:"
echo "  - exp0_results.mat"
echo "  - exp0_error_vs_k.png"
echo "  - exp0_time_vs_k.png"
echo "  - exp0_time_vs_N.png"
echo "=============================================================================="
