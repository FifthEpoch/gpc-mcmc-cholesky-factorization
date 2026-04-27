#!/bin/bash
#SBATCH --job-name=exp2_rose_full_first
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp2_rose_full_first/%x_%j.out
#SBATCH --error=slurm_logs/exp2_rose_full_first/%x_%j.err

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"

PROJECT_ROOT="/scratch/sd6701/gpc-mcmc-cholesky-factorization"
cd "${PROJECT_ROOT}"

mkdir -p slurm_logs/exp2_rose_full_first

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

TEST_EMB="datasets/pcam-hg/test/embeddings/projected_512.npy"
TEST_LABELS="datasets/pcam-hg/test/embeddings/y_embeddings.npy"

FACTOR_DIR="datasets/pcam-hg/train/embeddings/exp2_rpchol_full"
OUT_DIR="data/exp2_full_first"

K=200
LF=25
N_WARMUP=1000
N_SAMPLES=1000

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Running exp2.py with existing full RPCholesky factor"
echo "Factor dir: ${PROJECT_ROOT}/${FACTOR_DIR}"
echo "Output dir: ${OUT_DIR}"
echo "k=${K}, n_leapfrog=${LF}, n_samples=${N_SAMPLES}, n_warmup=${N_WARMUP}"

python experiments/exp2.py \
    --factor-dir "${FACTOR_DIR}" \
    --test-embeddings "${TEST_EMB}" \
    --test-labels "${TEST_LABELS}" \
    --k "${K}" \
    --n-samples "${N_SAMPLES}" \
    --n-warmup "${N_WARMUP}" \
    --n-leapfrog "${LF}" \
    --output-dir "${OUT_DIR}" \
    --seed 42

echo "Job finished at $(date)"
