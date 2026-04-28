#!/bin/bash
#SBATCH --job-name=exp2_rose_leapfrog_sweep
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp2_rose_leapfrog_sweep/%x_%j.out
#SBATCH --error=slurm_logs/exp2_rose_leapfrog_sweep/%x_%j.err

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"

PROJECT_ROOT="/scratch/sd6701/gpc-mcmc-cholesky-factorization"
cd "${PROJECT_ROOT}"

mkdir -p slurm_logs/exp2_rose_leapfrog_sweep

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

TEST_EMB="datasets/pcam-hg/test/embeddings/projected_512.npy"
TEST_LABELS="datasets/pcam-hg/test/embeddings/y_embeddings.npy"

FACTOR_DIR="/scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets/pcam-hg/train/embeddings/exp2_rpchol_50k"
EXP2_ROOT="data/exp2_leapfrog_sweep"
K=200
N_SAMPLES=1000
N_WARMUP=500
LF_VALUES=(15 25 35 50 75 100)

echo "Running exp2.py for k=${K}, n_samples=${N_SAMPLES}, n_warmup=${N_WARMUP}, and leapfrog=${LF_VALUES[*]}"
echo "Factor dir: ${FACTOR_DIR}"
for LF in "${LF_VALUES[@]}"; do
    OUT_DIR="${EXP2_ROOT}/k${K}_lf${LF}"
    mkdir -p "${OUT_DIR}"

    echo "------------------------------------------------------------"
    echo "Running exp2.py: k=${K}, n_leapfrog=${LF}, n_samples=${N_SAMPLES}, n_warmup=${N_WARMUP}"
    echo "Output dir: ${OUT_DIR}"
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
done

echo "Job finished at $(date)"