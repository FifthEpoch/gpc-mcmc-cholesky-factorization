#!/bin/bash
#SBATCH --job-name=exp2_rose_k_sweep
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp2_rose_k_sweep/%x_%j.out
#SBATCH --error=slurm_logs/exp2_rose_k_sweep/%x_%j.err

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"

PROJECT_ROOT="/scratch/sd6701/gpc-mcmc-cholesky-factorization"
cd "${PROJECT_ROOT}"

mkdir -p slurm_logs/exp2_rose_k_sweep

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

TRAIN_EMB="datasets/pcam-hg/train/embeddings/projected_512.npy"
TRAIN_LABELS="datasets/pcam-hg/train/embeddings/y_embeddings.npy"
TEST_EMB="datasets/pcam-hg/test/embeddings/projected_512.npy"
TEST_LABELS="datasets/pcam-hg/test/embeddings/y_embeddings.npy"

FACTOR_ROOT="datasets/pcam-hg/train/embeddings"
EXP2_ROOT="data/exp2_sweep"
K_VALUES=(100 200 300 500)
LF_VALUES=(12 25)

echo "Running exp2.py for k=${K_VALUES[*]} and leapfrog=${LF_VALUES[*]}"
for K in "${K_VALUES[@]}"; do
    FACTOR_DIR="${FACTOR_ROOT}/exp2_50k_k=${K}"

    echo "============================================================"
    echo "Generating 50k RPCholesky factor for k=${K}"
    echo "Factor dir: ${FACTOR_DIR}"
    python experiments/exp_rpcholesky_embeddings.py \
        --embeddings "${TRAIN_EMB}" \
        --labels "${TRAIN_LABELS}" \
        --subsample 50000 \
        --k-values "${K}" \
        --kernel gaussian \
        --bandwidth approx_median \
        --block-size 10 \
        --stoptol 1e-13 \
        --output-dir "${FACTOR_DIR}" \
        --seed 42

    for LF in "${LF_VALUES[@]}"; do
        OUT_DIR="${EXP2_ROOT}/k${K}_lf${LF}"
        mkdir -p "${OUT_DIR}"

        echo "------------------------------------------------------------"
        echo "Running exp2.py: k=${K}, n_leapfrog=${LF}, n_samples=500, n_warmup=750"
        echo "Output dir: ${OUT_DIR}"
        python experiments/exp2.py \
            --factor-dir "${FACTOR_DIR}" \
            --test-embeddings "${TEST_EMB}" \
            --test-labels "${TEST_LABELS}" \
            --k "${K}" \
            --n-samples 500 \
            --n-warmup 750 \
            --n-leapfrog "${LF}" \
            --output-dir "${OUT_DIR}" \
            --seed 42
    done
done

echo "Job finished at $(date)"
