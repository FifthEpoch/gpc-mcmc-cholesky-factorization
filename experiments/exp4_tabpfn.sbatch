#!/bin/bash
#SBATCH --job-name=exp4_tabpfn_test
#SBATCH --output=logs/exp4_tabpfn_test_%j.out
#SBATCH --error=logs/exp4_tabpfn_test_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=c12m85-a100-1
#SBATCH --account=csci_ga_2565-2026sp

set -euo pipefail

# ----- Environment setup ------------------------------------------------
# Adjust these blocks to whatever your cluster uses (modules / conda / venv).
# Pick ONE of the activation methods below and delete the others.

# module purge
# module load cuda/12.1
# module load anaconda3

# Conda env:
# source activate my_cholesky_env

# OR venv:
# source ~/venvs/my_cholesky/bin/activate

# TabPFN caches model weights here on first run; point it at scratch/persistent
# storage so re-runs don't re-download the weights.
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-$HOME/.cache/tabpfn}"
mkdir -p "$TABPFN_MODEL_CACHE_DIR"

# Quieter HF / tokenizers warnings
export TOKENIZERS_PARALLELISM=false

# ----- Job context (handy for debugging from the log) -------------------
echo "Job ID:      ${SLURM_JOB_ID:-n/a}"
echo "Node:        $(hostname)"
echo "Date:        $(date)"
echo "CWD:         $(pwd)"
echo "Python:      $(which python)"
echo "Python ver:  $(python --version 2>&1)"
nvidia-smi || true

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs data

# ----- Run experiment ---------------------------------------------------
# PCam has ~270k train rows; --max-train-samples controls the stratified
# subsample fed to TabPFN's in-context fit. 3500 is the safe default for
# a 16 GB GPU; bump it on larger cards (and add --ignore-pretraining-limits
# if you push past ~10k).
python experiments/exp4_tabpfn_baseline.py \
    --dataset pcam \
    --embedding-dir data/embeddings \
    --output-dir data \
    --device cuda \
    --max-train-samples 3500 \
    --predict-chunk-size 1000 \
    --n-estimators 4 \
    --seed 42

echo "Done at $(date)"
