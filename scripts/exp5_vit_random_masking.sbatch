#!/bin/bash
#SBATCH --job-name=exp5_vit_mask
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/exp5_vit_small/%x_%j.out
#SBATCH --error=slurm_logs/exp5_vit_small/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

NETID="${NETID:-<netid>}"
if [ "${NETID}" = "<netid>" ]; then
  echo "ERROR: NETID is required. Submit with --export=ALL,NETID=<your-netid>." >&2
  exit 1
fi

SCRATCH_BASE="/scratch/${NETID}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/gpc-mcmc-cholesky-factorization}"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"
DATASET="${DATASET:-all}"   # one of: pcam, camelyon17, all
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MASK_RATIO="${MASK_RATIO:-0.0}"   # 0 disables masking; set >0 to enable
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-ML-Final_project}"
WANDB_ENTITY="${WANDB_ENTITY:-a-salt}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-exp5-vit-mask}"

SCRIPT_DIR="${PROJECT_ROOT}/scripts"
if [ -f "${SCRIPT_DIR}/set_scratch_env.sh" ]; then
  source "${SCRIPT_DIR}/set_scratch_env.sh"
fi

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

if [ ! -d "${CONDA_ENV}" ]; then
  echo "ERROR: Conda env not found at ${CONDA_ENV}" >&2
  exit 1
fi

conda activate "${CONDA_ENV}"
unset PYTHONHOME
unset PYTHONPATH

PYTHON="${CONDA_ENV}/bin/python"
cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
mkdir -p data/exp5_vit

echo "=============================================================================="
echo "Experiment 5: ViT-Small + random patch masking"
echo "=============================================================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Node           : $(hostname)"
echo "Dataset        : ${DATASET}"
echo "Epochs         : ${EPOCHS}"
echo "Batch size     : ${BATCH_SIZE}"
echo "Mask ratio     : ${MASK_RATIO}"
echo "Learning rate  : ${LEARNING_RATE}"
echo "Weight decay   : ${WEIGHT_DECAY}"
echo "Use W&B        : ${USE_WANDB}"
echo "Project root   : ${PROJECT_ROOT}"
echo "Conda env      : ${CONDA_ENV}"
echo "=============================================================================="

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

WANDB_ARGS=()
if [ "${USE_WANDB}" = "1" ]; then
  WANDB_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT}" --wandb-mode "${WANDB_MODE}")
  if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
  fi
  if [ -n "${WANDB_GROUP}" ]; then
    WANDB_ARGS+=(--wandb-group "${WANDB_GROUP}")
  fi
fi

"${PYTHON}" experiments/exp5/exp5_vit_small_random_masking.py \
  --dataset "${DATASET}" \
  --data-root "${PROJECT_ROOT}/datasets" \
  --output-root "${PROJECT_ROOT}/data/exp5_vit" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --mask-ratio "${MASK_RATIO}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  "${WANDB_ARGS[@]}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Outputs saved under ${PROJECT_ROOT}/data/exp5_vit"
echo "=============================================================================="
