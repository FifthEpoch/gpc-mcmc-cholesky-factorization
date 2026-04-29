#!/bin/bash
#SBATCH --job-name=exp2_bw_sweep_std
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/hmc_rose/%x_%j.out
#SBATCH --error=slurm_logs/hmc_rose/%x_%j.err

set -euo pipefail

# Submit with:
#   sbatch --export=ALL,NETID=<your-netid> scripts/exp2_rose_exp1_metrics_pcam_3k_500_bandwidth_sweep_standardized.sbatch
#
# Optional overrides:
#   BANDWIDTH_VALUES="5 10 15 20 30 40 50"
#   K_VALUE, N_SAMPLES, N_WARMUP, N_LEAPFROG
#   TRAIN_SUBSAMPLE, TEST_SUBSAMPLE, SEED

BANDWIDTH_VALUES="${BANDWIDTH_VALUES:-5 10 15 20 30 40 50}"
K_VALUE="${K_VALUE:-200}"
N_SAMPLES="${N_SAMPLES:-2000}"
N_WARMUP="${N_WARMUP:-200}"
N_LEAPFROG="${N_LEAPFROG:-12}"
TRAIN_SUBSAMPLE="${TRAIN_SUBSAMPLE:-3000}"
TEST_SUBSAMPLE="${TEST_SUBSAMPLE:-500}"
SEED="${SEED:-42}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/exp2_rose_exp1_metrics_pcam_3k_500.sbatch"

if [ ! -f "${RUN_SCRIPT}" ]; then
  echo "ERROR: Missing run script: ${RUN_SCRIPT}" >&2
  exit 1
fi

mkdir -p slurm_logs/hmc_rose

echo "=============================================================================="
echo "Submitting standardized bandwidth sweep for exp2_rose_exp1_metrics"
echo "Bandwidths      : ${BANDWIDTH_VALUES}"
echo "Train/test      : ${TRAIN_SUBSAMPLE}/${TEST_SUBSAMPLE}"
echo "k               : ${K_VALUE}"
echo "samples/warmup  : ${N_SAMPLES}/${N_WARMUP}"
echo "leapfrog        : ${N_LEAPFROG}"
echo "seed            : ${SEED}"
echo "=============================================================================="

for bw in ${BANDWIDTH_VALUES}; do
  bw_tag="$(printf '%s' "${bw}" | tr '.' 'p')"
  factor_dir="datasets/pcam-hg/train/embeddings/exp2_rpchol_${TRAIN_SUBSAMPLE}_stratified_standardized_bw${bw_tag}"
  output_dir="data/exp2_rose_exp1_metrics_pcam_${TRAIN_SUBSAMPLE}_${TEST_SUBSAMPLE}_standardized_bw${bw_tag}"

  echo
  echo "Submitting bandwidth=${bw}"
  echo "  factor_dir=${factor_dir}"
  echo "  output_dir=${output_dir}"

  sbatch --export=ALL,\
NETID="${NETID:-}",\
BANDWIDTH="${bw}",\
K_VALUE="${K_VALUE}",\
N_SAMPLES="${N_SAMPLES}",\
N_WARMUP="${N_WARMUP}",\
N_LEAPFROG="${N_LEAPFROG}",\
TRAIN_SUBSAMPLE="${TRAIN_SUBSAMPLE}",\
TEST_SUBSAMPLE="${TEST_SUBSAMPLE}",\
STANDARDIZE_FACTOR=1,\
STANDARDIZE_FROM_FACTOR=1,\
SEED="${SEED}",\
FACTOR_DIR="${factor_dir}",\
OUTPUT_DIR="${output_dir}" \
    "${RUN_SCRIPT}"
done

echo
echo "All bandwidth jobs submitted."
