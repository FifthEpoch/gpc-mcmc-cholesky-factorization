#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATASET_ROOT="${DATASET_ROOT:-datasets/pcam-hg}"
SPLITS="${SPLITS:-train valid test}"
SIZES="${SIZES:-5k 10k 20k 50k full}"
K_VALUE="${K_VALUE:-200}"

move_if_present() {
  local src="$1"
  local dst="$2"

  if [ -f "${src}" ]; then
    mv -f "${src}" "${dst}"
    echo "  moved $(basename "${src}") -> $(basename "${dst}")"
  else
    echo "  missing $(basename "${src}"), skipping"
  fi
}

for split in ${SPLITS}; do
  split_dir="${PROJECT_ROOT}/${DATASET_ROOT}/${split}/embeddings"

  if [ ! -d "${split_dir}" ]; then
    echo "Skipping split=${split}: missing directory ${split_dir}"
    continue
  fi

  echo
  echo "Split ${split}: ${split_dir}"

  for tag in ${SIZES}; do
    target_dir="${split_dir}/exp2_rpchol_${tag}"
    mkdir -p "${target_dir}"

    echo "Organizing ${tag} -> ${target_dir}"

    move_if_present "${split_dir}/factor_${tag}_k${K_VALUE}.npy" "${target_dir}/factor_k${K_VALUE}.npy"
    move_if_present "${split_dir}/pivots_${tag}_k${K_VALUE}.npy" "${target_dir}/pivots_k${K_VALUE}.npy"
    move_if_present "${split_dir}/kernel_submatrix_${tag}_k${K_VALUE}.npy" "${target_dir}/kernel_submatrix_k${K_VALUE}.npy"
    move_if_present "${split_dir}/embeddings_${tag}_k${K_VALUE}.npy" "${target_dir}/embeddings.npy"
    move_if_present "${split_dir}/labels_${tag}_k${K_VALUE}.npy" "${target_dir}/labels.npy"
    move_if_present "${split_dir}/summary_${tag}_k${K_VALUE}.json" "${target_dir}/summary.json"
  done
done

echo
echo "Finished reorganizing RPCholesky outputs under ${PROJECT_ROOT}/${DATASET_ROOT}."
