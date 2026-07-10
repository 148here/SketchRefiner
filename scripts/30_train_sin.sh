#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

cd "${PROJECT_ROOT}"

SIN_CONFIG_PATH="${SIN_CONFIG_PATH:-${PROJECT_ROOT}/SIN_src/configs/example.yml}"
SIN_RESUME_CHECKPOINT="${SIN_RESUME_CHECKPOINT:-}"

if [[ ! -f "${SIN_CONFIG_PATH}" ]]; then
  echo "Missing SIN_CONFIG_PATH=${SIN_CONFIG_PATH}" >&2
  exit 1
fi

if [[ -n "${SIN_RESUME_CHECKPOINT}" ]]; then
  run_python SIN_train.py \
    --config_path "${SIN_CONFIG_PATH}" \
    --GPU_ids "${CUDA_VISIBLE_DEVICES}" \
    --nodes 1 \
    --gpus 1 \
    --node_rank 0 \
    --DDP \
    --resume_checkpoint "${SIN_RESUME_CHECKPOINT}"
else
  run_python SIN_train.py \
    --config_path "${SIN_CONFIG_PATH}" \
    --GPU_ids "${CUDA_VISIBLE_DEVICES}" \
    --nodes 1 \
    --gpus 1 \
    --node_rank 0 \
    --DDP
fi
