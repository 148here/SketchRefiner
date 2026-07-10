#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

cd "${PROJECT_ROOT}"

SRN_RM_CHECKPOINT="${SRN_RM_CHECKPOINT:-${SRN_STAGE3_OUTPUT}/checkpoints/latest.pth}"
SRN_EM_OUTPUT="${SRN_EM_OUTPUT:-${PROJECT_ROOT}/output/train/5_srn_em_after_stage3_finetune}"

if [[ ! -f "${SRN_RM_CHECKPOINT}" ]]; then
  echo "Missing SRN_RM_CHECKPOINT=${SRN_RM_CHECKPOINT}" >&2
  echo "Run scripts/10_train_srn_stage3_finetune.sh first, or export SRN_RM_CHECKPOINT." >&2
  exit 1
fi

python SRN_train.py \
  --images "${STAGE3_IMAGE_ROOTS}" \
  --image_scan_mode stage3 \
  --stage3_split train \
  --output "${SRN_EM_OUTPUT}" \
  --batch_size 48 \
  --size 256 \
  --max_iters 50000 \
  --epochs 150 \
  --num_workers 8 \
  --lr 1e-5 \
  --use_cosine_lr \
  --cosine_eta_min 1e-6 \
  --val_interval 0 \
  --checkpoint_interval 5000 \
  --latest_checkpoint_only \
  --checkpoint_name latest.pth \
  --sample_retry_limit 128 \
  --cache_clear_interval 1 \
  --train_EM \
  --RM_checkpoint "${SRN_RM_CHECKPOINT}"
