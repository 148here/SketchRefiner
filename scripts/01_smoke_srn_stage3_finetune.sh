#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

cd "${PROJECT_ROOT}"

python SRN_train.py \
  --images "${STAGE3_IMAGE_ROOTS}" \
  --image_scan_mode stage3 \
  --stage3_split train \
  --output "${SRN_STAGE3_SMOKE_OUTPUT}" \
  --batch_size 1 \
  --size 256 \
  --max_iters 0 \
  --epochs 1 \
  --num_workers 0 \
  --lr 1e-5 \
  --use_cosine_lr \
  --cosine_eta_min 1e-6 \
  --val_interval 0 \
  --sample_interval 1 \
  --checkpoint_interval 1 \
  --latest_checkpoint_only \
  --checkpoint_name latest.pth \
  --resume_checkpoint "${SRN_PRETRAINED_RM}" \
  --sample_retry_limit 128 \
  --cache_clear_interval 0
