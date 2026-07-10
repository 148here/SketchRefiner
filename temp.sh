#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/scripts/10_train_srn_stage3_finetune.sh"
bash "${SCRIPT_DIR}/scripts/20_train_srn_em.sh"
bash "${SCRIPT_DIR}/scripts/30_train_sin.sh"
