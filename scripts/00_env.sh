#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PROJECT_ROOT="${PROJECT_ROOT:-${LOCAL_PROJECT_ROOT}}"
export SKETCHINPAINTER_ROOT="${SKETCHINPAINTER_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)/SketchInpainter}"
export DATA_ROOT="${DATA_ROOT:-/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-sketchinpainter-train}"
export USE_CONDA_RUN="${USE_CONDA_RUN:-1}"
export SRN_NUM_WORKERS="${SRN_NUM_WORKERS:-0}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

export ARTBENCH_ROOT="${ARTBENCH_ROOT:-${DATA_ROOT}/artbench/export_512}"
export COCO_ROOT="${COCO_ROOT:-${DATA_ROOT}/coco}"
export MURAL1_ROOT="${MURAL1_ROOT:-${DATA_ROOT}/mural1}"
export STAGE3_IMAGE_ROOTS="${STAGE3_IMAGE_ROOTS:-${ARTBENCH_ROOT},${COCO_ROOT},${MURAL1_ROOT}}"

export SRN_PRETRAINED_RM="${SRN_PRETRAINED_RM:-${PROJECT_ROOT}/hf_download/sketch-refinement-networks/registrator.pth}"
export SRN_STAGE3_OUTPUT="${SRN_STAGE3_OUTPUT:-${PROJECT_ROOT}/output/train/4_stage3data_sketchinpainter_backend}"
export SRN_STAGE3_SMOKE_OUTPUT="${SRN_STAGE3_SMOKE_OUTPUT:-${PROJECT_ROOT}/output/train/4_stage3data_sketchinpainter_backend_smoke}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "SKETCHINPAINTER_ROOT=${SKETCHINPAINTER_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "CONDA_ENV_NAME=${CONDA_ENV_NAME}"
echo "SRN_NUM_WORKERS=${SRN_NUM_WORKERS}"
echo "MKL_THREADING_LAYER=${MKL_THREADING_LAYER}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

run_python() {
  if [[ "${USE_CONDA_RUN}" == "1" ]]; then
    conda run --no-capture-output -n "${CONDA_ENV_NAME}" python "$@"
  else
    python "$@"
  fi
}
