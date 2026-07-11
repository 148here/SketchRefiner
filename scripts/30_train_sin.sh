#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

cd "${PROJECT_ROOT}"

SIN_CONFIG_TEMPLATE="${SIN_CONFIG_TEMPLATE:-${PROJECT_ROOT}/SIN_src/configs/stage3_finetune.yml}"
SIN_CACHE_DIR="${SIN_CACHE_DIR:-${PROJECT_ROOT}/output/cache}"
SIN_STAGE3_FLIST="${SIN_STAGE3_FLIST:-${SIN_CACHE_DIR}/sin_stage3_train_flist.txt}"
SIN_OUTPUT="${SIN_OUTPUT:-${PROJECT_ROOT}/output/train/6_sin_after_srn_stage3}"
SIN_SKETCH_PATH="${SIN_SKETCH_PATH:-${SIN_CACHE_DIR}/unused_sin_sketches}"
SIN_CONFIG_PATH="${SIN_CONFIG_PATH:-${SIN_CACHE_DIR}/stage3_finetune.runtime.yml}"
SIN_MAX_ITERS="${SIN_MAX_ITERS:-100000}"
SIN_SAVE_INTERVAL="${SIN_SAVE_INTERVAL:-5000}"
SIN_SAMPLE_INTERVAL="${SIN_SAMPLE_INTERVAL:-5000}"
SIN_NUM_WORKERS="${SIN_NUM_WORKERS:-0}"
SIN_RESUME_CHECKPOINT="${SIN_RESUME_CHECKPOINT:-}"
SIN_PRETRAINED_CHECKPOINT="${SIN_PRETRAINED_CHECKPOINT:-${PROJECT_ROOT}/hf_download/sketch-refinement-networks/inpainter.pth}"

if [[ ! -f "${SIN_CONFIG_TEMPLATE}" ]]; then
  echo "Missing SIN_CONFIG_TEMPLATE=${SIN_CONFIG_TEMPLATE}" >&2
  exit 1
fi

mkdir -p "${SIN_CACHE_DIR}" "${SIN_OUTPUT}" "${SIN_SKETCH_PATH}"

run_python - "${STAGE3_IMAGE_ROOTS}" train "${SIN_STAGE3_FLIST}" <<'PY'
import os
import sys
from pathlib import Path

roots_arg, split, out_path = sys.argv[1:4]
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def image_files(root):
    return sorted(
        str(path)
        for path in Path(root).rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )

def scan_stage3_root(dataset_root):
    dataset_root = Path(dataset_root)
    split_root = dataset_root / split
    if not split_root.is_dir():
        split_root = dataset_root

    image_dirs = [
        path
        for path, dirs, files in os.walk(split_root)
        if Path(path).name.lower() == "images"
    ]

    files = []
    for image_dir in sorted(image_dirs):
        files.extend(image_files(image_dir))

    if not files:
        files.extend(image_files(split_root))
    return files

paths = []
for root in [item.strip() for item in roots_arg.replace(";", ",").split(",") if item.strip()]:
    if not os.path.isdir(root):
        print(f"Skipping missing dataset root: {root}", file=sys.stderr)
        continue
    paths.extend(scan_stage3_root(root))

paths = sorted(dict.fromkeys(paths))
if not paths:
    raise SystemExit("No training images found for SIN Stage3 flist.")

tmp_path = out_path + ".tmp"
with open(tmp_path, "w", encoding="utf-8", newline="\n") as handle:
    handle.write("\n".join(paths))
    handle.write("\n")
os.replace(tmp_path, out_path)
print(f"Wrote {len(paths)} SIN training images to {out_path}")
PY

run_python - \
  "${SIN_CONFIG_TEMPLATE}" \
  "${SIN_CONFIG_PATH}" \
  "${SIN_STAGE3_FLIST}" \
  "${SIN_SKETCH_PATH}" \
  "${SIN_OUTPUT}" \
  "${SIN_MAX_ITERS}" \
  "${SIN_SAVE_INTERVAL}" \
  "${SIN_SAMPLE_INTERVAL}" \
  "${SIN_NUM_WORKERS}" <<'PY'
import sys
import yaml

(
    template_path,
    config_path,
    train_flist,
    sketch_path,
    output_dir,
    max_iters,
    save_interval,
    sample_interval,
    num_workers,
) = sys.argv[1:10]

with open(template_path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config["TRAIN_FLIST"] = train_flist
config["SKETCH_PATH"] = sketch_path
config["OUTPUT_DIR"] = output_dir
config["MAX_ITERS"] = int(max_iters)
config["SAVE_INTERVAL"] = int(save_interval)
config["SAMPLE_INTERVAL"] = int(sample_interval)
config["EVAL_INTERVAL"] = 0
config["NUM_WORKERS"] = int(num_workers)
config["LATEST_CHECKPOINT_ONLY"] = True
config["GEN_CHECKPOINT_NAME"] = "latest_gen.pth"
config["DIS_CHECKPOINT_NAME"] = "latest_dis.pth"

with open(config_path, "w", encoding="utf-8", newline="\n") as handle:
    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=False)

print(f"Wrote SIN runtime config to {config_path}")
PY

if [[ -n "${SIN_RESUME_CHECKPOINT}" ]]; then
  if [[ ! -f "${SIN_RESUME_CHECKPOINT}" ]]; then
    echo "Missing SIN_RESUME_CHECKPOINT=${SIN_RESUME_CHECKPOINT}" >&2
    exit 1
  fi
elif [[ -f "${SIN_OUTPUT}/checkpoints/latest_gen.pth" ]]; then
  SIN_RESUME_CHECKPOINT="${SIN_OUTPUT}/checkpoints/latest_gen.pth"
elif [[ -f "${SIN_PRETRAINED_CHECKPOINT}" ]]; then
  SIN_RESUME_CHECKPOINT="${SIN_PRETRAINED_CHECKPOINT}"
else
  echo "No SIN checkpoint found; starting SIN from scratch."
fi

SIN_ARGS=(
  SIN_train.py
  --config_path "${SIN_CONFIG_PATH}"
  --GPU_ids "${CUDA_VISIBLE_DEVICES}"
  --nodes 1
  --gpus 1
  --node_rank 0
)

if [[ -n "${SIN_RESUME_CHECKPOINT}" ]]; then
  echo "Using SIN resume checkpoint: ${SIN_RESUME_CHECKPOINT}"
  SIN_ARGS+=(--resume_checkpoint "${SIN_RESUME_CHECKPOINT}")
fi

run_python "${SIN_ARGS[@]}"
