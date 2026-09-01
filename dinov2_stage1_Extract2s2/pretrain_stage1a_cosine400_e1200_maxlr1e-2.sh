#!/usr/bin/env bash
set -euo pipefail

stage1_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "${stage1_dir}/.." && pwd)"

python_bin="${PYTHON_BIN:-${workspace_dir}/dinov2/.venv/bin/python}"
data_root="${DATA_ROOT:-${workspace_dir}/Data/Pretrain}"
data_extra="${DATA_EXTRA:-${workspace_dir}/Data/Pretrain_extra}"
dino_weights="${DINO_WEIGHTS:-${workspace_dir}/dinov2/dinov2/results/dinov2_pretrain/eval/training_368302/teacher_checkpoint.pth}"
output_dir="${OUTPUT_DIR:-${stage1_dir}/dinov2/results/stage1a_spatial_fusion_cosine400_e1200_maxlr1e-2_minlr1e-8}"
gpu_id="${GPU_ID:-1}"

for required in "${python_bin}" "${data_root}" "${data_extra}" "${dino_weights}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required Stage-1A resource: ${required}" >&2
    exit 1
  fi
done
if [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite an existing Stage-1A run: ${output_dir}" >&2
  exit 2
fi

cd "${stage1_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

# sqrt_wrt_1024 scales 0.04 by sqrt(64 / 1024) = 0.25, producing the
# requested effective scheduler maximum of 0.01 on one GPU.
exec "${python_bin}" -m dinov2.train.train \
  --no-resume \
  --config-file dinov2/configs/train/vitl16_short_imgnet22k.yaml \
  --output-dir "${output_dir}" \
  MODEL.WEIGHTS="${dino_weights}" \
  optim.epochs=1200 \
  optim.base_lr=0.04 \
  optim.min_lr=1.0e-08 \
  optim.cosine_cycle_epochs=400 \
  train.saveckp_freq=400 \
  train.dataset_path="ImageNet22k:root=${data_root}:extra=${data_extra}"
