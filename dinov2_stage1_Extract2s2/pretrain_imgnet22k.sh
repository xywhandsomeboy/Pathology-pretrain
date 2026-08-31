#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=Job.%j.out
#SBATCH --error=Job.%j.err

# Stage-1B deterministic feature export. STAGE1_WEIGHTS must be a merged,
# trained Stage-1A checkpoint, not the original DINO-only checkpoint.
set -euo pipefail

stage1_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "${stage1_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-${workspace_dir}/dinov2/.venv/bin/python}"
data_root="${DATA_ROOT:-${workspace_dir}/Data/Pretrain}"
data_extra="${DATA_EXTRA:-${workspace_dir}/Data/Pretrain_extra}"
stage1_weights="${STAGE1_WEIGHTS:-}"
output_dir="${OUTPUT_DIR:-${stage1_dir}/dinov2/results/stage1b_features}"
gpu_id="${GPU_ID:-0}"
export_dense_tokens="${EXPORT_DENSE_TOKENS:-true}"

if [[ -z "${stage1_weights}" ]]; then
  echo "STAGE1_WEIGHTS must point to a trained, merged Stage-1A checkpoint." >&2
  exit 2
fi
for required in "${python_bin}" "${data_root}" "${data_extra}" "${stage1_weights}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required Stage-1B resource: ${required}" >&2
    exit 1
  fi
done

cd "${stage1_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
exec "${python_bin}" -m dinov2.train.train \
  --eval-only \
  --no-resume \
  --config-file dinov2/configs/train/vitl16_short_imgnet22k.yaml \
  --output-dir "${output_dir}" \
  MODEL.WEIGHTS="${stage1_weights}" \
  feature.export_dense_tokens="${export_dense_tokens}" \
  feature.require_trained_spatial=true \
  train.dataset_path="ImageNet22k:root=${data_root}:extra=${data_extra}"
