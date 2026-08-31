#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=Job.%j.out
#SBATCH --error=Job.%j.err

set -euo pipefail

stage2_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "${stage2_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-${workspace_dir}/dinov2/.venv/bin/python}"
graph_root="${GRAPH_ROOT:-${workspace_dir}/Graph/graphs-current}"
output_dir="${OUTPUT_DIR:-${stage2_dir}/dinov2/results/stage2_graph_pretrain}"
gpu_id="${GPU_ID:-0}"

for required in "${python_bin}" "${graph_root}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required Stage-2 resource: ${required}" >&2
    exit 1
  fi
done
if ! "${python_bin}" -c "import torch_geometric" >/dev/null 2>&1; then
  echo "The selected PYTHON_BIN does not provide torch_geometric." >&2
  exit 2
fi

cd "${stage2_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
exec "${python_bin}" -m dinov2.train.train \
  --no-resume \
  --config-file dinov2/configs/train/vitl16_short.yaml \
  --output-dir "${output_dir}" \
  train.dataset_path="ImageFolder:root=${graph_root}:edge_dim=2"
