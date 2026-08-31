#!/usr/bin/env bash

set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stage2_dir="$(cd "${experiment_dir}/../.." && pwd)"
workspace_dir="$(cd "${stage2_dir}/.." && pwd)"
variant="${1:-}"
gpu_id="${2:-}"
run_id="${3:-manual}"

if [[ ! "${variant}" =~ ^[a-z0-9_]+$ || ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 {baseline|distance_only|weighted_pretrain_distance_context} GPU_ID [RUN_ID]" >&2
  exit 2
fi
if [[ ! "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID may contain only letters, numbers, dot, underscore and dash." >&2
  exit 2
fi
definition="${experiment_dir}/variants/${variant}/variant.env"
if [[ ! -f "${definition}" ]]; then
  echo "Unknown Stage-2 variant: ${variant}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${definition}"

python_bin="${PYTHON_BIN:-${workspace_dir}/dinov2/.venv/bin/python}"
config_file="${CONFIG_FILE:-${stage2_dir}/dinov2/configs/train/vitl16_short.yaml}"
graph_root="${PRETRAIN_GRAPH_ROOT:-${workspace_dir}/Graph/stage2_variants/${PRETRAIN_GRAPH_SET}}"
context_graph_root="${CONTEXT_GRAPH_ROOT:-${workspace_dir}/Graph/stage2_variants/${CONTEXT_GRAPH_SET}}"
output_dir="${OUTPUT_DIR:-${stage2_dir}/dinov2/results/stage2_variants/${VARIANT_NAME}/${run_id}}"
num_workers="${NUM_WORKERS:-4}"
max_nodes="${MAX_NODES:-5000}"
run_seed="${RUN_SEED:-0}"

for required in "${python_bin}" "${config_file}" "${graph_root}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required Stage-2 resource: ${required}" >&2
    exit 1
  fi
done
if ! find "${graph_root}" -maxdepth 1 -type f -name '*.pt' -print -quit | grep -q .; then
  echo "No graph files found in ${graph_root}; run prepare_graphs.sh first." >&2
  exit 1
fi
graph_manifest="${graph_root}/stage2_graph_manifest.json"
if [[ ! -f "${graph_manifest}" ]]; then
  echo "Missing graph manifest: ${graph_manifest}" >&2
  exit 1
fi
read -r manifest_edge_mode manifest_edge_dim < <(
  "${python_bin}" -c \
    'import json,sys; value=json.load(open(sys.argv[1])); print(value["edge_mode"], value["edge_dim"])' \
    "${graph_manifest}"
)
if [[ "${manifest_edge_mode}" != "${PRETRAIN_EDGE_MODE}" ]]; then
  echo "Graph edge mode ${manifest_edge_mode} does not match variant mode ${PRETRAIN_EDGE_MODE}." >&2
  exit 1
fi
if [[ "${manifest_edge_dim}" != "${PRETRAIN_EDGE_DIM}" ]]; then
  echo "Graph edge_dim ${manifest_edge_dim} does not match variant edge_dim ${PRETRAIN_EDGE_DIM}." >&2
  exit 1
fi
if ! "${python_bin}" -c "import torch_geometric, torch_scatter" >/dev/null 2>&1; then
  echo "The selected PYTHON_BIN must provide torch_geometric and torch_scatter." >&2
  exit 3
fi

if command -v nvidia-smi >/dev/null 2>&1 && [[ "${ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  used_memory="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  max_used_memory="${GPU_MAX_USED_MB_BEFORE_START:-1024}"
  if [[ "${used_memory}" =~ ^[0-9]+$ ]] && (( used_memory > max_used_memory )); then
    echo "GPU ${gpu_id} already uses ${used_memory} MiB; refusing to start above ${max_used_memory} MiB." >&2
    exit 4
  fi
fi
if [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite an existing Stage-2 run: ${output_dir}" >&2
  exit 5
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Ready: variant=${VARIANT_NAME} gpu=${gpu_id} graph=${graph_root} output=${output_dir}"
  exit 0
fi

mkdir -p "${output_dir}"
source_hashes="$(
  cd "${stage2_dir}"
  sha256sum \
    build_graphs.py \
    dinov2/configs/ssl_default_config.yaml \
    dinov2/data/datasets/graph_builder.py \
    dinov2/models/gcn.py \
    dinov2/train/gcn_meta_arch.py
)"
{
  printf 'variant=%s\n' "${VARIANT_NAME}"
  printf 'run_id=%s\n' "${run_id}"
  printf 'gpu_id=%s\n' "${gpu_id}"
  printf 'graph_root=%s\n' "${graph_root}"
  printf 'graph_manifest=%s\n' "${graph_manifest}"
  printf 'pretrain_edge_mode=%s\n' "${PRETRAIN_EDGE_MODE}"
  printf 'pretrain_edge_dim=%s\n' "${PRETRAIN_EDGE_DIM}"
  printf 'pretrain_edge_injection=%s\n' "${PRETRAIN_EDGE_INJECTION}"
  printf 'edge_weight_objective=%s\n' "${EDGE_WEIGHT_OBJECTIVE}"
  printf 'edge_weight_weight=%s\n' "${EDGE_WEIGHT_WEIGHT}"
  printf 'edge_weight_noise_std=%s\n' "${EDGE_WEIGHT_NOISE_STD}"
  printf 'context_graph_root=%s\n' "${context_graph_root}"
  printf 'context_edge_mode=%s\n' "${CONTEXT_EDGE_MODE}"
  printf 'context_use_edge_attr=%s\n' "${CONTEXT_USE_EDGE_ATTR}"
  printf 'seed=%s\n' "${run_seed}"
  printf 'max_nodes=%s\n' "${max_nodes}"
  printf 'source_sha256:\n%s\n' "${source_hashes}"
} > "${output_dir}/variant_manifest.txt"

cd "${stage2_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
exec "${python_bin}" -m dinov2.train.train \
  --no-resume \
  --config-file "${config_file}" \
  --output-dir "${output_dir}" \
  "gcn.edge_dim=${PRETRAIN_EDGE_DIM}" \
  "gcn.edge_injection=${PRETRAIN_EDGE_INJECTION}" \
  "gcn.edge_weight_objective=${EDGE_WEIGHT_OBJECTIVE}" \
  "gcn.edge_weight_weight=${EDGE_WEIGHT_WEIGHT}" \
  "gcn.edge_weight_noise_std=${EDGE_WEIGHT_NOISE_STD}" \
  "gcn.context_edge_mode=${CONTEXT_EDGE_MODE}" \
  "gcn.context_use_edge_attr=${CONTEXT_USE_EDGE_ATTR}" \
  "train.seed=${run_seed}" \
  "train.num_workers=${num_workers}" \
  "train.dataset_path=ImageFolder:root=${graph_root}:max_nodes=${max_nodes}:edge_dim=${PRETRAIN_EDGE_DIM}"
