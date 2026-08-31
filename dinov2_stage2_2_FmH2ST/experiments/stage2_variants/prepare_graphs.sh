#!/usr/bin/env bash

set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stage2_dir="$(cd "${experiment_dir}/../.." && pwd)"
workspace_dir="$(cd "${stage2_dir}/.." && pwd)"
variant="${1:-}"
purpose="${2:-all}"

if [[ ! "${variant}" =~ ^[a-z0-9_]+$ ]]; then
  echo "Usage: $0 {baseline|distance_only|weighted_pretrain_distance_context} [all|pretrain|context]" >&2
  exit 2
fi
if [[ ! "${purpose}" =~ ^(all|pretrain|context)$ ]]; then
  echo "Graph purpose must be one of: all, pretrain, context" >&2
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
embeddings_dir="${EMBEDDINGS_DIR:-${workspace_dir}/Graph/embeddings-1024}"
metadata_dir="${METADATA_DIR:-${embeddings_dir}}"
require_metadata="${REQUIRE_DECODER_METADATA:-1}"
graph_k="${GRAPH_K:-8}"
distance_multiplier="${DISTANCE_MULTIPLIER:-3.0}"

for required in "${python_bin}" "${embeddings_dir}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing Stage-2 graph resource: ${required}" >&2
    exit 1
  fi
done
if ! "${python_bin}" -c "import torch_geometric" >/dev/null 2>&1; then
  echo "The selected PYTHON_BIN does not provide torch_geometric." >&2
  exit 3
fi

metadata_args=()
if find "${metadata_dir}" -maxdepth 1 -type f -name '*_metadata.pt' -print -quit | grep -q .; then
  metadata_args+=(--metadata-dir "${metadata_dir}")
elif [[ "${require_metadata}" == "1" ]]; then
  echo "No *_metadata.pt files found in ${metadata_dir}" >&2
  echo "Run Stage1 organization first, or set REQUIRE_DECODER_METADATA=0 for a temporary graph-only test." >&2
  exit 4
fi
if [[ "${require_metadata}" == "1" ]]; then
  metadata_args+=(--require-decoder-metadata)
fi

overwrite_args=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi
distance_args=(--distance-multiplier "${distance_multiplier}")
if [[ -n "${MAX_DISTANCE:-}" ]]; then
  distance_args+=(--max-distance "${MAX_DISTANCE}")
fi

prepare_graph_set() {
  local graph_role="$1"
  local graph_set="$2"
  local edge_mode="$3"
  local graph_root
  if [[ "${graph_role}" == "pretrain" ]]; then
    graph_root="${PRETRAIN_GRAPH_ROOT:-${workspace_dir}/Graph/stage2_variants/${graph_set}}"
  else
    graph_root="${CONTEXT_GRAPH_ROOT:-${workspace_dir}/Graph/stage2_variants/${graph_set}}"
  fi
  mkdir -p "${graph_root}"
  (
    cd "${stage2_dir}"
    "${python_bin}" build_graphs.py \
      --embeddings-dir "${embeddings_dir}" \
      --output-dir "${graph_root}" \
      --edge-mode "${edge_mode}" \
      --k "${graph_k}" \
      "${distance_args[@]}" \
      "${metadata_args[@]}" \
      "${overwrite_args[@]}"
  )
}

if [[ "${purpose}" == "all" || "${purpose}" == "pretrain" ]]; then
  prepare_graph_set pretrain "${PRETRAIN_GRAPH_SET}" "${PRETRAIN_EDGE_MODE}"
fi
if [[ "${purpose}" == "all" || "${purpose}" == "context" ]]; then
  if [[
    "${purpose}" != "all" ||
    "${CONTEXT_GRAPH_SET}" != "${PRETRAIN_GRAPH_SET}" ||
    "${CONTEXT_EDGE_MODE}" != "${PRETRAIN_EDGE_MODE}"
  ]]; then
    prepare_graph_set context "${CONTEXT_GRAPH_SET}" "${CONTEXT_EDGE_MODE}"
  fi
fi
