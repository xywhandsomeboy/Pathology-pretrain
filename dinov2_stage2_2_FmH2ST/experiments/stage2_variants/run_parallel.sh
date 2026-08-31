#!/usr/bin/env bash

set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stage2_dir="$(cd "${experiment_dir}/../.." && pwd)"
run_id="${1:-}"
variant_gpu0="${2:-baseline}"
variant_gpu1="${3:-spatial_only}"
gpu0_id="${GPU0_ID:-0}"
gpu1_id="${GPU1_ID:-1}"

if [[ -z "${run_id}" ]]; then
  echo "Usage: $0 RUN_ID [VARIANT_GPU0] [VARIANT_GPU1]" >&2
  exit 2
fi
if [[ "${variant_gpu0}" == "${variant_gpu1}" ]]; then
  echo "Parallel comparison requires two different variants." >&2
  exit 2
fi
if [[ "${gpu0_id}" == "${gpu1_id}" ]]; then
  echo "GPU0_ID and GPU1_ID must be different." >&2
  exit 2
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  echo "Do not set one shared OUTPUT_DIR for a parallel comparison." >&2
  exit 2
fi

# Validate both variants before launching either one. This avoids leaving a
# single unmatched run active when the other variant has a missing graph or a
# busy GPU.
DRY_RUN=1 "${experiment_dir}/run_variant.sh" \
  "${variant_gpu0}" "${gpu0_id}" "${run_id}"
DRY_RUN=1 "${experiment_dir}/run_variant.sh" \
  "${variant_gpu1}" "${gpu1_id}" "${run_id}"

log_dir="${stage2_dir}/dinov2/results/stage2_variants/launcher_logs/${run_id}"
mkdir -p "${log_dir}"

"${experiment_dir}/run_variant.sh" "${variant_gpu0}" "${gpu0_id}" "${run_id}" \
  > "${log_dir}/${variant_gpu0}.log" 2>&1 &
pid0=$!
"${experiment_dir}/run_variant.sh" "${variant_gpu1}" "${gpu1_id}" "${run_id}" \
  > "${log_dir}/${variant_gpu1}.log" 2>&1 &
pid1=$!

echo "Started ${variant_gpu0} on GPU ${gpu0_id}: PID ${pid0}"
echo "Started ${variant_gpu1} on GPU ${gpu1_id}: PID ${pid1}"
echo "Launcher logs: ${log_dir}"

status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
exit "${status}"
