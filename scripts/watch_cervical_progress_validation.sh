#!/usr/bin/env bash

# Independently evaluate immutable snapshots of DDP progress checkpoints.
# This process never changes trainer history, early stopping, or checkpoint_best.pt.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_root="${CERVICAL_WORK_ROOT:-${repo_dir}/Data/cervical_segmentation_latest_area_20260904}"
watch_root="${PARALLEL_OUTPUT_ROOT:-${work_root}/decoder_runs_parallel}"
stage2_source="${repo_dir}/dinov2_stage2_2_FmH2ST"
python_bin="${PYTHON_BIN:-${repo_dir}/dinov2/.venv/bin/python}"
gpu_id="${ASYNC_VALIDATION_GPU_ID:-0}"
subset_size="${ASYNC_VALIDATION_SUBSET_SIZE:-50000}"
workers="${ASYNC_VALIDATION_WORKERS:-4}"
poll_seconds="${ASYNC_VALIDATION_POLL_SECONDS:-30}"
retry_seconds="${ASYNC_VALIDATION_RETRY_SECONDS:-300}"
min_free_gib="${ASYNC_VALIDATION_MIN_FREE_GIB:-24}"

[[ -x "${python_bin}" ]] || { echo "Python executable not found: ${python_bin}" >&2; exit 1; }
[[ "${gpu_id}" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "GPU id must be non-negative" >&2; exit 2; }
for value in "${subset_size}" "${poll_seconds}" "${retry_seconds}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Positive integer required: ${value}" >&2; exit 2; }
done
[[ "${workers}" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "workers must be non-negative" >&2; exit 2; }

mkdir -p "${watch_root}"
cd "${stage2_source}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${stage2_source}:${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

exec nice -n 10 "${python_bin}" -m dinov2_segmentation.validate_progress_checkpoints \
  --watch-root "${watch_root}" \
  --device cuda \
  --workers "${workers}" \
  --subset-size "${subset_size}" \
  --poll-seconds "${poll_seconds}" \
  --retry-seconds "${retry_seconds}" \
  --min-free-memory-gib "${min_free_gib}"
