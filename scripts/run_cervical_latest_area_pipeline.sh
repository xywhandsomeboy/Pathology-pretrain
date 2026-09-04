#!/usr/bin/env bash

# Freeze the latest complete annotated cohort, prepare Stage1/graph inputs, and
# train S/ST/STA without the withdrawn tumor-area loss in isolated outputs.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_root="${CERVICAL_WORK_ROOT:-${repo_dir}/Data/cervical_segmentation_latest_area_20260904}"
python_bin="${PYTHON_BIN:-${repo_dir}/dinov2/.venv/bin/python}"
gpu_id="${TRAIN_GPU_ID:-1}"
stage2_variant="${STAGE2_VARIANT:-weighted_pretrain_distance_context}"
run_suffix="${RUN_SUFFIX:-_dp020_noarea}"
logs_dir="${work_root}/logs"

mkdir -p "${logs_dir}"
exec 9>"${work_root}/latest_pipeline.lock"
if ! flock -n 9; then
  echo "Another latest-cohort pipeline is already active for ${work_root}" >&2
  exit 1
fi

env \
  CERVICAL_WORK_ROOT="${work_root}" \
  PYTHON_BIN="${python_bin}" \
  PREPROCESS_ONLY=1 \
  REFRESH_COHORT="${REFRESH_COHORT:-1}" \
  PREPARE_WORKERS="${PREPARE_WORKERS:-6}" \
  STAGE1_GPU_ID="${STAGE1_GPU_ID:-${gpu_id}}" \
  DATA_SPLIT_MODE="${DATA_SPLIT_MODE:-train_val_80_20}" \
  VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.2}" \
  JOINT_VARIANTS="${stage2_variant}" \
  bash "${repo_dir}/scripts/run_cervical_six_models.sh"

pids=()
for profile in S ST STA; do
  log_file="${logs_dir}/decoder_noarea_${profile}_${stage2_variant}_v1.log"
  env \
    CERVICAL_WORK_ROOT="${work_root}" \
    PYTHON_BIN="${python_bin}" \
    DECODER_DROP_PATH_RATE=0.2 \
    RUN_SUFFIX="${run_suffix}" \
    BATCH_SIZE="${BATCH_SIZE:-16}" \
    DECODER_WORKERS="${DECODER_WORKERS:-8}" \
    "${repo_dir}/scripts/run_cervical_improved_variant.sh" \
      "${profile}" "${stage2_variant}" v1 "${gpu_id}" \
      >"${log_file}" 2>&1 &
  pids+=("$!")
done

"${python_bin}" "${repo_dir}/scripts/monitor_cervical_overfitting.py" \
  --work-root "${work_root}" --poll-seconds 30 --stop-on-detection \
  >"${logs_dir}/overfitting_monitor.log" 2>&1 &
monitor_pid=$!
trap 'kill -TERM "${monitor_pid}" 2>/dev/null || true' EXIT

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
exit "${status}"
