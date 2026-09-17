#!/usr/bin/env bash

# Wait for the legacy serial S/ST runs to produce a resumable checkpoint,
# stop only those exact runs, then continue S/ST and restart the already-lost
# STA run through the two-GPU DDP entry point.  One experiment uses both GPUs at
# a time; the three profiles are queued to avoid oversubscribing host memory.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_root="${CERVICAL_WORK_ROOT:-${repo_dir}/Data/cervical_segmentation_latest_area_20260904}"
stage2_variant="${STAGE2_VARIANT:-weighted_pretrain_distance_context}"
gpu_list="${GPU_LIST:-0,1}"
per_gpu_batch="${BATCH_SIZE:-8}"
workers="${DECODER_WORKERS:-4}"
poll_seconds="${POLL_SECONDS:-60}"
require_target_gpus_safe="${REQUIRE_TARGET_GPUS_SAFE:-1}"
run_suffix="${RUN_SUFFIX:-_from_serial_epoch_boundary_20260907}"
launcher="${repo_dir}/scripts/run_cervical_parallel_variant.sh"
state_dir="${work_root}/ddp_handoff"
log_dir="${work_root}/logs/ddp_handoff"
migration_sources=(
  "${repo_dir}/dinov2_segmentation/distributed_execution.py"
  "${repo_dir}/dinov2_segmentation/joint_optim.py"
  "${repo_dir}/dinov2_segmentation/probability_metrics.py"
  "${repo_dir}/dinov2_segmentation/train_joint.py"
  "${repo_dir}/dinov2_segmentation/train_joint_parallel.py"
  "${launcher}"
)

[[ "${gpu_list}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))+$ ]] || {
  echo "GPU_LIST must contain at least two comma-separated GPU indices for DDP" >&2
  exit 2
}
IFS=',' read -r -a requested_gpu_ids <<< "${gpu_list}"
declare -A seen_gpu_ids=()
for gpu_id in "${requested_gpu_ids[@]}"; do
  [[ -z "${seen_gpu_ids[${gpu_id}]:-}" ]] || {
    echo "GPU_LIST contains duplicate GPU index ${gpu_id}" >&2
    exit 2
  }
  seen_gpu_ids["${gpu_id}"]=1
done
for value_name in per_gpu_batch workers poll_seconds; do
  [[ "${!value_name}" =~ ^[1-9][0-9]*$ ]] || {
    echo "${value_name} must be a positive integer" >&2
    exit 2
  }
done
[[ "${run_suffix}" =~ ^_[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || {
  echo "RUN_SUFFIX has an invalid format: ${run_suffix}" >&2
  exit 2
}
[[ "${require_target_gpus_safe}" == 0 || "${require_target_gpus_safe}" == 1 ]] || {
  echo "REQUIRE_TARGET_GPUS_SAFE must be 0 or 1" >&2
  exit 2
}

mkdir -p "${state_dir}" "${log_dir}"
nvidia-smi -L >/dev/null || {
  echo "nvidia-smi cannot access the host GPUs" >&2
  exit 1
}
for gpu_id in "${requested_gpu_ids[@]}"; do
  nvidia-smi -i "${gpu_id}" --query-gpu=index --format=csv,noheader >/dev/null || {
    echo "GPU ${gpu_id} is not available" >&2
    exit 1
  }
done
exec 9>"${state_dir}/handoff.lock"
if ! flock -n 9; then
  echo "Another serial-to-DDP handoff is already active" >&2
  exit 1
fi
exec > >(tee -a "${log_dir}/handoff.log") 2>&1

source_digest() {
  sha256sum "${migration_sources[@]}" | sha256sum | cut -d' ' -f1
}

initial_source_digest="$(source_digest)"
printf '%s\n' "${initial_source_digest}" >"${state_dir}/source.sha256"

legacy_output() {
  local profile="$1"
  printf '%s/decoder_runs_improved/%s/%s/v1_dp020_noarea' \
    "${work_root}" "${profile}" "${stage2_variant}"
}

legacy_checkpoint() {
  local output progress last
  output="$(legacy_output "$1")"
  progress="${output}/checkpoint_progress.pt"
  last="${output}/checkpoint_last.pt"
  if [[ -s "${progress}" ]] && \
     { [[ ! -s "${last}" ]] || [[ "${progress}" -nt "${last}" ]]; }; then
    printf '%s' "${progress}"
  else
    printf '%s' "${last}"
  fi
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

is_legacy_training_pid() {
  local pid="$1"
  local command_line
  command_line="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
  [[ "${command_line}" == *"-m dinov2_segmentation.train_joint "* ]] && \
    [[ "${command_line}" == *"--output-dir ${work_root}/decoder_runs_improved/"* ]]
}

target_gpus_have_only_legacy_compute() {
  local gpu_id pid
  local -a gpu_ids compute_pids
  IFS=',' read -r -a gpu_ids <<< "${gpu_list}"
  for gpu_id in "${gpu_ids[@]}"; do
    mapfile -t compute_pids < <(
      nvidia-smi -i "${gpu_id}" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
    )
    for pid in "${compute_pids[@]}"; do
      pid="${pid//[[:space:]]/}"
      [[ -z "${pid}" ]] || is_legacy_training_pid "${pid}" || return 1
    done
  done
  return 0
}

checkpoint_ready() {
  local path="$1"
  [[ -s "${path}" && ! -e "${path}.tmp" && -s "$(dirname "${path}")/history.json" ]]
}

echo "[$(timestamp)] Waiting for serial S/ST resumable step or epoch checkpoints."
echo "[$(timestamp)] DDP target GPUs=${gpu_list}, per-GPU batch=${per_gpu_batch}, workers/rank=${workers}."
while true; do
  s_checkpoint="$(legacy_checkpoint S)"
  st_checkpoint="$(legacy_checkpoint ST)"
  if checkpoint_ready "${s_checkpoint}" && checkpoint_ready "${st_checkpoint}"; then
    if [[ "${require_target_gpus_safe}" == 0 ]] || target_gpus_have_only_legacy_compute; then
      # The writer renames checkpoint_last.pt atomically and writes history
      # immediately afterwards.  A short second check avoids stopping in the
      # narrow interval of a following epoch-boundary update.
      sleep 15
      if checkpoint_ready "${s_checkpoint}" && checkpoint_ready "${st_checkpoint}"; then
        break
      fi
    fi
    echo "[$(timestamp)] Checkpoints are ready; a target GPU has another compute job, so serial runs remain untouched."
  else
    for profile in S ST; do
      checkpoint="$(legacy_checkpoint "${profile}")"
      if ! checkpoint_ready "${checkpoint}" && \
         ! pgrep -u "$(id -u)" -f -- "--output-dir $(legacy_output "${profile}")" >/dev/null; then
        echo "[$(timestamp)] ${profile} stopped before creating ${checkpoint}; refusing to discard its state." >&2
        exit 1
      fi
    done
    echo "[$(timestamp)] Serial checkpoints are not both ready yet; leaving S/ST running."
  fi
  sleep "${poll_seconds}"
done

if [[ "$(source_digest)" != "${initial_source_digest}" ]]; then
  echo "[$(timestamp)] Migration source changed while waiting; leaving serial runs untouched." >&2
  exit 1
fi
echo "[$(timestamp)] Both checkpoints are ready and target GPUs are safe; stopping only the legacy runs."
for profile in S ST STA; do
  mapfile -t profile_pids < <(
    pgrep -u "$(id -u)" -f -- "--output-dir $(legacy_output "${profile}")" || true
  )
  if (( ${#profile_pids[@]} > 0 )); then
    kill -TERM "${profile_pids[@]}"
  fi
done

# Let the old launcher reap its children and terminate its monitor.  Do not use
# SIGKILL: failure to stop cleanly must be visible instead of risking a corrupt
# CUDA/process state.
deadline=$((SECONDS + 180))
while (( SECONDS < deadline )); do
  remaining=0
  for profile in S ST STA; do
    if pgrep -u "$(id -u)" -f -- "--output-dir $(legacy_output "${profile}")" >/dev/null; then
      remaining=1
    fi
  done
  (( remaining == 1 )) || break
  sleep 2
done
if (( remaining == 1 )); then
  echo "[$(timestamp)] Legacy processes did not stop after SIGTERM; DDP was not started." >&2
  exit 1
fi

echo "[$(timestamp)] Legacy processes stopped. Starting the sequential two-GPU DDP queue."
for profile in S ST STA; do
  output_dir="${work_root}/decoder_runs_parallel/${profile}/${stage2_variant}/v1_ddp${run_suffix}"
  profile_log="${log_dir}/${profile}_v1_ddp${run_suffix}.log"
  transfer_env=()
  if [[ "${profile}" == S || "${profile}" == ST ]]; then
    if [[ ! -f "${output_dir}/checkpoint_last.pt" && \
          ! -f "${output_dir}/checkpoint_progress.pt" ]]; then
      transfer_env=(MIGRATE_RESUME="$(legacy_checkpoint "${profile}")")
    fi
  fi
  echo "[$(timestamp)] Launching ${profile}; output=${output_dir}"
  env \
    CERVICAL_WORK_ROOT="${work_root}" \
    BATCH_SIZE="${per_gpu_batch}" \
    DECODER_WORKERS="${workers}" \
    RUN_SUFFIX="${run_suffix}" \
    "${transfer_env[@]}" \
    bash "${launcher}" ddp "${profile}" "${stage2_variant}" v1 "${gpu_list}" \
    > >(tee -a "${profile_log}") 2>&1
  echo "[$(timestamp)] ${profile} DDP run finished."
done

touch "${state_dir}/complete"
echo "[$(timestamp)] All migrated DDP runs finished."
