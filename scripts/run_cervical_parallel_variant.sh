#!/usr/bin/env bash

# Run one model with the same architecture in single-process or multi-GPU DDP
# mode. Prepared input data are reused; this launcher never refreshes a cohort.
set -euo pipefail

if (( $# < 5 || $# > 6 )); then
  echo "Usage: $0 <serial|ddp> <S|ST|STA> <baseline|distance_only|weighted_pretrain_distance_context> <v1|v2> <gpu-list> [stage2-run-id]" >&2
  exit 2
fi

execution_mode="$1"
profile="$2"
stage2_variant="$3"
decoder_version="$4"
gpu_list="$5"
stage2_run_id="${6:-s1_iter39999_compare_01_retry2}"

case "${execution_mode}" in
  serial|ddp) ;;
  *) echo "execution mode must be serial or ddp" >&2; exit 2 ;;
esac
case "${profile}" in
  S) profile_args=(--overlap-loss dice --color-augmentation none) ;;
  ST|STA)
    color_augmentation=none
    [[ "${profile}" != STA ]] || color_augmentation=mild
    profile_args=(
      --overlap-loss foreground_tversky --tversky-alpha 0.3
      --tversky-beta 0.7 --color-augmentation "${color_augmentation}"
    )
    ;;
  *) echo "profile must be S, ST or STA" >&2; exit 2 ;;
esac
case "${stage2_variant}" in
  baseline) graph_name=dual ;;
  distance_only|weighted_pretrain_distance_context) graph_name=distance ;;
  *) echo "Unsupported Stage2 variant: ${stage2_variant}" >&2; exit 2 ;;
esac
case "${decoder_version}" in
  v1|v2) ;;
  *) echo "decoder version must be v1 or v2" >&2; exit 2 ;;
esac
[[ "${stage2_run_id}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || {
  echo "stage2-run-id must be a single directory name" >&2
  exit 2
}
[[ "${gpu_list}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || {
  echo "gpu-list must contain comma-separated GPU indices, for example 1 or 0,1" >&2
  exit 2
}
IFS=',' read -r -a gpu_ids <<< "${gpu_list}"
declare -A seen_gpus=()
for gpu_id in "${gpu_ids[@]}"; do
  [[ -z "${seen_gpus[${gpu_id}]:-}" ]] || {
    echo "Duplicate GPU index: ${gpu_id}" >&2
    exit 2
  }
  seen_gpus["${gpu_id}"]=1
done
world_size="${#gpu_ids[@]}"
if [[ "${execution_mode}" == serial ]] && (( world_size != 1 )); then
  echo "serial mode requires exactly one GPU index" >&2
  exit 2
fi
if [[ "${execution_mode}" == ddp ]] && (( world_size < 2 )); then
  echo "ddp mode requires at least two distinct GPUs; use serial for one GPU" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_dir}/dinov2/.venv/bin/python}"
work_root="${CERVICAL_WORK_ROOT:-${repo_dir}/Data/cervical_segmentation_latest_area_20260904}"
stage1_root="${STAGE1_ROOT:-${repo_dir}/dinov2_stage1_Extract2s2/dinov2/results/stage1a_spatial_fusion_cosine200_e800_minlr1e-8}"
stage2_root="${STAGE2_ROOT:-${repo_dir}/dinov2_stage2_2_FmH2ST/dinov2/results/stage2_variants/${stage2_variant}/${stage2_run_id}}"
stage2_source="${repo_dir}/dinov2_stage2_2_FmH2ST"
epochs="${DECODER_EPOCHS:-50}"
batch_size="${BATCH_SIZE:-16}"
workers="${DECODER_WORKERS:-8}"
gradient_accumulation="${GRADIENT_ACCUMULATION:-1}"
decoder_drop_path_rate="${DECODER_DROP_PATH_RATE:-0.2}"
max_train_batches="${MAX_TRAIN_BATCHES:-0}"
max_val_batches="${MAX_VAL_BATCHES:-0}"
run_suffix="${RUN_SUFFIX:-_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
[[ "${run_suffix}" =~ ^_[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || {
  echo "RUN_SUFFIX must start with _ and contain only letters, digits, _, . or -" >&2
  exit 2
}
output_dir="${PARALLEL_OUTPUT_ROOT:-${work_root}/decoder_runs_parallel}/${profile}/${stage2_variant}/${decoder_version}_${execution_mode}${run_suffix}"

for name in epochs batch_size gradient_accumulation; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || {
    echo "${name} must be a positive integer" >&2
    exit 2
  }
done
for name in workers max_train_batches max_val_batches; do
  [[ "${!name}" =~ ^(0|[1-9][0-9]*)$ ]] || {
    echo "${name} must be a non-negative integer" >&2
    exit 2
  }
done
(( epochs > 8 )) || {
  echo "DECODER_EPOCHS must be at least 9 for the configured unfreezing phases" >&2
  exit 2
}
[[ -x "${python_bin}" ]] || {
  echo "Python executable not found: ${python_bin}" >&2
  exit 1
}
[[ -f "${work_root}/preprocessing.complete" ]] || {
  echo "Preprocessing must finish first: ${work_root}/preprocessing.complete" >&2
  exit 1
}
[[ -d "${work_root}/graphs/${graph_name}" ]] || {
  echo "Missing prepared graph directory: ${work_root}/graphs/${graph_name}" >&2
  exit 1
}
required=(
  "${work_root}/decoder_selection/train.csv"
  "${work_root}/decoder_selection/valid.csv"
  "${stage1_root}/config.yaml"
  "${stage1_root}/cycle_checkpoints/model_0039999.rank_0.pth"
  "${stage2_root}/config.yaml"
  "${stage2_root}/model_final.rank_0.pth"
)
for path in "${required[@]}"; do
  [[ -s "${path}" ]] || {
    echo "Missing or empty required file: ${path}" >&2
    exit 1
  }
done

init_args=()
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  [[ -s "${INIT_CHECKPOINT}" ]] || {
    echo "Missing or empty INIT_CHECKPOINT: ${INIT_CHECKPOINT}" >&2
    exit 1
  }
  init_args=(--init-checkpoint "${INIT_CHECKPOINT}")
fi

# DRY_RUN is strictly read-only, including no lock/output-directory creation.
# Real runs inspect output state while holding an output-specific lock.
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  command -v flock >/dev/null || { echo "flock is required" >&2; exit 1; }
  mkdir -p "$(dirname "${output_dir}")"
  exec 9>"${output_dir}.lock"
  if ! flock -n 9; then
    echo "Another run holds the output lock: ${output_dir}" >&2
    exit 1
  fi
fi
resume_args=()
if [[ -f "${output_dir}/complete" ]]; then
  echo "Parallel-capable run is already complete: ${output_dir}"
  exit 0
elif [[ -f "${output_dir}/checkpoint_last.pt" ]]; then
  [[ -z "${INIT_CHECKPOINT:-}" ]] || {
    echo "INIT_CHECKPOINT requires a fresh output; this run already has a resume checkpoint" >&2
    exit 1
  }
  resume_args=(--resume "${output_dir}/checkpoint_last.pt")
elif [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty output without checkpoint_last.pt: ${output_dir}" >&2
  exit 1
elif [[ -e "${output_dir}" && ! -d "${output_dir}" ]]; then
  echo "Output path is not a directory: ${output_dir}" >&2
  exit 1
fi

command=("${python_bin}")
if [[ "${execution_mode}" == ddp ]]; then
  command+=(
    -m torch.distributed.run --standalone --nnodes=1
    --nproc_per_node "${world_size}"
  )
fi
command+=(
  -m dinov2_segmentation.train_joint_parallel
  --execution-mode "${execution_mode}"
  --experiment-profile "${profile}"
  --decoder-version "${decoder_version}"
  --decoder-drop-path-rate "${decoder_drop_path_rate}"
  --train-manifest "${work_root}/decoder_selection/train.csv"
  --val-manifest "${work_root}/decoder_selection/valid.csv"
  --graph-dir "${work_root}/graphs/${graph_name}"
  --stage1-config "${stage1_root}/config.yaml"
  --stage1-checkpoint "${stage1_root}/cycle_checkpoints/model_0039999.rank_0.pth"
  --stage2-config "${stage2_root}/config.yaml"
  --stage2-checkpoint "${stage2_root}/model_final.rank_0.pth"
  --output-dir "${output_dir}"
  --num-classes 2
  --epochs "${epochs}"
  --batch-size "${batch_size}"
  --workers "${workers}"
  --gradient-accumulation "${gradient_accumulation}"
  --decoder-lr 1e-4
  --stage2-lr 1e-5
  --stage1-fusion-lr 1e-5
  --stage1-backbone-lr 2e-6
  --layer-decay 0.8
  --warmup-ratio 0.1
  --min-lr-ratio 0.01
  --decoder-only-epochs 3
  --stage1-top-unfreeze-epoch 8
  --stage1-unfreeze-blocks 4
  --final-phase-pretrained-lr-scale 0.5
  --final-phase-decoder-lr-scale 0.5
  --early-stopping-patience 3
  --early-stopping-start-epoch 12
  --early-stopping-min-delta 0.001
  --cross-entropy-weight 1.0
  --overlap-weight 1.0
  --tumor-class-weight 1.0
  --sampling-mode slide_stratified
  --sampling-positive-fraction 0.60
  --sampling-boundary-positive-fraction 0.50
  --sampling-interior-threshold 0.999999
  --sampling-slide-balance-power 0.5
  --sampling-max-patch-repeats 2
  --sampling-epoch-samples 0
  --probability-metric-bins 256
  --max-train-batches "${max_train_batches}"
  --max-val-batches "${max_val_batches}"
  "${profile_args[@]}"
  "${init_args[@]}"
  "${resume_args[@]}"
)

echo "mode=${execution_mode}; GPUs=${gpu_list}; per-GPU batch=${batch_size}; effective batch=$((batch_size * world_size * gradient_accumulation))"
echo "output=${output_dir}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'cd %q && CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q ' \
    "${stage2_source}" "${gpu_list}" "${stage2_source}:${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

cd "${stage2_source}"
export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONPATH="${stage2_source}:${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
exec "${command[@]}"
