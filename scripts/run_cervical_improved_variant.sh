#!/usr/bin/env bash

# Launch one additive imbalance-aware training ablation without touching the
# original six-model output directories.
set -euo pipefail

if (( $# < 4 || $# > 5 )); then
  echo "Usage: $0 <S|ST|STA> <baseline|distance_only|weighted_pretrain_distance_context> <v1|v2> <gpu-id> [stage2-run-id]" >&2
  exit 2
fi

profile="$1"
stage2_variant="$2"
decoder_version="$3"
gpu_id="$4"
stage2_run_id="${5:-s1_iter39999_compare_01_retry2}"

case "${profile}" in
  S)
    profile_args=(--overlap-loss dice --color-augmentation none)
    ;;
  ST)
    profile_args=(
      --overlap-loss foreground_tversky
      --tversky-alpha 0.3
      --tversky-beta 0.7
      --color-augmentation none
    )
    ;;
  STA)
    profile_args=(
      --overlap-loss foreground_tversky
      --tversky-alpha 0.3
      --tversky-beta 0.7
      --color-augmentation mild
    )
    ;;
  *)
    echo "Unsupported profile: ${profile}; expected S, ST or STA" >&2
    exit 2
    ;;
esac
case "${stage2_variant}" in
  baseline) graph_name=dual ;;
  distance_only|weighted_pretrain_distance_context) graph_name=distance ;;
  *)
    echo "Unsupported Stage2 variant: ${stage2_variant}" >&2
    exit 2
    ;;
esac
case "${decoder_version}" in
  v1|v2) ;;
  *)
    echo "Unsupported decoder version: ${decoder_version}" >&2
    exit 2
    ;;
esac
[[ "${gpu_id}" =~ ^[0-9]+$ ]] || {
  echo "gpu-id must be a non-negative integer" >&2
  exit 2
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_dir}/dinov2/.venv/bin/python}"
work_root="${CERVICAL_WORK_ROOT:-${repo_dir}/Data/cervical_segmentation_expanded_min64_b4}"
stage1_root="${repo_dir}/dinov2_stage1_Extract2s2/dinov2/results/stage1a_spatial_fusion_cosine200_e800_minlr1e-8"
stage2_root="${repo_dir}/dinov2_stage2_2_FmH2ST/dinov2/results/stage2_variants/${stage2_variant}/${stage2_run_id}"
stage2_source="${repo_dir}/dinov2_stage2_2_FmH2ST"
epochs="${DECODER_EPOCHS:-50}"
batch_size="${BATCH_SIZE:-16}"
workers="${DECODER_WORKERS:-8}"
decoder_drop_path_rate="${DECODER_DROP_PATH_RATE:-0.1}"
run_suffix="${RUN_SUFFIX:-}"
output_dir="${IMPROVED_OUTPUT_ROOT:-${work_root}/decoder_runs_improved}/${profile}/${stage2_variant}/${decoder_version}${run_suffix}"

required=(
  "${python_bin}"
  "${work_root}/decoder_selection/train.csv"
  "${work_root}/decoder_selection/valid.csv"
  "${work_root}/graphs/${graph_name}"
  "${stage1_root}/config.yaml"
  "${stage1_root}/cycle_checkpoints/model_0039999.rank_0.pth"
  "${stage2_root}/config.yaml"
  "${stage2_root}/model_final.rank_0.pth"
)
for path in "${required[@]}"; do
  [[ -e "${path}" ]] || {
    echo "Missing required path: ${path}" >&2
    exit 1
  }
done

resume_args=()
if [[ -f "${output_dir}/complete" ]]; then
  echo "Improved run is already complete: ${output_dir}"
  exit 0
elif [[ -f "${output_dir}/checkpoint_last.pt" ]]; then
  resume_args=(--resume "${output_dir}/checkpoint_last.pt")
elif [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing non-empty output without checkpoint_last.pt: ${output_dir}" >&2
  exit 1
fi

limit_args=()
if (( ${MAX_TRAIN_BATCHES:-0} > 0 )); then
  limit_args+=(--max-train-batches "${MAX_TRAIN_BATCHES}")
fi
if (( ${MAX_VAL_BATCHES:-0} > 0 )); then
  limit_args+=(--max-val-batches "${MAX_VAL_BATCHES}")
fi

command=(
  "${python_bin}" -m dinov2_segmentation.train_joint
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
  --gradient-accumulation 1
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
  "${profile_args[@]}"
  "${limit_args[@]}"
  "${resume_args[@]}"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'cd %q && CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q ' \
    "${stage2_source}" "${gpu_id}" "${stage2_source}:${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${output_dir}"
cd "${stage2_source}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${stage2_source}:${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${command[@]}"
