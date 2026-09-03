#!/usr/bin/env bash

# Persistent end-to-end runner for the frozen partial cervical cohort.
# Final experiment matrix: 3 Stage-2 variants x Decoder V1/V2 = 6 models.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_dir}/dinov2/.venv/bin/python}"
raw_root="${CERVICAL_RAW_ROOT:-/home/user/90T/xiayw/muvit/data/Segment-Data/Cervical Image}"
download_manifest="${CERVICAL_DOWNLOAD_MANIFEST:-/home/user/90T/xiayw/muvit/data/Segment-Data/S-BIAD1168_Cervical_Images.json}"
metadata_xlsx="${CERVICAL_METADATA_XLSX:-${raw_root}/Cervical image anootation categories.xlsx}"
work_root="${CERVICAL_WORK_ROOT:-${repo_dir}/Data/cervical_segmentation_partial}"
stage2_run_id="${STAGE2_RUN_ID:-s1_iter39999_compare_01_retry2}"
minimum_ready="${MINIMUM_READY_SLIDES:-32}"
minimum_train="${MINIMUM_TRAIN_SLIDES:-16}"
minimum_valid="${MINIMUM_VALID_SLIDES:-8}"
poll_seconds="${POLL_SECONDS:-300}"
gpu_threshold_mb="${GPU_BUSY_THRESHOLD_MB:-4096}"
stage1_batch="${STAGE1_BATCH_SIZE:-64}"
stage1_workers="${STAGE1_WORKERS:-8}"
decoder_epochs="${DECODER_EPOCHS:-50}"
decoder_workers="${DECODER_WORKERS:-8}"
prepare_workers="${PREPARE_WORKERS:-4}"
v1_batch="${V1_BATCH_SIZE:-16}"
v2_batch="${V2_BATCH_SIZE:-16}"
v1_gpu="${V1_GPU_ID:-0}"
v2_gpu="${V2_GPU_ID:-1}"
joint_accumulation="${JOINT_GRADIENT_ACCUMULATION:-1}"
decoder_only_epochs="${DECODER_ONLY_EPOCHS:-3}"
stage1_top_unfreeze_epoch="${STAGE1_TOP_UNFREEZE_EPOCH:-8}"
stage1_unfreeze_blocks="${STAGE1_UNFREEZE_BLOCKS:-4}"
early_stopping_patience="${EARLY_STOPPING_PATIENCE:-3}"
early_stopping_start_epoch="${EARLY_STOPPING_START_EPOCH:-12}"
early_stopping_min_delta="${EARLY_STOPPING_MIN_DELTA:-0.001}"
tumor_ce_weight="${TUMOR_CE_WEIGHT:-1.5}"
joint_variants_raw="${JOINT_VARIANTS:-baseline distance_only weighted_pretrain_distance_context}"
read -r -a joint_variants <<< "${joint_variants_raw}"
(( ${#joint_variants[@]} > 0 )) || {
  echo "JOINT_VARIANTS must select at least one Stage2 variant." >&2
  exit 1
}
for variant in "${joint_variants[@]}"; do
  case "${variant}" in
    baseline|distance_only|weighted_pretrain_distance_context) ;;
    *)
      echo "Unsupported Stage2 variant in JOINT_VARIANTS: ${variant}" >&2
      exit 1
      ;;
  esac
done

stage1_dir="${repo_dir}/dinov2_stage1_Extract2s2"
stage2_dir="${repo_dir}/dinov2_stage2_2_FmH2ST"
stage2_experiments="${stage2_dir}/experiments/stage2_variants"
stage2_results="${stage2_dir}/dinov2/results/stage2_variants"
selected_stage1_checkpoint="${STAGE1_CHECKPOINT:-${stage1_dir}/dinov2/results/stage1a_spatial_fusion_cosine200_e800_minlr1e-8/cycle_checkpoints/model_0039999.rank_0.pth}"
selected_stage1_config="${STAGE1_CONFIG:-${stage1_dir}/dinov2/results/stage1a_spatial_fusion_cosine200_e800_minlr1e-8/config.yaml}"
logs_dir="${work_root}/logs"
status_log="${logs_dir}/pipeline.log"

mkdir -p "${logs_dir}"
cd "${repo_dir}"
exec 9>"${work_root}/pipeline.lock"
if ! flock -n 9; then
  echo "Another cervical six-model pipeline is already active." >&2
  exit 1
fi

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "${status_log}"
}

fail() {
  log "ERROR: $*"
  exit 1
}

count_files() {
  find "$1" -maxdepth 1 -type f -name "$2" -print 2>/dev/null | wc -l
}

wait_for_gpu() {
  local gpu_id="$1"
  while true; do
    local used
    used="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= gpu_threshold_mb )); then
      return
    fi
    log "Waiting for GPU ${gpu_id}: used=${used}MiB threshold=${gpu_threshold_mb}MiB"
    sleep "${poll_seconds}"
  done
}

wait_for_existing_stage2() {
  local variant="$1"
  local run_dir="${stage2_results}/${variant}/${stage2_run_id}"
  local checkpoint="${run_dir}/model_final.rank_0.pth"
  while [[ ! -f "${checkpoint}" ]]; do
    if ! pgrep -f -- "${run_dir}" >/dev/null; then
      fail "Stage2 ${variant} stopped before producing ${checkpoint}"
    fi
    local latest="-"
    if [[ -f "${run_dir}/training_metrics.json" ]]; then
      latest="$(tail -n 1 "${run_dir}/training_metrics.json" | sed -n 's/.*"iteration": *\([0-9]*\).*/\1/p')"
    fi
    log "Waiting for existing Stage2 ${variant}; latest_iteration=${latest:--}"
    sleep "${poll_seconds}"
  done
  log "Stage2 ${variant} complete: ${checkpoint}"
}

status_args=(
  --data-root "${raw_root}"
  --download-manifest "${download_manifest}"
  --metadata-xlsx "${metadata_xlsx}"
  --minimum-ready "${minimum_ready}"
  --minimum-train "${minimum_train}"
  --minimum-valid "${minimum_valid}"
)

for required in "${python_bin}" "${download_manifest}" "${metadata_xlsx}" "${selected_stage1_checkpoint}" "${selected_stage1_config}"; do
  [[ -e "${required}" ]] || fail "Missing required resource: ${required}"
done
for command in flock nvidia-smi; do
  command -v "${command}" >/dev/null 2>&1 || fail "Required command is unavailable: ${command}"
done
"${python_bin}" -c "import isyntax, openpyxl, torch, torch_geometric" \
  >/dev/null 2>&1 || fail "Python environment lacks a cervical-pipeline dependency"

log "Pipeline commit: $(git -C "${repo_dir}" rev-parse HEAD)"
log "Waiting for the partial-data readiness gate"
while ! "${python_bin}" -m dinov2_segmentation.prepare_cervical_data status \
  "${status_args[@]}" >> "${status_log}" 2>&1; do
  sleep "${poll_seconds}"
done
log "Partial-data threshold reached; freezing the currently complete annotated cohort"

"${python_bin}" -m dinov2_segmentation.prepare_cervical_data prepare \
  "${status_args[@]}" \
  --output-root "${work_root}" \
  --level 1 \
  --patch-size 224 \
  --stride 224 \
  --minimum-tissue-fraction 0.2 \
  --selection-seed 42 \
  --negative-ratio 3 \
  --workers "${prepare_workers}" \
  2>&1 | tee -a "${logs_dir}/prepare_data.log"
log "WSI patch extraction and binary mask preparation complete"

# Preserve the user's already-running jobs.  The third Stage-2 variant starts
# only after both existing variants finish and GPU 0 becomes idle.
wait_for_existing_stage2 baseline
wait_for_existing_stage2 distance_only

weighted_dir="${stage2_results}/weighted_pretrain_distance_context/${stage2_run_id}"
weighted_checkpoint="${weighted_dir}/model_final.rank_0.pth"
if [[ ! -f "${weighted_checkpoint}" ]]; then
  if pgrep -f -- "${weighted_dir}" >/dev/null; then
    log "Weighted-pretrain/distance-context Stage2 is already running"
    wait_for_existing_stage2 weighted_pretrain_distance_context
  else
    if [[ -n "$(find "${weighted_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      fail "Refusing to overwrite incomplete Stage2 directory: ${weighted_dir}"
    fi
    wait_for_gpu 0
    log "Starting Stage2 weighted_pretrain_distance_context on GPU 0"
    PYTHON_BIN="${python_bin}" GPU_MAX_USED_MB_BEFORE_START="${gpu_threshold_mb}" \
      bash "${stage2_experiments}/run_variant.sh" \
      weighted_pretrain_distance_context 0 "${stage2_run_id}" \
      > "${logs_dir}/stage2_weighted_pretrain_distance_context.log" 2>&1
    [[ -f "${weighted_checkpoint}" ]] || fail "Third Stage2 run exited without a final checkpoint"
  fi
fi

stage1_output="${work_root}/stage1b"
stage1_embeddings="${stage1_output}/embeddings"
stage1_organized="${work_root}/stage1_organized"
if [[ ! -f "${stage1_output}/complete" ]]; then
  if [[ -n "$(find "${stage1_output}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    fail "Incomplete Stage1B output exists; preserve it and choose a new CERVICAL_WORK_ROOT: ${stage1_output}"
  fi
  wait_for_gpu 0
  log "Extracting cervical patch features with the selected Stage1 checkpoint"
  (
    cd "${stage1_dir}"
    export CUDA_VISIBLE_DEVICES=0
    exec "${python_bin}" -m dinov2.train.train \
      --eval-only \
      --no-resume \
      --config-file dinov2/configs/train/vitl16_short_imgnet22k.yaml \
      --output-dir "${stage1_output}" \
      "MODEL.WEIGHTS=${selected_stage1_checkpoint}" \
      "feature.export_dense_tokens=false" \
      "feature.require_trained_spatial=true" \
      "train.batch_size_per_gpu=${stage1_batch}" \
      "train.num_workers=${stage1_workers}" \
      "train.dataset_path=PatchManifest:root=${work_root}/all_patches.csv"
  ) > "${logs_dir}/stage1b.log" 2>&1
  "${python_bin}" "${stage1_dir}/organize_features.py" \
    --embeddings-dir "${stage1_embeddings}" \
    --patch-csv "${work_root}/all_patches.csv" \
    --output-dir "${stage1_organized}" \
    2>&1 | tee -a "${logs_dir}/organize_stage1.log"
  touch "${stage1_output}/complete"
fi
log "Stage1 feature extraction complete"

graph_dual="${work_root}/graphs/dual"
graph_distance="${work_root}/graphs/distance"
(
  cd "${stage2_dir}"
  "${python_bin}" build_graphs.py \
    --embeddings-dir "${stage1_organized}" \
    --metadata-dir "${stage1_organized}" \
    --output-dir "${graph_dual}" \
    --edge-mode dual --k 8 --distance-multiplier 3.0 \
    --require-decoder-metadata
  "${python_bin}" build_graphs.py \
    --embeddings-dir "${stage1_organized}" \
    --metadata-dir "${stage1_organized}" \
    --output-dir "${graph_distance}" \
    --edge-mode distance --k 8 --distance-multiplier 3.0 \
    --require-decoder-metadata
) 2>&1 | tee -a "${logs_dir}/build_graphs.log"
log "Cervical dual-edge and distance-only graph topology/feature memories complete"

run_decoder_pair() {
  local variant="$1"
  local graph_dir="$2"
  local stage2_run_dir="${stage2_results}/${variant}/${stage2_run_id}"
  local output_root="${work_root}/decoder_runs/${variant}"
  local v1_output="${output_root}/v1"
  local v2_output="${output_root}/v2"
  local -a v1_resume=()
  local -a v2_resume=()
  while pgrep -f -- "--output-dir ${v1_output}" >/dev/null; do
    log "Waiting for independently running decoder: ${variant}/v1"
    sleep "${poll_seconds}"
  done
  while pgrep -f -- "--output-dir ${v2_output}" >/dev/null; do
    log "Waiting for independently running decoder: ${variant}/v2"
    sleep "${poll_seconds}"
  done
  if [[ -f "${v1_output}/checkpoint_last.pt" ]]; then
    v1_resume=(--resume "${v1_output}/checkpoint_last.pt")
  elif [[ -n "$(find "${v1_output}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    fail "V1 output is non-empty without a resumable checkpoint: ${v1_output}"
  fi
  if [[ -f "${v2_output}/checkpoint_last.pt" ]]; then
    v2_resume=(--resume "${v2_output}/checkpoint_last.pt")
  elif [[ -n "$(find "${v2_output}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    fail "V2 output is non-empty without a resumable checkpoint: ${v2_output}"
  fi
  if [[ -f "${v1_output}/complete" && -f "${v2_output}/complete" ]]; then
    log "Decoder pair already complete: ${variant}"
    return
  fi
  wait_for_gpu "${v1_gpu}"
  if [[ "${v2_gpu}" != "${v1_gpu}" ]]; then
    wait_for_gpu "${v2_gpu}"
  fi
  mkdir -p "${v1_output}" "${v2_output}"
  log "Starting joint Stage1+Stage2+decoder pair ${variant}: V1/GPU${v1_gpu} and V2/GPU${v2_gpu}"
  (
    cd "${stage2_dir}"
    export CUDA_VISIBLE_DEVICES="${v1_gpu}"
    export PYTHONPATH="${stage2_dir}:${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
    exec "${python_bin}" -m dinov2_segmentation.train_joint \
      --decoder-version v1 \
      --train-manifest "${work_root}/decoder_selection/train.csv" \
      --val-manifest "${work_root}/decoder_selection/valid.csv" \
      --graph-dir "${graph_dir}" \
      --stage1-config "${selected_stage1_config}" \
      --stage1-checkpoint "${selected_stage1_checkpoint}" \
      --stage2-config "${stage2_run_dir}/config.yaml" \
      --stage2-checkpoint "${stage2_run_dir}/model_final.rank_0.pth" \
      --output-dir "${v1_output}" \
      --num-classes 2 --epochs "${decoder_epochs}" \
      --batch-size "${v1_batch}" --workers "${decoder_workers}" \
      --gradient-accumulation "${joint_accumulation}" \
      --decoder-lr 1e-4 --stage2-lr 1e-5 \
      --stage1-fusion-lr 1e-5 --stage1-backbone-lr 2e-6 \
      --layer-decay 0.8 --warmup-ratio 0.1 --min-lr-ratio 0.01 \
      --decoder-only-epochs "${decoder_only_epochs}" \
      --stage1-top-unfreeze-epoch "${stage1_top_unfreeze_epoch}" \
      --stage1-unfreeze-blocks "${stage1_unfreeze_blocks}" \
      --final-phase-pretrained-lr-scale 0.5 \
      --final-phase-decoder-lr-scale 0.5 \
      --tumor-ce-weight "${tumor_ce_weight}" \
      --early-stopping-patience "${early_stopping_patience}" \
      --early-stopping-start-epoch "${early_stopping_start_epoch}" \
      --early-stopping-min-delta "${early_stopping_min_delta}" \
      "${v1_resume[@]}"
  ) >> "${logs_dir}/decoder_${variant}_v1.log" 2>&1 &
  local pid_v1=$!
  (
    cd "${stage2_dir}"
    export CUDA_VISIBLE_DEVICES="${v2_gpu}"
    export PYTHONPATH="${stage2_dir}:${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
    exec "${python_bin}" -m dinov2_segmentation.train_joint \
      --decoder-version v2 \
      --train-manifest "${work_root}/decoder_selection/train.csv" \
      --val-manifest "${work_root}/decoder_selection/valid.csv" \
      --graph-dir "${graph_dir}" \
      --stage1-config "${selected_stage1_config}" \
      --stage1-checkpoint "${selected_stage1_checkpoint}" \
      --stage2-config "${stage2_run_dir}/config.yaml" \
      --stage2-checkpoint "${stage2_run_dir}/model_final.rank_0.pth" \
      --output-dir "${v2_output}" \
      --num-classes 2 --epochs "${decoder_epochs}" \
      --batch-size "${v2_batch}" --workers "${decoder_workers}" \
      --gradient-accumulation "${joint_accumulation}" \
      --decoder-lr 1e-4 --stage2-lr 1e-5 \
      --stage1-fusion-lr 1e-5 --stage1-backbone-lr 2e-6 \
      --layer-decay 0.8 --warmup-ratio 0.1 --min-lr-ratio 0.01 \
      --decoder-only-epochs "${decoder_only_epochs}" \
      --stage1-top-unfreeze-epoch "${stage1_top_unfreeze_epoch}" \
      --stage1-unfreeze-blocks "${stage1_unfreeze_blocks}" \
      --final-phase-pretrained-lr-scale 0.5 \
      --final-phase-decoder-lr-scale 0.5 \
      --tumor-ce-weight "${tumor_ce_weight}" \
      --early-stopping-patience "${early_stopping_patience}" \
      --early-stopping-start-epoch "${early_stopping_start_epoch}" \
      --early-stopping-min-delta "${early_stopping_min_delta}" \
      "${v2_resume[@]}"
  ) >> "${logs_dir}/decoder_${variant}_v2.log" 2>&1 &
  local pid_v2=$!
  local status_v1=0 status_v2=0
  wait "${pid_v1}" || status_v1=$?
  wait "${pid_v2}" || status_v2=$?
  (( status_v1 == 0 )) || fail "Decoder V1 failed for ${variant} with status ${status_v1}"
  (( status_v2 == 0 )) || fail "Decoder V2 failed for ${variant} with status ${status_v2}"
  touch "${v1_output}/complete" "${v2_output}/complete"
  log "Decoder pair complete: ${variant}"
}

for variant in "${joint_variants[@]}"; do
  case "${variant}" in
    baseline) run_decoder_pair "${variant}" "${graph_dual}" ;;
    distance_only|weighted_pretrain_distance_context)
      run_decoder_pair "${variant}" "${graph_distance}"
      ;;
  esac
done
"${python_bin}" - "${work_root}" "${decoder_epochs}" "${joint_variants[@]}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
epochs = int(sys.argv[2])
variants = tuple(sys.argv[3:])
for variant in variants:
    for decoder in ("v1", "v2"):
        output = root / "decoder_runs" / variant / decoder
        required = (output / "complete", output / "checkpoint_last.pt", output / "history.json", output / "gradient_audit.json", output / "early_stopping.json")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise SystemExit(f"Incomplete model {variant}/{decoder}: {missing}")
        history = json.loads((output / "history.json").read_text())
        early_stopping = json.loads((output / "early_stopping.json").read_text())
        valid_length = (
            len(history) == epochs and int(history[-1]["epoch"]) == epochs - 1
        ) or (
            0 < len(history) < epochs
            and bool(early_stopping.get("stopped_early"))
            and int(early_stopping.get("stop_epoch", -1)) == int(history[-1]["epoch"])
        )
        if not valid_length:
            raise SystemExit(f"Epoch audit failed for {variant}/{decoder}")
        gradients = json.loads((output / "gradient_audit.json").read_text())
        expected = ("stage1_grad_norm", "stage1_backbone_grad_norm", "stage2_grad_norm", "decoder_grad_norm")
        if not all(math.isfinite(float(gradients.get(key, 0))) and float(gradients[key]) > 0 for key in expected):
            raise SystemExit(f"Gradient audit failed for {variant}/{decoder}: {gradients}")
print(f"Verified {len(variants) * 2} complete joint models and non-zero Stage1/Stage2/decoder gradients")
PY
if (( ${#joint_variants[@]} == 3 )); then
  touch "${work_root}/six_models.complete"
else
  touch "${work_root}/selected_models.complete"
fi
log "All $(( ${#joint_variants[@]} * 2 )) selected segmentation models completed"
