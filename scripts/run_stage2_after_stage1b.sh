#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_dir}/dinov2/.venv/bin/python}"
stage1b_session="${STAGE1B_SESSION:-cervipath_stage1b_iter39999}"
stage1b_dir="${STAGE1B_DIR:-${repo_dir}/dinov2_stage1_Extract2s2/dinov2/results/stage1b_features_iter39999}"
embeddings_dir="${STAGE1B_EMBEDDINGS_DIR:-${stage1b_dir}/embeddings}"
patch_csv="${PATCH_CSV:-${repo_dir}/Data/meta/patch/patch_grid_positions-reconstructed-train.csv}"
organized_dir="${ORGANIZED_DIR:-${repo_dir}/Graph/embeddings-current}"
stage2_dir="${repo_dir}/dinov2_stage2_2_FmH2ST"
experiment_dir="${stage2_dir}/experiments/stage2_variants"
expected_shards="${EXPECTED_SHARDS:-3594}"
expected_slides="${EXPECTED_SLIDES:-2188}"
run_id="${RUN_ID:-s1_iter39999_compare_01}"
gpu_threshold_mb="${GPU_BUSY_THRESHOLD_MB:-4096}"
poll_seconds="${POLL_SECONDS:-60}"
gpu_poll_seconds="${GPU_POLL_SECONDS:-300}"

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*"
}

count_files() {
  local directory="$1"
  local pattern="$2"
  find "${directory}" -maxdepth 1 -type f -name "${pattern}" -print | wc -l
}

for required in "${python_bin}" "${patch_csv}" "${embeddings_dir}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing Stage-2 pipeline input: ${required}" >&2
    exit 1
  fi
done
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required to observe the Stage-1B export session." >&2
  exit 2
fi

log "Waiting for Stage-1B session ${stage1b_session}"
while true; do
  pane_status="$(
    tmux list-panes -t "${stage1b_session}" \
      -F '#{pane_dead} #{pane_dead_status}' 2>/dev/null || true
  )"
  if [[ -z "${pane_status}" ]]; then
    log "Stage-1B session is absent; validating completed shards directly"
    break
  fi
  read -r pane_dead pane_exit <<< "${pane_status}"
  if [[ "${pane_dead}" == "0" ]]; then
    sleep "${poll_seconds}"
    continue
  fi
  if [[ "${pane_exit:-1}" != "0" ]]; then
    echo "Stage-1B exited unsuccessfully with status ${pane_exit:-unknown}." >&2
    exit 3
  fi
  log "Stage-1B exited successfully"
  break
done

for pattern in \
  'decoder_features_s1_*.pt' \
  'features_pretrained_s1_*.npz' \
  'filenames_pretrained_s1_*.npz'; do
  count="$(count_files "${embeddings_dir}" "${pattern}")"
  if [[ "${count}" != "${expected_shards}" ]]; then
    echo "Expected ${expected_shards} files matching ${pattern}, found ${count}." >&2
    exit 4
  fi
done
log "Validated ${expected_shards} complete Stage-1B shard groups"

if [[ -n "$(find "${organized_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite organized Stage-1 output: ${organized_dir}" >&2
  exit 5
fi
log "Organizing Stage-1 features by WSI"
"${python_bin}" "${repo_dir}/dinov2_stage1_Extract2s2/organize_features.py" \
  --embeddings-dir "${embeddings_dir}" \
  --patch-csv "${patch_csv}" \
  --output-dir "${organized_dir}" \
  --require-dense-tokens

for suffix in features.npy coords.npy metadata.pt dense_tokens.npy; do
  count="$(count_files "${organized_dir}" "*_${suffix}")"
  if [[ "${count}" != "${expected_slides}" ]]; then
    echo "Expected ${expected_slides} organized *_${suffix} files, found ${count}." >&2
    exit 6
  fi
done
log "Validated ${expected_slides} organized WSI feature sets"

log "Building baseline dual-edge graphs"
PYTHON_BIN="${python_bin}" \
EMBEDDINGS_DIR="${organized_dir}" \
METADATA_DIR="${organized_dir}" \
  bash "${experiment_dir}/prepare_graphs.sh" baseline all

log "Building distance-only graphs"
PYTHON_BIN="${python_bin}" \
EMBEDDINGS_DIR="${organized_dir}" \
METADATA_DIR="${organized_dir}" \
  bash "${experiment_dir}/prepare_graphs.sh" distance_only all

for graph_set in dual distance; do
  graph_dir="${repo_dir}/Graph/stage2_variants/${graph_set}"
  count="$(count_files "${graph_dir}" '*.pt')"
  if [[ "${count}" != "${expected_slides}" ]]; then
    echo "Expected ${expected_slides} ${graph_set} graphs, found ${count}." >&2
    exit 7
  fi
  if [[ ! -f "${graph_dir}/stage2_graph_manifest.json" ]]; then
    echo "Missing graph manifest: ${graph_dir}/stage2_graph_manifest.json" >&2
    exit 7
  fi
done
log "Validated both Stage-2 graph sets"

log "Waiting for GPUs 0 and 1 to fall below ${gpu_threshold_mb} MiB used"
while true; do
  memory0="$(
    nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits |
      head -n 1 | tr -d ' '
  )"
  memory1="$(
    nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits |
      head -n 1 | tr -d ' '
  )"
  if [[ "${memory0}" =~ ^[0-9]+$ && "${memory1}" =~ ^[0-9]+$ ]] &&
     (( memory0 <= gpu_threshold_mb && memory1 <= gpu_threshold_mb )); then
    break
  fi
  log "GPU memory still busy: gpu0=${memory0}MiB gpu1=${memory1}MiB"
  sleep "${gpu_poll_seconds}"
done

log "Starting Stage-2 baseline and distance_only with run_id=${run_id}"
cd "${stage2_dir}"
PYTHON_BIN="${python_bin}" \
GPU_MAX_USED_MB_BEFORE_START="${gpu_threshold_mb}" \
  exec bash "${experiment_dir}/run_parallel.sh" \
    "${run_id}" baseline distance_only
