#!/usr/bin/env bash

# Run one non-interactive Codex training check. Scheduling belongs to cron.
set -euo pipefail

repo_dir="/home/user/90T/xiayw/CerviPath"
codex_bin="${CERVIPATH_CODEX_BIN:-/home/user/90T/xiayw/.nvm/versions/node/v24.18.0/bin/codex}"
prompt_file="${CERVIPATH_CODEX_PROMPT:-${repo_dir}/automation/hourly_training_monitor_prompt.md}"
state_dir="${CERVIPATH_CODEX_STATE_DIR:-/home/user/90T/xiayw/.local/state/cervipath-codex-monitor}"
runs_dir="${state_dir}/runs"

export PATH="/home/user/90T/xiayw/.nvm/versions/node/v24.18.0/bin:/usr/local/bin:/usr/bin:/bin"
umask 077

for required in "${codex_bin}" "${prompt_file}" "${repo_dir}/.git"; do
  if [[ ! -e "${required}" ]]; then
    printf 'Missing required path: %s\n' "${required}" >&2
    exit 1
  fi
done
for command in flock git; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "${command}" >&2
    exit 1
  fi
done

mkdir -p "${runs_dir}"
exec 9>"${state_dir}/hourly-training-check.lock"
if ! flock -n 9; then
  printf '%s skipped: previous hourly Codex check is still running\n' \
    "$(date '+%F %T %Z')" >> "${state_dir}/scheduler.log"
  exit 0
fi

if [[ "${1:-}" == "--check" ]]; then
  "${codex_bin}" --version
  "${codex_bin}" login status
  git -C "${repo_dir}" rev-parse --show-toplevel
  printf 'prompt=%s\nstate=%s\n' "${prompt_file}" "${state_dir}"
  exit 0
fi
if [[ $# -ne 0 ]]; then
  printf 'Usage: %s [--check]\n' "$0" >&2
  exit 2
fi

run_id="$(date '+%Y%m%dT%H%M%S%z')"
run_log="${runs_dir}/${run_id}.log"
last_message="${runs_dir}/${run_id}.last-message.md"
printf '%s\n' "${run_id}" > "${state_dir}/latest-run-id"

{
  printf '%s start commit=%s\n' \
    "$(date '+%F %T %Z')" "$(git -C "${repo_dir}" rev-parse HEAD)"
  set +e
  "${codex_bin}" \
    --ask-for-approval never \
    --sandbox danger-full-access \
    --search \
    --cd "${repo_dir}" \
    exec \
    --ephemeral \
    --color never \
    --output-last-message "${last_message}" \
    - < "${prompt_file}"
  exit_code=$?
  set -e
  printf '%s finish exit_code=%d\n' "$(date '+%F %T %Z')" "${exit_code}"
} >> "${run_log}" 2>&1

printf '%s exit_code=%d log=%s\n' \
  "$(date '+%F %T %Z')" "${exit_code}" "${run_log}" \
  >> "${state_dir}/scheduler.log"
exit "${exit_code}"
