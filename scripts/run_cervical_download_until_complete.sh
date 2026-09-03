#!/usr/bin/env bash

# Re-run the resumable downloader after transient failures until every entry
# in the upstream manifest passes its exact-size check.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
segment_data_dir="${CERVICAL_SEGMENT_DATA_DIR:-/home/user/90T/xiayw/muvit/data/Segment-Data}"
dataset_dir="${1:-${segment_data_dir}/Cervical Image}"
downloader="${repo_dir}/scripts/download_cervical_dataset.sh"
log_file="${CERVICAL_DOWNLOAD_LOG:-${segment_data_dir}/cervical_download.log}"
retry_pause="${DOWNLOAD_RETRY_PAUSE_SECONDS:-60}"
supervisor_lock_file="${CERVICAL_DOWNLOAD_SUPERVISOR_LOCK_FILE:-/tmp/S-BIAD1168-cervical-download-supervisor.lock}"

if [[ ! "${retry_pause}" =~ ^[0-9]+$ ]] || (( retry_pause < 1 )); then
    echo "DOWNLOAD_RETRY_PAUSE_SECONDS must be a positive integer" >&2
    exit 2
fi
mkdir -p "$(dirname "${log_file}")"

exec 8>"${supervisor_lock_file}"
if ! flock -n 8; then
    echo "Another cervical download supervisor is already running." >&2
    exit 1
fi

pass=1
while true; do
    printf '%s START download pass %d\n' "$(date '+%F %T')" "${pass}" \
        >> "${log_file}"
    bash "${downloader}" "${dataset_dir}" >> "${log_file}" 2>&1
    status=$?
    if (( status == 0 )); then
        printf '%s DOWNLOAD COMPLETE after pass %d\n' \
            "$(date '+%F %T')" "${pass}" >> "${log_file}"
        exit 0
    fi
    printf '%s pass %d failed with status %d; retrying in %d seconds\n' \
        "$(date '+%F %T')" "${pass}" "${status}" "${retry_pause}" \
        >> "${log_file}"
    sleep "${retry_pause}"
    ((pass += 1))
done
