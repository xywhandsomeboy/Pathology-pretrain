#!/usr/bin/env bash

# Size-verified and resumable downloader for the S-BIAD1168 cervical cohort.
set -uo pipefail

segment_data_dir="${CERVICAL_SEGMENT_DATA_DIR:-/home/user/90T/xiayw/muvit/data/Segment-Data}"
dataset_dir="${1:-${segment_data_dir}/Cervical Image}"
base_url="${CERVICAL_DOWNLOAD_BASE_URL:-https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/168/S-BIAD1168/Files/Cervical%20Images}"
manifest_url="${CERVICAL_DOWNLOAD_MANIFEST_URL:-${base_url}.json}"
manifest_file="${CERVICAL_DOWNLOAD_MANIFEST:-${segment_data_dir}/S-BIAD1168_Cervical_Images.json}"
parallel_transfers="${CERVICAL_DOWNLOAD_PARALLEL:-4}"
manifest_tmp="${manifest_file}.tmp"
lock_file="${CERVICAL_DOWNLOAD_LOCK_FILE:-/tmp/S-BIAD1168-cervical-download.lock}"
quarantine_dir="${dataset_dir}/.download-quarantine"

if [[ ! "${parallel_transfers}" =~ ^[0-9]+$ ]] || (( parallel_transfers < 1 )); then
    echo "CERVICAL_DOWNLOAD_PARALLEL must be a positive integer" >&2
    exit 2
fi
for command in curl flock jq xargs; do
    command -v "${command}" >/dev/null 2>&1 || {
        echo "Required command is unavailable: ${command}" >&2
        exit 1
    }
done

mkdir -p "${dataset_dir}" "$(dirname "${manifest_file}")"

exec 9>"${lock_file}"
if ! flock -n 9; then
    echo "Another S-BIAD1168 download is already running." >&2
    exit 1
fi

if ! curl --fail --silent --show-error --location \
    --retry 10 --retry-all-errors --connect-timeout 30 \
    --speed-limit 1024 --speed-time 120 \
    --output "${manifest_tmp}" "${manifest_url}"; then
    echo "Failed to refresh the download manifest; keeping the existing manifest unchanged." >&2
    exit 1
fi
if ! jq -e '
    type == "array" and length > 0 and
    all(.[];
        (.path | type) == "string" and (.path | length) > 0 and
        (.size | type) == "number" and .size >= 0 and
        (.size | floor) == .size
    )
' "${manifest_tmp}" >/dev/null; then
    echo "Downloaded manifest has an invalid path/size schema; keeping the existing manifest unchanged." >&2
    exit 1
fi
if ! mv -- "${manifest_tmp}" "${manifest_file}"; then
    echo "Failed to install the verified download manifest." >&2
    exit 1
fi

download_one() {
    local expected_size="$1"
    local remote_filename="$2"
    local encoded_filename="$3"
    local local_filename="${remote_filename}"
    local legacy_target
    local target
    local current_size=0
    local final_size

    # The upstream manifest contains 2,348 names such as
    # ``IC-CX-00239-01 .geojson``. The blank is part of the remote URL but not
    # of the WSI stem used by the metadata and preprocessing pipeline.
    if [[ "${local_filename}" == *" .geojson" ]]; then
        local_filename="${local_filename% .geojson}.geojson"
    fi
    legacy_target="${dataset_dir}/${remote_filename}"
    target="${dataset_dir}/${local_filename}"
    if [[ "${legacy_target}" != "${target}" && -f "${legacy_target}" && ! -e "${target}" ]]; then
        mv -- "${legacy_target}" "${target}"
        echo "MIGRATE ${remote_filename} -> ${local_filename}"
    fi

    if [[ -f "${target}" ]]; then
        current_size="$(stat -c '%s' "${target}")"
    fi
    if [[ -f "${target}" && "${current_size}" == "${expected_size}" ]]; then
        echo "SKIP ${local_filename}"
        return 0
    fi
    if (( current_size > expected_size )); then
        if ! mkdir -p "${quarantine_dir}"; then
            echo "ERROR ${local_filename}: cannot create quarantine directory ${quarantine_dir}" >&2
            return 1
        fi
        local quarantine_target="${quarantine_dir}/${local_filename}.$(date '+%Y%m%dT%H%M%S').${BASHPID}.oversize"
        if ! mv -- "${target}" "${quarantine_target}"; then
            echo "ERROR ${local_filename}: cannot quarantine oversized local file" >&2
            return 1
        fi
        echo "QUARANTINE ${local_filename}: local size ${current_size} exceeds expected size ${expected_size}; moved to ${quarantine_target}" >&2
        current_size=0
    fi

    echo "GET  ${local_filename} (${current_size}/${expected_size} bytes)"
    if ! curl --fail --location --silent --show-error \
        --retry 20 --retry-all-errors --retry-delay 5 --connect-timeout 30 \
        --speed-limit 1024 --speed-time 120 \
        --continue-at - --output "${target}" "${base_url}/${encoded_filename}"; then
        echo "ERROR ${local_filename}: transfer failed" >&2
        return 1
    fi

    final_size="$(stat -c '%s' "${target}")"
    if [[ "${final_size}" != "${expected_size}" ]]; then
        echo "ERROR ${local_filename}: downloaded size ${final_size}, expected ${expected_size}" >&2
        return 1
    fi
    echo "DONE ${local_filename}"
}

export dataset_dir base_url quarantine_dir
export -f download_one

echo "Manifest: ${manifest_file}"
echo "Destination: ${dataset_dir}"
echo "Parallel transfers: ${parallel_transfers}"

# Fetch annotations first so every already-complete multi-GB WSI becomes a
# usable segmentation pair as early as possible.
jq -j '
    sort_by([
        (if (.path | endswith(".geojson")) then 0
         elif (.path | endswith(".isyntax")) then 1
         elif (.path | endswith("_mask.png")) then 2
         else 3 end),
        .path
    ])
    | .[]
    | (.path | split("/")[-1]) as $filename
    | "\(.size)\u0000\($filename)\u0000\($filename | @uri)\u0000"
' "${manifest_file}" \
    | xargs -0 -P "${parallel_transfers}" -n 3 \
        bash -c 'download_one "$1" "$2" "$3"' _

download_status=$?
if (( download_status == 0 )); then
    echo "All manifest files have been downloaded and size-verified."
else
    echo "One or more files failed; rerun this script to resume them." >&2
fi
exit "${download_status}"
