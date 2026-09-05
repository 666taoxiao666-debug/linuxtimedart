#!/usr/bin/env bash

# Shared SDWPF log layout. Entry scripts must call sdwpf_log_init after their
# parameters and RUN_ID have been resolved. Explicit SDWPF_LOG_FILE and
# SDWPF_LOG_DIR always take precedence over automatic allocation.

if [[ -n "${_SDWPF_LOG_LIBRARY_LOADED:-}" ]]; then
    return 0
fi
_SDWPF_LOG_LIBRARY_LOADED=1

_SDWPF_LOG_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_SDWPF_LOG_OWNS_DIR=0
_SDWPF_LOG_CAPTURE_ACTIVE=0
_SDWPF_LOG_TEE_PID=""

sdwpf_log_init() {
    local task="$1"
    local parameters="$2"
    local default_log_name="$3"
    local run_id="${4:-}"
    local requested_file="${SDWPF_LOG_FILE:-}"
    local requested_dir="${SDWPF_LOG_DIR:-}"

    if [[ -n "${requested_file}" ]]; then
        SDWPF_LOG_FILE="${requested_file}"
        if [[ -n "${requested_dir}" ]]; then
            SDWPF_LOG_DIR="${requested_dir}"
        else
            SDWPF_LOG_DIR="$(dirname "${requested_file}")"
        fi
    elif [[ -n "${requested_dir}" ]]; then
        SDWPF_LOG_DIR="${requested_dir}"
        SDWPF_LOG_FILE="${SDWPF_LOG_DIR}/${default_log_name}"
    else
        SDWPF_LOG_DIR="$(
            cd "${_SDWPF_LOG_REPO_ROOT}"
            python -m utils.sdwpf_logging allocate \
                --task "${task}" \
                --parameters "${parameters}" \
                --run-id "${run_id}"
        )"
        SDWPF_LOG_FILE="${SDWPF_LOG_DIR}/${default_log_name}"
        _SDWPF_LOG_OWNS_DIR=1
    fi

    mkdir -p "${SDWPF_LOG_DIR}"
    export SDWPF_LOG_DIR SDWPF_LOG_FILE
    echo "[LOG] Directory: ${SDWPF_LOG_DIR}"
    echo "[LOG] File: ${SDWPF_LOG_FILE}"
}

sdwpf_log_sidecar() {
    local extension="$1"
    local directory filename stem
    directory="$(dirname -- "${SDWPF_LOG_FILE}")"
    filename="$(basename -- "${SDWPF_LOG_FILE}")"
    if [[ "${filename}" == ?*.log ]]; then
        stem="${filename%.log}"
    else
        stem="${filename}"
    fi
    printf '%s/%s.%s\n' "${directory}" "${stem}" "${extension}"
}

sdwpf_log_capture() {
    if [[ "${_SDWPF_LOG_CAPTURE_ACTIVE}" == "1" ]]; then
        echo "sdwpf_log_capture may only be called once" >&2
        return 2
    fi
    exec 3>&1 4>&2
    exec > >(tee -a "${SDWPF_LOG_FILE}" >&3) 2>&1
    _SDWPF_LOG_TEE_PID="$!"
    _SDWPF_LOG_CAPTURE_ACTIVE=1
}

sdwpf_log_on_exit() {
    local exit_code="$1"
    local tee_code=0
    trap - EXIT
    if [[ "${_SDWPF_LOG_CAPTURE_ACTIVE}" == "1" ]]; then
        # Restoring the original descriptors closes the process-substitution
        # pipe. Waiting makes a tee/write failure part of the script result.
        exec 1>&3 2>&4
        if wait "${_SDWPF_LOG_TEE_PID}"; then
            tee_code=0
        else
            tee_code="$?"
        fi
        exec 3>&- 4>&-
        if [[ "${exit_code}" == "0" && "${tee_code}" != "0" ]]; then
            exit_code="${tee_code}"
            echo "[LOG] ERROR: tee failed for ${SDWPF_LOG_FILE}" >&2
        fi
    fi
    if [[ "${_SDWPF_LOG_OWNS_DIR}" == "1" ]]; then
        local finalize_code=0
        if (
            cd "${_SDWPF_LOG_REPO_ROOT}"
            python -m utils.sdwpf_logging finish \
                --log-dir "${SDWPF_LOG_DIR}" \
                --exit-code "${exit_code}"
        ) >/dev/null; then
            finalize_code=0
        else
            finalize_code="$?"
            echo "[LOG] WARNING: could not finalize status for ${SDWPF_LOG_DIR}" >&2
            if [[ "${exit_code}" == "0" ]]; then
                exit_code="${finalize_code}"
            fi
        fi
    fi
    exit "${exit_code}"
}

sdwpf_log_install_exit_trap() {
    trap 'sdwpf_log_on_exit "$?"' EXIT
}
