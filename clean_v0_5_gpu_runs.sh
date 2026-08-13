#!/usr/bin/env bash
# Delete generated V0.5 GPU training runs while preserving committed validation reports.

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
TARGET_DIR="${SCRIPT_DIR}/runs/v0_5/gpu"

usage() {
    printf '%s\n' \
        "Usage: $0 [--yes]" \
        "" \
        "Deletes:  all contents under ${TARGET_DIR}" \
        "Keeps:    ${SCRIPT_DIR}/gpu_validation/v0_5/results" \
        "" \
        "Use --yes for non-interactive one-command cleanup."
}

ASSUME_YES=false
case "${1:-}" in
    "") ;;
    --yes|-y) ASSUME_YES=true ;;
    --help|-h)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

if [[ ! -d "${SCRIPT_DIR}/.git" ]] || [[ ! -f "${SCRIPT_DIR}/pyproject.toml" ]]; then
    printf 'Refusing cleanup: script is not inside the expected repository root: %s\n' \
        "${SCRIPT_DIR}" >&2
    exit 1
fi

EXPECTED_DIR="${SCRIPT_DIR}/runs/v0_5/gpu"
if [[ "${TARGET_DIR}" != "${EXPECTED_DIR}" || "${TARGET_DIR}" == "/" ]]; then
    printf 'Refusing cleanup: unsafe target path: %s\n' "${TARGET_DIR}" >&2
    exit 1
fi

if [[ ! -e "${TARGET_DIR}" ]]; then
    printf 'Nothing to delete: %s does not exist.\n' "${TARGET_DIR}"
    exit 0
fi

if [[ -L "${TARGET_DIR}" ]]; then
    printf 'Refusing cleanup: target directory must not be a symbolic link: %s\n' \
        "${TARGET_DIR}" >&2
    exit 1
fi

printf 'Training output to delete:\n  %s\n' "${TARGET_DIR}"
printf 'Committed validation reports will be kept:\n  %s\n' \
    "${SCRIPT_DIR}/gpu_validation/v0_5/results"
du -sh -- "${TARGET_DIR}" 2>/dev/null || true

if [[ "${ASSUME_YES}" != true ]]; then
    printf 'Type DELETE to continue: '
    read -r CONFIRMATION
    if [[ "${CONFIRMATION}" != "DELETE" ]]; then
        printf 'Cleanup cancelled.\n'
        exit 0
    fi
fi

find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
printf 'Deleted all V0.5 GPU training output under: %s\n' "${TARGET_DIR}"
printf 'Kept the empty output directory for the next validation run.\n'
printf 'This deletion is not recoverable from Git because runs/ is ignored.\n'
