#!/bin/bash

set -u

SOURCE="/Users/sun/Library/UniFormTSV/backups/energy_benchmark_3090_publication"
TARGET="/Users/sun/Documents/UniFormTSV/output/imputation/cuda/energy_benchmark_3090_publication"
LOG_DIR="${TARGET}/continuous_logs"
LOG_FILE="${LOG_DIR}/hourly_project_sync.log"
LOCK_DIR="${LOG_DIR}/hourly_project_sync.lock"

mkdir -p "${LOG_DIR}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  printf '%s another project sync is still running\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" >> "${LOG_FILE}"
  exit 0
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

{
  printf '%s starting Library-to-project sync\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
  if [[ ! -d "${SOURCE}" ]]; then
    printf '%s source missing: %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${SOURCE}"
    exit 1
  fi
  rsync -az --partial "${SOURCE}/" "${TARGET}/"
  status=$?
  printf '%s sync exit=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${status}"
  exit "${status}"
} >> "${LOG_FILE}" 2>&1
