#!/bin/bash

set -u

ROOT="/Users/sun/Documents/UniFormTSV/output/imputation/cuda/energy_benchmark_3090_publication"
KEY="/Users/sun/Documents/UniFormTSV/.lightcc_codex_key"
REMOTE="root@swiftlink.lightcc.cloud"
PORT="56656"
REMOTE_ROOT="/root/lightcc-tmp/UniFormTSV/output/imputation/cuda/energy_benchmark_3090_publication"
LOG_DIR="${ROOT}/continuous_logs"
LOG_FILE="${LOG_DIR}/local_pull_weights.log"
LOCK_DIR="${LOG_DIR}/local_pull_weights.lock"

mkdir -p "${LOG_DIR}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  printf '%s another weight pull is still running\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" >> "${LOG_FILE}"
  exit 0
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

{
  printf '%s starting weight rsync pull\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
  rsync -az --partial --timeout=180 \
    -e "ssh -i ${KEY} -p ${PORT} -o StrictHostKeyChecking=no -o BatchMode=yes" \
    --include '*/' \
    --include '*.pypots' \
    --include '*.pt' \
    --include '*.pth' \
    --include '*.ckpt' \
    --exclude '*' \
    "${REMOTE}:${REMOTE_ROOT}/" \
    "${ROOT}/"
  status=$?
  printf '%s weight rsync exit=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${status}"
  exit "${status}"
} >> "${LOG_FILE}" 2>&1
