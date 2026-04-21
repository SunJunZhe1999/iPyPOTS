#!/bin/bash

set -u

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUT_DIR="${OUT_DIR:-output/paper_reproduction}"
SEEDS="${SEEDS:-42 123 456}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SCALE="${SCALE:-1.0}"
DEVICE="${DEVICE:-auto}"
SPLIT="${SPLIT:-random}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:--1.0}"
RESIDUAL_SEED="${RESIDUAL_SEED:-20260421}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" script/paper_reproduction.py \
  --cache-dir datasets \
  --out-dir "${OUT_DIR}" \
  --seeds "${SEEDS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --scale "${SCALE}" \
  --device "${DEVICE}" \
  --split "${SPLIT}" \
  --residual-scale "${RESIDUAL_SCALE}" \
  --residual-seed "${RESIDUAL_SEED}" \
  ${EXTRA_ARGS}
