#!/bin/bash

set -u

cd "$(dirname "$0")"

timestamp() {
  date "+%Y-%m-%dT%H:%M:%S%z"
}

ROOT_OUT="${ROOT_OUT:-output/imputation/mps/energy_benchmark_200}"
LOG_DIR="${ROOT_OUT}/continuous_logs"
mkdir -p "${LOG_DIR}"

echo "Local continuation benchmark started at $(timestamp)."
echo "Output root: ${ROOT_OUT}"

export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export ROOT_OUT
export MODELS="${MODELS:-mean median locf saits tefn uniformtsv moment}"
export DATASETS="${DATASETS:-physionet_2012 appliances_energy household_power citylearn_zone5 opsd_germany etth1 etth2 ettm1 ettm2 solar eld}"
export MISSING_RATES="${MISSING_RATES:-0.1 0.3 0.5 0.7}"
export CTCAR_DATASETS="${CTCAR_DATASETS:-physionet_2012 appliances_energy household_power citylearn_zone5 opsd_germany etth1 etth2 ettm1 ettm2 solar eld}"
export CTCAR_MISSING_RATES="${CTCAR_MISSING_RATES:-0.1 0.3 0.5 0.7}"
export SEEDS="${SEEDS:-7 123 456}"
export TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-21600}"
export EPOCH="${EPOCH:-50}"
export PATIENCE="${PATIENCE:-10}"
export MAX_SAMPLES="${MAX_SAMPLES:-800}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export N_FOD="${N_FOD:-2}"
export RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-240}"
export TRANSFORMER_BACKBONE="${TRANSFORMER_BACKBONE:-PatchTST}"
export TRANSFORMER_TYPE="${TRANSFORMER_TYPE:-encoder_only}"
export FINETUNING_MODE="${FINETUNING_MODE:-end-to-end}"

bash run_energy_benchmark.sh
benchmark_status=$?
echo "Benchmark phase exited with status ${benchmark_status} at $(timestamp)."

"${PYTHON_BIN}" script/collect_metrics.py \
  --root "${ROOT_OUT}" \
  --out "${ROOT_OUT}/metrics_summary.csv"

"${PYTHON_BIN}" script/model_routing_analysis.py \
  --metrics "${ROOT_OUT}/metrics_summary.csv" \
  --out-dir "${ROOT_OUT}/routing"

"${PYTHON_BIN}" script/ctcar_features.py \
  --datasets "${CTCAR_DATASETS}" \
  --missing-rates "${CTCAR_MISSING_RATES}" \
  --max-samples 1000 \
  --window-stride 8 \
  --seed 42 \
  --out "${ROOT_OUT}/ctcar_features.csv"

"${PYTHON_BIN}" script/ctcar_routing_analysis.py \
  --metrics "${ROOT_OUT}/metrics_summary.csv" \
  --ctcar "${ROOT_OUT}/ctcar_features.csv" \
  --out-dir "${ROOT_OUT}/ctcar_routing"

"${PYTHON_BIN}" script/energy_benchmark_200_report.py \
  --root "${ROOT_OUT}" \
  --out script/energy_benchmark_200_report.md

echo "Local continuation benchmark finished at $(timestamp)."
exit "${benchmark_status}"
