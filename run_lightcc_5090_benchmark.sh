#!/bin/bash

set -u

cd "$(dirname "$0")"

timestamp() {
  date "+%Y-%m-%dT%H:%M:%S%z"
}

ROOT_OUT="${ROOT_OUT:-output/imputation/cuda/energy_benchmark_5090}"
LOG_DIR="${ROOT_OUT}/continuous_logs"
mkdir -p "${LOG_DIR}"

if [[ -x "/root/miniconda3/envs/uniformtsv/bin/python" ]]; then
  DEFAULT_PYTHON="/root/miniconda3/envs/uniformtsv/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  DEFAULT_PYTHON=".venv/bin/python"
else
  DEFAULT_PYTHON="python"
fi

export PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export USE_GPUS="${USE_GPUS:-0}"
export USE_MPS="${USE_MPS:-}"
export ROOT_OUT

export MODELS="${MODELS:-timemixerpp uniformtsv moment tslanet saits tefn}"
export DATASETS="${DATASETS:-etth1 etth2 ettm1 ettm2 appliances_energy household_power citylearn_zone5 opsd_germany physionet_2012 solar eld}"
export MISSING_RATES="${MISSING_RATES:-0.1 0.3 0.5 0.7}"
export SEEDS="${SEEDS:-42 7}"
export CTCAR_DATASETS="${CTCAR_DATASETS:-physionet_2012 appliances_energy household_power citylearn_zone5 opsd_germany etth1 etth2 ettm1 ettm2 solar eld}"
export CTCAR_MISSING_RATES="${CTCAR_MISSING_RATES:-0.1 0.3 0.5 0.7}"

export TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-7200}"
export EPOCH="${EPOCH:-80}"
export PATIENCE="${PATIENCE:-12}"
export MAX_SAMPLES="${MAX_SAMPLES:-4000}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export D_MODEL="${D_MODEL:-128}"
export D_FFN="${D_FFN:-256}"
export N_HEAD="${N_HEAD:-8}"
export N_LAYER="${N_LAYER:-2}"
export N_FOD="${N_FOD:-4}"
export RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-900}"
export WINDOW_STRIDE="${WINDOW_STRIDE:-2}"
export PATCH_SIZE="${PATCH_SIZE:-12}"
export PATCH_STRIDE="${PATCH_STRIDE:-12}"
export TRANSFORMER_BACKBONE="${TRANSFORMER_BACKBONE:-PatchTST}"
export TRANSFORMER_TYPE="${TRANSFORMER_TYPE:-encoder_only}"
export FINETUNING_MODE="${FINETUNING_MODE:-end-to-end}"

echo "LightCC RTX 5090 benchmark started at $(timestamp)."
echo "Output root: ${ROOT_OUT}"
echo "Python: ${PYTHON_BIN}"
echo "Models: ${MODELS}"
echo "Datasets: ${DATASETS}"
echo "Missing rates: ${MISSING_RATES}"
echo "Seeds: ${SEEDS}"

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
  --max-samples 2000 \
  --window-stride 4 \
  --seed 42 \
  --out "${ROOT_OUT}/ctcar_features.csv"

"${PYTHON_BIN}" script/ctcar_routing_analysis.py \
  --metrics "${ROOT_OUT}/metrics_summary.csv" \
  --ctcar "${ROOT_OUT}/ctcar_features.csv" \
  --out-dir "${ROOT_OUT}/ctcar_routing"

"${PYTHON_BIN}" script/energy_benchmark_200_report.py \
  --root "${ROOT_OUT}" \
  --out "${ROOT_OUT}/energy_benchmark_5090_report.md"

echo "LightCC RTX 5090 benchmark finished at $(timestamp)."
exit "${benchmark_status}"
