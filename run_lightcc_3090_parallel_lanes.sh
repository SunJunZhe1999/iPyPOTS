#!/bin/bash

set -u

cd "$(dirname "$0")"

timestamp() {
  date "+%Y-%m-%dT%H:%M:%S%z"
}

ROOT_OUT="${ROOT_OUT:-output/imputation/cuda/energy_benchmark_3090_publication}"
LOG_DIR="${ROOT_OUT}/continuous_logs"
LANE_DIR="${LOG_DIR}/parallel_lanes"
mkdir -p "${LANE_DIR}"

if [[ "${STOP_EXISTING:-0}" == "1" ]]; then
  if [[ -f "${LOG_DIR}/benchmark.pid" ]]; then
    while read -r pid; do
      [[ -n "${pid}" ]] || continue
      pkill -TERM -P "${pid}" 2>/dev/null || true
      kill -TERM "${pid}" 2>/dev/null || true
    done < "${LOG_DIR}/benchmark.pid"
    sleep 5
    while read -r pid; do
      [[ -n "${pid}" ]] || continue
      pkill -KILL -P "${pid}" 2>/dev/null || true
      kill -KILL "${pid}" 2>/dev/null || true
    done < "${LOG_DIR}/benchmark.pid"
  fi
  pkill -TERM -f "main.py --model" 2>/dev/null || true
  sleep 2
fi

export ROOT_OUT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export USE_GPUS="${USE_GPUS:-0}"
export USE_MPS="${USE_MPS:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"

export MODELS="${MODELS:-saits timemixerpp tefn uniformtsv moment tslanet gpt4ts mean median locf}"
export DATASETS="${DATASETS:-appliances_energy household_power citylearn_zone5 opsd_germany etth1 etth2 ettm1 ettm2 solar eld physionet_2012}"
export MISSING_RATES="${MISSING_RATES:-0.1 0.3 0.5 0.7}"
export TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-172800}"
export EPOCH="${EPOCH:-180}"
export PATIENCE="${PATIENCE:-30}"
export MAX_SAMPLES="${MAX_SAMPLES:-20000}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export D_MODEL="${D_MODEL:-256}"
export D_FFN="${D_FFN:-512}"
export N_HEAD="${N_HEAD:-8}"
export N_LAYER="${N_LAYER:-4}"
export N_FOD="${N_FOD:-8}"
export RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-3600}"
export WINDOW_STRIDE="${WINDOW_STRIDE:-2}"
export PATCH_SIZE="${PATCH_SIZE:-12}"
export PATCH_STRIDE="${PATCH_STRIDE:-12}"
export TRANSFORMER_BACKBONE="${TRANSFORMER_BACKBONE:-PatchTST}"
export TRANSFORMER_TYPE="${TRANSFORMER_TYPE:-encoder_only}"
export FINETUNING_MODE="${FINETUNING_MODE:-end-to-end}"

start_lane() {
  local lane="$1"
  local seeds="$2"
  local log_file="${LANE_DIR}/${lane}_$(date +%Y%m%dT%H%M%S).log"
  (
    export SEEDS="${seeds}"
    export LANE_ID="${lane}"
    echo "Lane ${lane} started at $(timestamp) with seeds: ${SEEDS}"
    bash run_lightcc_3090_publication_benchmark.sh
  ) >> "${log_file}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${LANE_DIR}/${lane}.pid"
  echo "${lane} ${pid} ${seeds} ${log_file}" >> "${LANE_DIR}/lanes.tsv"
  echo "${pid}"
}

: > "${LANE_DIR}/lanes.tsv"
: > "${LOG_DIR}/benchmark.pid"

echo "Parallel LightCC RTX 3090 lanes started at $(timestamp)." | tee -a "${LANE_DIR}/launcher.log"
LANE_SPECS="${LANE_SPECS:-lane_a:42,123;lane_b:456;lane_c:7;lane_d:999}"
IFS=';' read -ra LANE_ITEMS <<< "${LANE_SPECS}"
for spec in "${LANE_ITEMS[@]}"; do
  [[ -n "${spec}" ]] || continue
  lane="${spec%%:*}"
  seeds="${spec#*:}"
  seeds="${seeds//,/ }"
  pid="$(start_lane "${lane}" "${seeds}")"
  echo "${pid}" >> "${LOG_DIR}/benchmark.pid"
  echo "Started ${lane}: pid=${pid}, seeds=${seeds}" | tee -a "${LANE_DIR}/launcher.log"
done

echo "Active lane PIDs:" | tee -a "${LANE_DIR}/launcher.log"
cat "${LOG_DIR}/benchmark.pid" | tee -a "${LANE_DIR}/launcher.log"
