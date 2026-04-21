#!/bin/bash

set -u

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

TOTAL_BUDGET_SECONDS="${TOTAL_BUDGET_SECONDS:-7200}"
SUMMARY_RESERVE_SECONDS="${SUMMARY_RESERVE_SECONDS:-600}"
BENCHMARK_BUDGET_SECONDS=$(( TOTAL_BUDGET_SECONDS - SUMMARY_RESERVE_SECONDS ))
if [[ "${BENCHMARK_BUDGET_SECONDS}" -lt 600 ]]; then
  BENCHMARK_BUDGET_SECONDS="${TOTAL_BUDGET_SECONDS}"
fi

BACKEND="$("${PYTHON_BIN}" - <<'PY'
import torch
if torch.cuda.is_available():
    print("cuda")
elif torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
)"
ROOT_OUT="${ROOT_OUT:-output/imputation/${BACKEND}/energy_benchmark_200}"
export ROOT_OUT

export PYTHON_BIN
export SEEDS="${SEEDS:-42 123 456}"
export EPOCH="${EPOCH:-50}"
export PATIENCE="${PATIENCE:-8}"
export MAX_SAMPLES="${MAX_SAMPLES:-1200}"
export WINDOW_STRIDE="${WINDOW_STRIDE:-8}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export D_MODEL="${D_MODEL:-64}"
export D_FFN="${D_FFN:-128}"
export N_LAYER="${N_LAYER:-2}"
export N_HEAD="${N_HEAD:-4}"
export PATCH_SIZE="${PATCH_SIZE:-12}"
export PATCH_STRIDE="${PATCH_STRIDE:-12}"
if [[ -z "${TRANSFORMER_BACKBONE:-}" ]]; then
  if [[ "${BACKEND}" == "cuda" ]]; then
    export TRANSFORMER_BACKBONE="t5-small"
    export TRANSFORMER_TYPE="${TRANSFORMER_TYPE:-encoder_decoder}"
    export FINETUNING_MODE="${FINETUNING_MODE:-linear-probing}"
  else
    export TRANSFORMER_BACKBONE="PatchTST"
    export TRANSFORMER_TYPE="${TRANSFORMER_TYPE:-encoder_only}"
    export FINETUNING_MODE="${FINETUNING_MODE:-end-to-end}"
  fi
else
  export TRANSFORMER_BACKBONE
  export TRANSFORMER_TYPE="${TRANSFORMER_TYPE:-encoder_decoder}"
  export FINETUNING_MODE="${FINETUNING_MODE:-linear-probing}"
fi
export LEARNING_RATE="${LEARNING_RATE:-0.001}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
export N_FOD="${N_FOD:-3}"
export RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-180}"

echo "Expanded C-TCAR benchmark root: ${ROOT_OUT}"
echo "Benchmark budget: ${BENCHMARK_BUDGET_SECONDS}s"
echo "Seeds: ${SEEDS}"
echo "MOMENT/UniFormTSV backbone: ${TRANSFORMER_BACKBONE} (${TRANSFORMER_TYPE}, ${FINETUNING_MODE})"

PHASE1_BUDGET_SECONDS="${PHASE1_BUDGET_SECONDS:-2700}"
PHASE2_BUDGET_SECONDS="${PHASE2_BUDGET_SECONDS:-$(( BENCHMARK_BUDGET_SECONDS - PHASE1_BUDGET_SECONDS ))}"
if [[ "${PHASE2_BUDGET_SECONDS}" -lt 600 ]]; then
  PHASE2_BUDGET_SECONDS=600
fi

run_phase () {
  local phase_name="$1"
  local phase_budget="$2"
  local phase_models="$3"
  local phase_datasets="$4"
  local phase_missing_rates="$5"

  export TIME_BUDGET_SECONDS="${phase_budget}"
  export MODELS="${phase_models}"
  export DATASETS="${phase_datasets}"
  export MISSING_RATES="${phase_missing_rates}"

  echo "===== ${phase_name} ====="
  echo "Budget: ${TIME_BUDGET_SECONDS}s"
  echo "Models: ${MODELS}"
  echo "Datasets: ${DATASETS}"
  echo "Missing rates: ${MISSING_RATES}"
  bash run_energy_benchmark.sh
}

run_phase \
  "Phase 1: broad model diversity on representative energy datasets" \
  "${PHASE1_BUDGET_SECONDS}" \
  "${PHASE1_MODELS:-mean median locf saits timemixerpp tefn uniformtsv moment tslanet gpt4ts}" \
  "${PHASE1_DATASETS:-appliances_energy opsd_germany etth1 ettm1 solar}" \
  "${PHASE1_MISSING_RATES:-0.1 0.3 0.5}"

run_phase \
  "Phase 2: expanded dataset and high-missingness sweep" \
  "${PHASE2_BUDGET_SECONDS}" \
  "${PHASE2_MODELS:-mean median locf saits timemixerpp tefn uniformtsv}" \
  "${PHASE2_DATASETS:-physionet_2012 appliances_energy household_power citylearn_zone5 opsd_germany etth1 etth2 ettm1 ettm2 solar eld}" \
  "${PHASE2_MISSING_RATES:-0.1 0.3 0.5 0.7}"

"${PYTHON_BIN}" script/collect_metrics.py \
  --root "${ROOT_OUT}" \
  --out "${ROOT_OUT}/metrics_summary.csv"

"${PYTHON_BIN}" script/model_routing_analysis.py \
  --metrics "${ROOT_OUT}/metrics_summary.csv" \
  --out-dir "${ROOT_OUT}/routing"

"${PYTHON_BIN}" script/ctcar_features.py \
  --datasets "${CTCAR_DATASETS:-physionet_2012 appliances_energy household_power citylearn_zone5 opsd_germany etth1 etth2 ettm1 ettm2 solar eld}" \
  --missing-rates "${CTCAR_MISSING_RATES:-0.1 0.3 0.5 0.7}" \
  --max-samples "${CTCAR_MAX_SAMPLES:-1200}" \
  --window-stride "${CTCAR_WINDOW_STRIDE:-${WINDOW_STRIDE}}" \
  --seed "${CTCAR_SEED:-42}" \
  --out "${ROOT_OUT}/ctcar_features.csv"

"${PYTHON_BIN}" script/ctcar_routing_analysis.py \
  --metrics "${ROOT_OUT}/metrics_summary.csv" \
  --ctcar "${ROOT_OUT}/ctcar_features.csv" \
  --out-dir "${ROOT_OUT}/ctcar_routing"

"${PYTHON_BIN}" script/energy_benchmark_200_report.py \
  --root "${ROOT_OUT}" \
  --metrics "${ROOT_OUT}/metrics_summary.csv" \
  --out "script/energy_benchmark_200_report.md"

echo "Expanded C-TCAR benchmark artifacts are ready under ${ROOT_OUT}"
