#!/bin/bash

hostname

timestamp() {
  date "+%Y-%m-%dT%H:%M:%S%z"
}

PYTHON_BIN="${PYTHON_BIN:-}"
ENV_NAME="pypots"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
      if ! command -v conda &>/dev/null; then echo "❌ Conda not found and .venv/bin/python is unavailable."; exit 1; fi
      source activate "$ENV_NAME" || { echo "❌ Could not activate Conda env '$ENV_NAME'."; exit 1; }
    fi
    PYTHON_BIN="python"
  fi
fi
echo "Using Python: ${PYTHON_BIN}"

# ==================== GPU SELECTION (Template Style) ====================
# 1) Expose only GPUs you're allowed to see (physical indices)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 2) Choose a subset of the visible GPUs by *logical* indices ("" → CPU)
USE_GPUS="${USE_GPUS:-1}"        # e.g., "0" or "0,1"; set "" to force CPU
USE_MPS="${USE_MPS:-1}"          # set "" to force CPU on Apple Silicon

# 3) Build DEVICE list and BACKEND tag
DEVICE=()
BACKEND="cpu"
if command -v nvidia-smi &>/dev/null && [[ -n "$CUDA_VISIBLE_DEVICES" ]] && [[ -n "$USE_GPUS" ]]; then
  IFS=',' read -ra SEL <<< "$USE_GPUS"
  for lg in "${SEL[@]}"; do DEVICE+=("cuda:${lg}"); done
  BACKEND="cuda"
fi
if [[ ${#DEVICE[@]} -eq 0 ]]; then
  if [[ -n "$USE_MPS" ]] && "${PYTHON_BIN}" -c "import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)" >/dev/null 2>&1; then
    DEVICE=("mps")
    BACKEND="mps"
  else
    DEVICE=("cpu")
    BACKEND="cpu"
  fi
fi

echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Using devices: ${DEVICE[*]}"

# ==================== EXPERIMENT CONFIGS ====================
MODEL="uniformtsv"

read -ra DATASETS <<< "${DATASETS:-physionet_2012 appliances_energy household_power citylearn_zone5 opsd_germany}"
read -ra MISSING_RATES <<< "${MISSING_RATES:-0.1 0.3 0.5}"
read -ra BATCH_SIZES <<< "${BATCH_SIZES:-32}"
read -ra D_MODELS <<< "${D_MODELS:-64}"
read -ra D_FFNS <<< "${D_FFNS:-128}"
read -ra N_HEADS <<< "${N_HEADS:-4}"
read -ra N_LAYERS <<< "${N_LAYERS:-2}"

ENABLE_PROFILING_VALUES=("true")

# Paths (device-aware)
ROOT_OUT="output/imputation/${BACKEND}"
PROFILING_PATH="${ROOT_OUT}/profiling"
PROFILING_PREFIX="backbone_uniformtsv"
mkdir -p "${ROOT_OUT}" "${PROFILING_PATH}"

# Fixed
EPOCH="${EPOCH:-50}"
PATIENCE="${PATIENCE:-10}"
PATCH_SIZE="${PATCH_SIZE:-12}"
PATCH_STRIDE="${PATCH_STRIDE:-12}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
MAX_SAMPLES="${MAX_SAMPLES:-20000}"
WINDOW_STRIDE="${WINDOW_STRIDE:-4}"

# Session log
SESSION_TAG="$(date +%Y%m%dT%H%M%S)"
SESSION_LOG="${ROOT_OUT}/session_${MODEL}_${SESSION_TAG}.log"
echo "Session log: ${SESSION_LOG}" | tee -a "${SESSION_LOG}"

# ==================== Helper: dataset dims ====================
set_dims_for_dataset () {
  local ds="$1"
  case "$ds" in
    "physionet_2012") N_STEPS=48;  N_FEATURES=35 ;;
    "appliances_energy") N_STEPS=72; N_FEATURES=32 ;;
    "household_power") N_STEPS=168; N_FEATURES=11 ;;
    "citylearn_zone5") N_STEPS=168; N_FEATURES=20 ;;
    "opsd_germany") N_STEPS=168; N_FEATURES=10 ;;
    "etth1"|"etth2"|"ettm1"|"ettm2") N_STEPS=96;  N_FEATURES=7  ;;
    "air_quality"|"beijing_multisite_air_quality"|"italy_air_quality") N_STEPS=48; N_FEATURES=36 ;;
    "pems_traffic")   N_STEPS=96;  N_FEATURES=228 ;;
    "solar_alabama")  N_STEPS=96;  N_FEATURES=137 ;;
    "electricity_load_diagrams") N_STEPS=168; N_FEATURES=370 ;;
    "ucr_uea_MelbournePedestrian") N_STEPS=24; N_FEATURES=1 ;;
    "ucr_uea_ECG200") N_STEPS=96; N_FEATURES=1 ;;
    "ucr_uea_LargeKitchenAppliances") N_STEPS=720; N_FEATURES=3 ;;
    "ucr_uea_PowerCons") N_STEPS=144; N_FEATURES=1 ;;
    "ucr_uea_ItalyPowerDemand") N_STEPS=24; N_FEATURES=1 ;;
    *) echo "❌ Unknown dataset: $ds" | tee -a "${SESSION_LOG}"; return 1 ;;
  esac
  return 0
}

# ==================== Main sweep ====================
for DATASET in "${DATASETS[@]}"; do
  if ! set_dims_for_dataset "$DATASET"; then exit 1; fi

  for MISSING_RATE in "${MISSING_RATES[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
      for D_MODEL in "${D_MODELS[@]}"; do
        for D_FFN in "${D_FFNS[@]}"; do
          for N_HEAD in "${N_HEADS[@]}"; do
            for N_LAYER in "${N_LAYERS[@]}"; do
              for ENABLE_PROFILING in "${ENABLE_PROFILING_VALUES[@]}"; do

                # SAITS requires d_k and d_v (assume divisible)
                D_K=$(( D_MODEL / N_HEAD ))
                D_V=$D_K

                SAVE_DIR="${ROOT_OUT}/${MODEL}/${DATASET}/epoch${EPOCH}/mr${MISSING_RATE}_bs${BATCH_SIZE}_dm${D_MODEL}_ffn${D_FFN}_h${N_HEAD}_ly${N_LAYER}_prof${ENABLE_PROFILING}"
                LOG_DIR="${SAVE_DIR}/logs"
                mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

                RUN_TAG="mr${MISSING_RATE}_bs${BATCH_SIZE}_dm${D_MODEL}_ffn${D_FFN}_h${N_HEAD}_ly${N_LAYER}_prof${ENABLE_PROFILING}"
                RUN_LOG="${LOG_DIR}/run_${RUN_TAG}.log"
                DONE_MARK="${SAVE_DIR}/.done"

                if [[ -f "${DONE_MARK}" ]]; then
                  echo "⏭️  Skip (done): ${DATASET} | ${RUN_TAG}" | tee -a "${SESSION_LOG}"
                  continue
                fi

                echo "🔥 RUN: ${MODEL} | ${DATASET} | ${RUN_TAG} (DEVICE=${DEVICE[*]}, N_STEPS=${N_STEPS}, N_FEATURES=${N_FEATURES}, d_k=${D_K}, d_v=${D_V})" | tee -a "${SESSION_LOG}"

                cmd=("${PYTHON_BIN}" main.py
                  --model "${MODEL}"
                  --dataset_name "${DATASET}"
                  --epochs "${EPOCH}"
                  --patience "${PATIENCE}"
                  --missing_rate "${MISSING_RATE}"
                  --saving_path "${SAVE_DIR}"
                  --device "${DEVICE[*]}"
                  --n_steps "${N_STEPS}"
                  --n_features "${N_FEATURES}"
                  --n_layers "${N_LAYER}"
                  --d_model "${D_MODEL}"
                  --d_ffn "${D_FFN}"
                  --n_heads "${N_HEAD}"
                  --d_k "${D_K}"
                  --d_v "${D_V}"
                  --batch_size "${BATCH_SIZE}"
                  --patch_size "${PATCH_SIZE}"
                  --patch_stride "${PATCH_STRIDE}"
                  --learning_rate "${LEARNING_RATE}"
                  --weight_decay "${WEIGHT_DECAY}"
                  --max_samples "${MAX_SAMPLES}"
                  --window_stride "${WINDOW_STRIDE}"
                  --enable_profiling "${ENABLE_PROFILING}"
                  --profiling_path "${PROFILING_PATH}"
                  --profiling_prefix "${PROFILING_PREFIX}"
                )

                {
                  echo "===== CMD @ $(timestamp) =====" >> "${RUN_LOG}"
                  printf '%q ' "${cmd[@]}" >> "${RUN_LOG}"; echo >> "${RUN_LOG}"
                  echo "================================" >> "${RUN_LOG}"

                  if command -v srun &>/dev/null; then
                    srun --quiet --unbuffered "${cmd[@]}" 2>&1 | tee -a "${RUN_LOG}"
                  else
                    "${cmd[@]}" 2>&1 | tee -a "${RUN_LOG}"
                  fi
                }

                status=${PIPESTATUS[0]}
                if [[ $status -eq 0 ]]; then
                  touch "${DONE_MARK}"
                  echo "✅ DONE: ${DATASET} | ${RUN_TAG}" | tee -a "${SESSION_LOG}"
                else
                  echo "❌ FAIL(${status}): ${DATASET} | ${RUN_TAG}" | tee -a "${SESSION_LOG}"
                fi

              done
            done
          done
        done
      done
    done
  done
done

echo "✅ All ${MODEL} runs completed at $(timestamp)."
