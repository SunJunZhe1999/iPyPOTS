#!/bin/bash

source ~/.bashrc
hostname

# 1) Expose only the GPUs you’re permitted to use
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 2) Choose a subset of the visible GPUs
USE_GPUS="6"    # adjust as needed, e.g. "0,1" or "" for CPU

# 3) Build DEVICE array
DEVICE=()
BACKEND="cpu"
if command -v nvidia-smi &>/dev/null && [[ -n "$CUDA_VISIBLE_DEVICES" ]] && [[ -n "$USE_GPUS" ]]; then
  IFS=',' read -ra SEL <<< "$USE_GPUS"
  for lg in "${SEL[@]}"; do
    DEVICE+=("cuda:${lg}")
  done
  BACKEND="cuda"
fi
if [[ ${#DEVICE[@]} -eq 0 ]]; then
  DEVICE=("cpu")
  BACKEND="cpu"
fi

echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Using devices: ${DEVICE[*]}"

# 4) Activate Conda environment
ENV_NAME="pypots"
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
  if ! command -v conda &>/dev/null; then
    echo "❌ Conda not found."
    exit 1
  fi
  source activate "$ENV_NAME" || { echo "❌ Could not activate Conda env '$ENV_NAME'."; exit 1; }
fi

# ========= EXPERIMENT CONFIGS =========
MODEL="gpt4ts"
LLM_MODELS=("openai-community/gpt2")
DATASETS=("physionet_2012" "beijing_multisite_air_quality" "italy_air_quality" "pems_traffic" "solar_alabama")
MISSING_RATES=("0.1" "0.2" "0.3" "0.4" "0.5")
BATCH_SIZES=("32")
D_MODELS=("64")
D_FFNS=("64")
N_HEADS=("2")
N_LAYERS=("1")
ENABLE_PROFILING_VALUES=("true")

# Paths
ROOT_OUT="output/imputation/${DEVICE}"
PROFILING_PATH="${ROOT_OUT}/profiling"
PROFILING_PREFIX="backbone_gpt4ts"
mkdir -p "${ROOT_OUT}" "${PROFILING_PATH}"

# Fixed config
EPOCH=15
PATIENCE=5

# Session log
SESSION_TAG="$(date +%Y%m%dT%H%M%S)"
SESSION_LOG="${ROOT_OUT}/session_${MODEL}_${SESSION_TAG}.log"
echo "Session log: ${SESSION_LOG}" | tee -a "${SESSION_LOG}"

# ---- helper: map dataset -> dims ----
set_dims_for_dataset () {
  local ds="$1"
  case "$ds" in
    "physionet_2012") N_STEPS=48;  N_FEATURES=35 ;;
    "etth1"|"etth2"|"ettm1"|"ettm2") N_STEPS=96;  N_FEATURES=7 ;;
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

# ========= Main sweep =========
for DATASET in "${DATASETS[@]}"; do
  if ! set_dims_for_dataset "$DATASET"; then
    exit 1
  fi

  for MISSING_RATE in "${MISSING_RATES[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
      for D_MODEL in "${D_MODELS[@]}"; do
        for D_FFN in "${D_FFNS[@]}"; do
          for N_HEAD in "${N_HEADS[@]}"; do
            for N_LAYER in "${N_LAYERS[@]}"; do
              for LLM_MODEL in "${LLM_MODELS[@]}"; do
                for ENABLE_PROFILING in "${ENABLE_PROFILING_VALUES[@]}"; do

                  SAVE_DIR="${ROOT_OUT}/${MODEL}/${DATASET}/epoch${EPOCH}/mr${MISSING_RATE}_bs${BATCH_SIZE}_dm${D_MODEL}_ffn${D_FFN}_h${N_HEAD}_ly${N_LAYER}_prof${ENABLE_PROFILING}"
                  LOG_DIR="${SAVE_DIR}/logs"
                  mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

                  RUN_TAG="mr${MISSING_RATE}_bs${BATCH_SIZE}_dm${D_MODEL}_ffn${D_FFN}_h${N_HEAD}_ly${N_LAYER}_prof${ENABLE_PROFILING}"
                  RUN_LOG="${LOG_DIR}/run_${RUN_TAG}.log"
                  DONE_MARK="${SAVE_DIR}/.done"

                  if [[ -f "${DONE_MARK}" ]]; then
                    echo "⏭️ Skip (done): ${DATASET} | ${RUN_TAG}" | tee -a "${SESSION_LOG}"
                    continue
                  fi

                  echo "🔥 RUN: ${MODEL} | ${DATASET} | ${RUN_TAG} (DEVICE=${DEVICE[*]}, N_STEPS=${N_STEPS}, N_FEATURES=${N_FEATURES})" | tee -a "${SESSION_LOG}"

                  cmd=(python main.py
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
                    --llm_model_type "${LLM_MODEL}"
                    --batch_size "${BATCH_SIZE}"
                    --enable_profiling "${ENABLE_PROFILING}"
                    --profiling_path "${PROFILING_PATH}"
                    --profiling_prefix "${PROFILING_PREFIX}"
                  )

                  {
                    echo "===== CMD @ $(date -Is) =====" >> "${RUN_LOG}"
                    printf '%q ' "${cmd[@]}" >> "${RUN_LOG}"; echo >> "${RUN_LOG}"
                    echo "================================" >> "${RUN_LOG}"

                    if command -v srun &>/dev/null; then
                      srun --quiet --unbuffered "${cmd[@]}" |& tee -a "${RUN_LOG}"
                    else
                      "${cmd[@]}" |& tee -a "${RUN_LOG}"
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
done

echo "✅ All ${MODEL} runs completed at $(date -Is)."