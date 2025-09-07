#!/bin/bash

source ~/.bashrc
hostname

# ===================== GPU SELECTION (Template style) =====================
# Expose only the physical GPUs you’re allowed to use
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Pick a subset of the visible GPUs by their logical indices (comma-separated).
# Leave empty to force CPU.
USE_GPUS="3"

DEVICE=()
BACKEND="cpu"
if command -v nvidia-smi &>/dev/null && [[ -n "$CUDA_VISIBLE_DEVICES" ]] && [[ -n "$USE_GPUS" ]]; then
  IFS=',' read -ra SEL <<< "$USE_GPUS"
  for lg in "${SEL[@]}"; do
    DEVICE+=("cuda:${lg}")
  done
  if [[ ${#DEVICE[@]} -gt 0 ]]; then BACKEND="cuda"; fi
fi
if [[ ${#DEVICE[@]} -eq 0 ]]; then DEVICE=("cpu"); BACKEND="cpu"; fi

echo "Visible (physical) GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Using (logical) devices: ${DEVICE[*]}"

# ===================== CONDA ENV =====================
ENV_NAME="pypots"
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
  if ! command -v conda &>/dev/null; then echo "❌ Conda not found."; exit 1; fi
  source activate "$ENV_NAME" || { echo "❌ Could not activate Conda env '$ENV_NAME'."; exit 1; }
fi

# ===================== EXPERIMENT CONFIGS =====================
MODEL="tslanet"

DATASETS=("physionet_2012" "beijing_multisite_air_quality" "italy_air_quality" "pems_traffic" "solar_alabama")
MISSING_RATES=("0.1" "0.2" "0.3" "0.4" "0.5")
BATCH_SIZES=("32")
PATCH_SIZES=("24")
D_EMBEDDINGS=("128")
N_LAYERS_LIST=("2" "3" "4")

# Paths (device-aware: cuda/cpu)
ROOT_OUT="output/imputation/${BACKEND}"
mkdir -p "${ROOT_OUT}"

# Fixed train config
EPOCH=15
PATIENCE=5

# Session log
SESSION_TAG="$(date +%Y%m%dT%H%M%S)"
SESSION_LOG="${ROOT_OUT}/session_${MODEL}_${SESSION_TAG}.log"
echo "Session log: ${SESSION_LOG}" | tee -a "${SESSION_LOG}"

# ===================== dataset → dims =====================
set_dims_for_dataset () {
  local ds="$1"
  case "$ds" in
    "physionet_2012") N_STEPS=48;  N_FEATURES=35 ;;
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

# ===================== main sweep =====================
for DATASET in "${DATASETS[@]}"; do
  if ! set_dims_for_dataset "$DATASET"; then exit 1; fi

  for MISSING_RATE in "${MISSING_RATES[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
      for PATCH_SIZE in "${PATCH_SIZES[@]}"; do
        for D_EMBEDDING in "${D_EMBEDDINGS[@]}"; do
          for N_LAYERS in "${N_LAYERS_LIST[@]}"; do

            SAVE_DIR="${ROOT_OUT}/${MODEL}/${DATASET}/epoch${EPOCH}/mr${MISSING_RATE}_bs${BATCH_SIZE}_ps${PATCH_SIZE}_de${D_EMBEDDING}_ly${N_LAYERS}"
            LOG_DIR="${SAVE_DIR}/logs"
            mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

            RUN_TAG="mr${MISSING_RATE}_bs${BATCH_SIZE}_ps${PATCH_SIZE}_de${D_EMBEDDING}_ly${N_LAYERS}"
            RUN_LOG="${LOG_DIR}/run_${RUN_TAG}.log"
            DONE_MARK="${SAVE_DIR}/.done"

            if [[ -f "${DONE_MARK}" ]]; then
              echo "⏭️  Skip (done): ${DATASET} | ${RUN_TAG}" | tee -a "${SESSION_LOG}"
              continue
            fi

            echo "🔥 RUN: ${MODEL} | ${DATASET} | ${RUN_TAG} (DEVICE=${DEVICE[*]}, N_STEPS=${N_STEPS}, N_FEATURES=${N_FEATURES})" | tee -a "${SESSION_LOG}"

            # Command array (no eval)
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
              --batch_size "${BATCH_SIZE}"
              --patch_size "${PATCH_SIZE}"
              --d_embedding "${D_EMBEDDING}"
              --n_layers "${N_LAYERS}"
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

echo "✅ All ${MODEL} runs completed at $(date -Is)."