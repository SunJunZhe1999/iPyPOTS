#!/bin/bash

source ~/.bashrc
hostname

# ==================== GPU SELECTION (Template) ====================
# Expose only GPUs you're allowed to see (physical indices)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Choose a subset of the visible GPUs by *logical* indices ("" → CPU)
USE_GPUS="0,6"        # e.g., "0" or "0,1"; set "" to force CPU

# Build DEVICE list and BACKEND tag
DEVICE=()
BACKEND="cpu"
if command -v nvidia-smi &>/dev/null && [[ -n "$CUDA_VISIBLE_DEVICES" ]] && [[ -n "$USE_GPUS" ]]; then
  IFS=',' read -ra SEL <<< "$USE_GPUS"
  for lg in "${SEL[@]}"; do DEVICE+=("cuda:${lg}"); done
  BACKEND="cuda"
fi
if [[ ${#DEVICE[@]} -eq 0 ]]; then DEVICE=("cpu"); BACKEND="cpu"; fi

echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Using devices: ${DEVICE[*]}"

# ==================== ENV ====================
ENV_NAME="pypots"
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
  if ! command -v conda &>/dev/null; then echo "❌ Conda not found."; exit 1; fi
  source activate "$ENV_NAME" || { echo "❌ Could not activate Conda env '$ENV_NAME'."; exit 1; }
fi

# ==================== EXPERIMENT CONFIGS ====================
MODEL="timemixerpp"

DATASETS=("physionet_2012" "beijing_multisite_air_quality" "italy_air_quality" "pems_traffic" "solar_alabama")
MISSING_RATES=("0.1" "0.2" "0.3" "0.4" "0.5")
BATCH_SIZES=("32")

D_MODELS=("64")
D_FFNS=("64")
N_HEADS=("2")
N_LAYERS=("1")

TOP_KS=("5")
N_KERNELS=("3")
CHANNEL_MIXING_VALUES=("true" "false")
CHANNEL_INDEPENDENCE_VALUES=("true" "false")
DOWNSAMPLE_LAYERS=("1")
DOWNSAMPLE_WINDOWS=("1")

# Paths (device-aware)
ROOT_OUT="output/imputation/${BACKEND}"
mkdir -p "${ROOT_OUT}"

# Fixed train config
EPOCH=15
PATIENCE=5

# Global session log
SESSION_TAG="$(date +%Y%m%dT%H%M%S)"
SESSION_LOG="${ROOT_OUT}/session_${MODEL}_${SESSION_TAG}.log"
echo "Session log: ${SESSION_LOG}" | tee -a "${SESSION_LOG}"

# ==================== Helper: dataset dims ====================
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

# ==================== Main sweep ====================
for DATASET in "${DATASETS[@]}"; do
  if ! set_dims_for_dataset "$DATASET"; then exit 1; fi

  for MISSING_RATE in "${MISSING_RATES[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
      for D_MODEL in "${D_MODELS[@]}"; do
        for D_FFN in "${D_FFNS[@]}"; do
          for N_HEAD in "${N_HEADS[@]}"; do
            for N_LAYER in "${N_LAYERS[@]}"; do
              for TOP_K in "${TOP_KS[@]}"; do
                for N_KERNEL in "${N_KERNELS[@]}"; do
                  for CHANNEL_MIXING in "${CHANNEL_MIXING_VALUES[@]}"; do
                    for CHANNEL_INDEPENDENCE in "${CHANNEL_INDEPENDENCE_VALUES[@]}"; do
                      for DOWNSAMPLE_LAYER in "${DOWNSAMPLE_LAYERS[@]}"; do
                        for DOWNSAMPLE_WINDOW in "${DOWNSAMPLE_WINDOWS[@]}"; do

                          SAVE_DIR="${ROOT_OUT}/${MODEL}/${DATASET}/epoch${EPOCH}/mr${MISSING_RATE}_bs${BATCH_SIZE}_dm${D_MODEL}_ffn${D_FFN}_h${N_HEAD}_ly${N_LAYER}_topk${TOP_K}_kernels${N_KERNEL}"
                          LOG_DIR="${SAVE_DIR}/logs"
                          mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

                          RUN_TAG="mr${MISSING_RATE}_bs${BATCH_SIZE}_dm${D_MODEL}_ffn${D_FFN}_h${N_HEAD}_ly${N_LAYER}_topk${TOP_K}_kernels${N_KERNEL}_cmix${CHANNEL_MIXING}_cind${CHANNEL_INDEPENDENCE}_dly${DOWNSAMPLE_LAYER}_dwin${DOWNSAMPLE_WINDOW}"
                          RUN_LOG="${LOG_DIR}/run_${RUN_TAG}.log"
                          DONE_MARK="${SAVE_DIR}/.done"

                          if [[ -f "${DONE_MARK}" ]]; then
                            echo "⏭️  Skip (done): ${DATASET} | ${RUN_TAG}" | tee -a "${SESSION_LOG}"
                            continue
                          fi

                          echo "🔥 RUN: ${MODEL} | ${DATASET} | ${RUN_TAG} (DEVICE=${DEVICE[*]}, N_STEPS=${N_STEPS}, N_FEATURES=${N_FEATURES})" | tee -a "${SESSION_LOG}"

                          # Build command as array (no eval)
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
                            --top_k "${TOP_K}"
                            --n_kernels "${N_KERNEL}"
                            --channel_mixing "${CHANNEL_MIXING}"
                            --channel_independence "${CHANNEL_INDEPENDENCE}"
                            --downsampling_layers "${DOWNSAMPLE_LAYER}"
                            --downsampling_window "${DOWNSAMPLE_WINDOW}"
                            --batch_size "${BATCH_SIZE}"
                            --apply_nonstationary_norm
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
      done
    done
  done
done

echo "✅ All ${MODEL} runs completed at $(date -Is)."