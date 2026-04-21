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
      if ! command -v conda &>/dev/null; then echo "Conda not found and .venv/bin/python is unavailable."; exit 1; fi
      source activate "$ENV_NAME" || { echo "Could not activate Conda env '$ENV_NAME'."; exit 1; }
    fi
    PYTHON_BIN="python"
  fi
fi
echo "Using Python: ${PYTHON_BIN}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
USE_GPUS="${USE_GPUS:-1}"
USE_MPS="${USE_MPS:-1}"

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

read -ra MODELS <<< "${MODELS:-mean median locf saits tefn timemixerpp uniformtsv}"
read -ra DATASETS <<< "${DATASETS:-physionet_2012 appliances_energy household_power citylearn_zone5 opsd_germany}"
read -ra MISSING_RATES <<< "${MISSING_RATES:-0.1 0.3 0.5}"
read -ra SEEDS <<< "${SEEDS:-42 123 456}"

ROOT_OUT="output/imputation/${BACKEND}/energy_benchmark"
mkdir -p "${ROOT_OUT}"

EPOCH="${EPOCH:-50}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
D_MODEL="${D_MODEL:-64}"
D_FFN="${D_FFN:-128}"
N_HEAD="${N_HEAD:-4}"
N_LAYER="${N_LAYER:-2}"
PATCH_SIZE="${PATCH_SIZE:-12}"
PATCH_STRIDE="${PATCH_STRIDE:-12}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
N_FOD="${N_FOD:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-20000}"
WINDOW_STRIDE="${WINDOW_STRIDE:-4}"

set_dims_for_dataset () {
  local ds="$1"
  case "$ds" in
    "physionet_2012") N_STEPS=48;  N_FEATURES=35 ;;
    "appliances_energy") N_STEPS=72; N_FEATURES=32 ;;
    "household_power") N_STEPS=168; N_FEATURES=11 ;;
    "citylearn_zone5") N_STEPS=168; N_FEATURES=20 ;;
    "opsd_germany") N_STEPS=168; N_FEATURES=10 ;;
    *) echo "Unknown dataset: $ds"; return 1 ;;
  esac
  return 0
}

SESSION_TAG="$(date +%Y%m%dT%H%M%S)"
SESSION_LOG="${ROOT_OUT}/session_energy_benchmark_${SESSION_TAG}.log"
echo "Session log: ${SESSION_LOG}" | tee -a "${SESSION_LOG}"

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    if ! set_dims_for_dataset "$DATASET"; then exit 1; fi

    for MISSING_RATE in "${MISSING_RATES[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        D_K=$(( D_MODEL / N_HEAD ))
        D_V=$D_K
        RUN_TAG="${MODEL}_${DATASET}_mr${MISSING_RATE}_seed${SEED}"
        SAVE_DIR="${ROOT_OUT}/${MODEL}/${DATASET}/mr${MISSING_RATE}/seed${SEED}"
        LOG_DIR="${SAVE_DIR}/logs"
        RUN_LOG="${LOG_DIR}/run_${RUN_TAG}.log"
        DONE_MARK="${SAVE_DIR}/.done"
        mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

        if [[ -f "${DONE_MARK}" ]]; then
          echo "Skip done: ${RUN_TAG}" | tee -a "${SESSION_LOG}"
          continue
        fi

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
          --n_fod "${N_FOD}"
          --max_samples "${MAX_SAMPLES}"
          --window_stride "${WINDOW_STRIDE}"
          --random_seed "${SEED}"
          --model_saving_strategy "best"
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
          echo "DONE: ${RUN_TAG}" | tee -a "${SESSION_LOG}"
        else
          echo "FAIL(${status}): ${RUN_TAG}" | tee -a "${SESSION_LOG}"
        fi
      done
    done
  done
done

echo "Energy benchmark completed at $(timestamp)." | tee -a "${SESSION_LOG}"
