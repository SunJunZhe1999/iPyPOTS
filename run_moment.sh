#!/usr/bin/env bash
set -Eeuo pipefail
# ==================== Python interpreter selection (macOS/Linux compatible) ====================
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python &>/dev/null; then
    PYTHON_BIN="python"
  elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
  else
    echo "❌ Neither 'python' nor 'python3' found in PATH. Please install Python 3.8+ or set PYTHON_BIN."
    exit 1
  fi
fi

# ==================== System info ====================
hostname

# ==================== GPU SELECTION (Portable) ====================
# 1) Choose GPUs by *logical* indices ("" → CPU). Default to GPU 0 if available.
USE_GPUS="${USE_GPUS:-0}"    # e.g., "0", "0,1", or "" for CPU

# 2) Build DEVICE list and BACKEND tag without overriding CUDA_VISIBLE_DEVICES
DEVICE=()
BACKEND="cpu"
if command -v nvidia-smi &>/dev/null && [[ -n "${USE_GPUS}" ]]; then
  IFS=',' read -ra SEL <<< "${USE_GPUS}"
  for lg in "${SEL[@]}"; do
    if [[ "${lg}" =~ ^[0-9]+$ ]]; then DEVICE+=("cuda:${lg}"); fi
  done
  if [[ ${#DEVICE[@]} -gt 0 ]]; then BACKEND="cuda"; fi
fi

# If no GPU selected/available, fallback to CPU
if [[ ${#DEVICE[@]} -eq 0 ]]; then DEVICE=("cpu"); BACKEND="cpu"; fi

echo "Using devices: ${DEVICE[*]}  | backend=${BACKEND}"
echo "USE_GPUS=${USE_GPUS} (set USE_GPUS=\"\" to force CPU, or \"0,1\" for multi-GPU)"

# ==================== Conda ENV (robust activation) ====================
ENV_NAME="${ENV_NAME:-pypots}"
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
  if command -v conda &>/dev/null; then
    # Prefer the modern shell hook
    if conda shell.bash hook &>/dev/null; then
      eval "$(conda shell.bash hook)"
      conda activate "${ENV_NAME}" || { echo "❌ Could not activate Conda env '${ENV_NAME}'."; exit 1; }
    else
      # Fallback to common install paths
      if [[ -r "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
        conda activate "${ENV_NAME}" || { echo "❌ Could not activate Conda env '${ENV_NAME}'."; exit 1; }
      else
        echo "⚠️  Conda found but shell hook not available; continuing without activation."
      fi
    fi
  else
    echo "⚠️  Conda not found; continuing with current Python environment."
  fi
fi

# ==================== EXPERIMENT CONFIGS ====================
MODEL="${MODEL:-moment}"

# Allow overriding dataset/missing-rate grids via env, e.g. DATASETS_OVERRIDE="physionet_2012,ettm1"
declare -a DATASETS
if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  IFS=',' read -ra DATASETS <<< "${DATASETS_OVERRIDE}"
else
  DATASETS=("physionet_2012")
fi

# NOTE: d_k / d_v only used in specific transformer configs, guarded below
function set_hyperparams_for_dataset() {
  local ds="$1"
  case "$ds" in
    "physionet_2012")
      D_MODEL=512
      D_FFN=1024
      N_HEAD=8
      N_LAYER=4
      BATCH_SIZE=16
      EPOCH=100
      PATIENCE=20
      ;;
    *)
      D_MODEL=512
      D_FFN=1024
      N_HEAD=8
      N_LAYER=4
      BATCH_SIZE=16
      EPOCH=100
      PATIENCE=20
      ;;
  esac
}

# Missing rate grid
declare -a MISSING_RATES
if [[ -n "${MISSING_RATES_OVERRIDE:-}" ]]; then
  IFS=',' read -ra MISSING_RATES <<< "${MISSING_RATES_OVERRIDE}"
else
  MISSING_RATES=("0.1" "0.2")
fi

# Model sweep knobs (patch size/backbones/transformer types/d_k/d_v)
declare -a PATCH_SIZES
if [[ -n "${PATCH_SIZES_OVERRIDE:-}" ]]; then
  IFS=',' read -ra PATCH_SIZES <<< "${PATCH_SIZES_OVERRIDE}"
else
  PATCH_SIZES=(8)
fi

declare -a TRANSFORMER_BACKBONES
if [[ -n "${TRANSFORMER_BACKBONES_OVERRIDE:-}" ]]; then
  IFS=',' read -ra TRANSFORMER_BACKBONES <<< "${TRANSFORMER_BACKBONES_OVERRIDE}"
else
  TRANSFORMER_BACKBONES=("t5-small")
fi

declare -a TRANSFORMER_TYPES
if [[ -n "${TRANSFORMER_TYPES_OVERRIDE:-}" ]]; then
  IFS=',' read -ra TRANSFORMER_TYPES <<< "${TRANSFORMER_TYPES_OVERRIDE}"
else
  TRANSFORMER_TYPES=("encoder_decoder")
fi

declare -a D_K_VALUES
if [[ -n "${D_K_OVERRIDE:-}" ]]; then
  IFS=',' read -ra D_K_VALUES <<< "${D_K_OVERRIDE}"
else
  D_K_VALUES=(64)
fi

declare -a D_V_VALUES
if [[ -n "${D_V_OVERRIDE:-}" ]]; then
  IFS=',' read -ra D_V_VALUES <<< "${D_V_OVERRIDE}"
else
  D_V_VALUES=(64)
fi

# Root output path (align with imputation layout)
ROOT_OUT="${ROOT_OUT:-./output/imputation/${BACKEND}}"
mkdir -p "${ROOT_OUT}"

# Profiling
ENABLE_PROFILING="${ENABLE_PROFILING:-false}"
PROFILING_PATH="${PROFILING_PATH:-${ROOT_OUT}/profiling}"
PROFILING_PREFIX="${PROFILING_PREFIX:-moment}"


# ==================== CPU-friendly defaults ====================
if [[ "${BACKEND}" == "cpu" ]]; then
  auto_fast_dev=0
  if [[ -z "${FAST_DEV_RUN:-}" ]]; then
    FAST_DEV_RUN="auto"
    auto_fast_dev=1
    echo "⚠️ 检测到 CPU 后端，自动开启 FAST_DEV_RUN（若需完整训练请设置 FAST_DEV_RUN=0）。"
  fi
  if [[ ${auto_fast_dev} -eq 1 && -z "${TRAIN_SUBSET:-}" ]]; then
    TRAIN_SUBSET="64"
    echo "⚠️ 使用 TRAIN_SUBSET=64 加速调试（设置 TRAIN_SUBSET=all 覆盖此行为）。"
  fi
  if [[ ${auto_fast_dev} -eq 1 && -z "${EPOCH_OVERRIDE:-}" ]]; then
    EPOCH_OVERRIDE=1
  fi
  if [[ ${auto_fast_dev} -eq 1 && -z "${PATIENCE_OVERRIDE:-}" ]]; then
    PATIENCE_OVERRIDE=1
  fi
fi

# ==================== Path and script targets ====================
# Auto-detect main entry: prefer ./main.py, fallback to ./script/main.py, or respect MAIN_PY if provided
if [[ -z "${MAIN_PY:-}" ]]; then
  if [[ -f "./main.py" ]]; then
    MAIN_PY="./main.py"
  elif [[ -f "./script/main.py" ]]; then
    MAIN_PY="./script/main.py"
  else
    echo "❌ Could not find main.py. Tried ./main.py and ./script/main.py. You can also run: MAIN_PY=/path/to/main.py ./run_moment.sh"
    exit 1
  fi
fi
echo "MAIN_PY=${MAIN_PY}"

SESSION_LOG="./output/run_moment_session.log"
mkdir -p "$(dirname "${SESSION_LOG}")"

# ==================== CLI Introspection helpers ====================

# Detect whether --device is a 'store_true' nargs style or single string
#  - We check parser for nargs in add_argument('--device', ...)
device_arg_style="single"
if [[ -f "${MAIN_PY}" ]] && grep -Eq "add_argument\(([^)]*['\"]--device['\"]).*nargs" "${MAIN_PY}"; then
  device_arg_style="multi"
fi

# Detect whether CLI supports --d_k and --d_v
accepts_dk=1
accepts_dv=1
if [[ -f "${MAIN_PY}" ]]; then
  if grep -q -- "--d_k" "${MAIN_PY}"; then accepts_dk=0; fi
  if grep -q -- "--d_v" "${MAIN_PY}"; then accepts_dv=0; fi
fi

# Detect whether --enable_profiling is a store_true flag
enable_prof_is_flag=""
if [[ -f "${MAIN_PY}" ]] && grep -Eq "add_argument\(([^)]*['\"]--enable_profiling['\"]).*action=['\"]store_true" "${MAIN_PY}"; then
  enable_prof_is_flag="true"
fi

# ==================== Utility: default dimensions per dataset ====================
function set_dims_for_dataset() {
  local ds="$1"
  case "$ds" in
    "physionet_2012") N_STEPS=48;  N_FEATURES=37 ;;
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

# ==================== MAIN GRID ====================
for DATASET in "${DATASETS[@]}"; do
  set_dims_for_dataset "${DATASET}" || continue

  set_hyperparams_for_dataset "${DATASET}"

  # Allow lightweight experimentation via optional env overrides
  if [[ -n "${D_MODEL_OVERRIDE:-}" ]]; then D_MODEL="${D_MODEL_OVERRIDE}"; fi
  if [[ -n "${D_FFN_OVERRIDE:-}" ]]; then D_FFN="${D_FFN_OVERRIDE}"; fi
  if [[ -n "${N_HEAD_OVERRIDE:-}" ]]; then N_HEAD="${N_HEAD_OVERRIDE}"; fi
  if [[ -n "${N_LAYER_OVERRIDE:-}" ]]; then N_LAYER="${N_LAYER_OVERRIDE}"; fi
  if [[ -n "${BATCH_SIZE_OVERRIDE:-}" ]]; then BATCH_SIZE="${BATCH_SIZE_OVERRIDE}"; fi
  if [[ -n "${EPOCH_OVERRIDE:-}" ]]; then EPOCH="${EPOCH_OVERRIDE}"; fi
  if [[ -n "${PATIENCE_OVERRIDE:-}" ]]; then PATIENCE="${PATIENCE_OVERRIDE}"; fi

  for MISSING_RATE in "${MISSING_RATES[@]}"; do
    for PATCH_SIZE in "${PATCH_SIZES[@]}"; do
      PATCH_STRIDE="${PATCH_SIZE}"
      for TRANS_BACKBONE in "${TRANSFORMER_BACKBONES[@]}"; do
        for TRANS_TYPE in "${TRANSFORMER_TYPES[@]}"; do
          for D_K in "${D_K_VALUES[@]}"; do
            for D_V in "${D_V_VALUES[@]}"; do

                SAVE_DIR="${ROOT_OUT}/${MODEL}/${DATASET}/mr${MISSING_RATE}"
                LOG_DIR="${SAVE_DIR}/logs"
                mkdir -p "${SAVE_DIR}" "${LOG_DIR}"
                rm -f "${SAVE_DIR}"/*.pypots 2>/dev/null || true

                RUN_TAG="mr${MISSING_RATE}_bs${BATCH_SIZE}_dm${D_MODEL}_ffn${D_FFN}_h${N_HEAD}_ly${N_LAYER}_prof${ENABLE_PROFILING}"
                RUN_LOG="${LOG_DIR}/run_${RUN_TAG}.log"
                DONE_MARK="${SAVE_DIR}/.done"

                # Allow force rerun
                if [[ -n "${FORCE_RERUN:-}" ]]; then rm -f "${DONE_MARK}"; fi

                if [[ -f "${DONE_MARK}" ]]; then
                  echo "⏭️  Skip (done): ${DATASET} | ${RUN_TAG}" | tee -a "${SESSION_LOG}"
                  continue
                fi

                echo "🔥 RUN: ${MODEL} | ${DATASET} | ${RUN_TAG} (n_steps=${N_STEPS}, n_features=${N_FEATURES}, d_k=${D_K}, d_v=${D_V})" | tee -a "${SESSION_LOG}"

                # build command
                cmd=(${PYTHON_BIN} "${MAIN_PY}"
                  --model "${MODEL}"
                  --dataset_name "${DATASET}"
                  --epochs "${EPOCH}"
                  --patience "${PATIENCE}"
                  --missing_rate "${MISSING_RATE}"
                  --saving_path "${SAVE_DIR}"
                  --n_steps "${N_STEPS}"
                  --n_features "${N_FEATURES}"
                  --n_layers "${N_LAYER}"
                  --d_model "${D_MODEL}"
                  --d_ffn "${D_FFN}"
                  --n_heads "${N_HEAD}"
                  --batch_size "${BATCH_SIZE}"
                  --profiling_path "${PROFILING_PATH}"
                  --profiling_prefix "${PROFILING_PREFIX}"
                )

                # device passing (multi-arg vs single string)
                if [[ "${device_arg_style}" == "multi" ]]; then
                  cmd+=(--device)
                  for d in "${DEVICE[@]}"; do cmd+=("${d}"); done
                else
                  devices_joined="$(IFS=,; echo "${DEVICE[*]}")"
                  if [[ -z "${devices_joined}" && "${BACKEND}" == "cpu" ]]; then devices_joined="cpu"; fi
                  cmd+=(--device "${devices_joined}")
                fi

                # enable_profiling handling (flag vs value)
                if [[ "${ENABLE_PROFILING}" == "true" ]]; then
                  if [[ -n "${enable_prof_is_flag}" ]]; then
                    cmd+=(--enable_profiling)
                  else
                    cmd+=(--enable_profiling "true")
                  fi
                else
                  if [[ -z "${enable_prof_is_flag}" ]]; then
                    cmd+=(--enable_profiling "false")
                  fi
                fi

                # only append d_k/d_v if CLI accepts them
                if [[ "${accepts_dk}" -eq 0 ]]; then cmd+=(--d_k "${D_K}"); fi
                if [[ "${accepts_dv}" -eq 0 ]]; then cmd+=(--d_v "${D_V}"); fi

                fast_flag="${FAST_DEV_RUN:-}"
                fast_flag_lower=$(printf '%s' "${fast_flag}" | tr '[:upper:]' '[:lower:]')
                if [[ "${fast_flag_lower}" == "true" || "${fast_flag_lower}" == "auto" || "${fast_flag}" == "1" ]]; then
                  cmd+=(--fast_dev_run)
                fi
                if [[ -n "${TRAIN_SUBSET:-}" && "${TRAIN_SUBSET}" != "all" && "${TRAIN_SUBSET}" != "ALL" ]]; then
                  cmd+=(--train_subset "${TRAIN_SUBSET}")
                fi

                # Other MOMENT-specific args
                cmd+=(--patch_size "${PATCH_SIZE}")
                cmd+=(--patch_stride "${PATCH_STRIDE}")
                cmd+=(--transformer_backbone "${TRANS_BACKBONE}")
                cmd+=(--transformer_type "${TRANS_TYPE}")
                cmd+=(--verbose)                                # ← 修正：不再传 "true"
                cmd+=(--orth_gain "0")
                cmd+=(--model_saving_strategy "best")

                # run
                set +e
                {
                  "${cmd[@]}" 2>&1 | tee "${RUN_LOG}"
                }
                status=${PIPESTATUS[0]}
                set -e

                if [[ ${status} -eq 0 ]]; then
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
echo "✅ All ${MODEL} runs completed at $(date "+%Y-%m-%dT%H:%M:%S%z")."
