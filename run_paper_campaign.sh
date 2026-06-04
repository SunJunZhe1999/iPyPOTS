#!/bin/bash
# Real C-TCAR model-routing benchmark for the journal paper.
# Fans (model x dataset x missing_rate x seed) across 4 GPU lanes (4,5,6,7).
# Resumable: each finished run drops a .done marker and is skipped on re-run.
set -u
cd "$(dirname "$0")"

PY="${PYTHON_BIN:-python}"
ROOT_OUT="${ROOT_OUT:-output/imputation/cuda/paper_campaign}"
MODELS="${MODELS:-mean median locf saits tefn timemixerpp tslanet gpt4ts}"
MISSING_RATES="${MISSING_RATES:-0.1 0.3 0.5}"
SEEDS="${SEEDS:-42 123 456}"
EPOCH="${EPOCH:-60}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_SAMPLES="${MAX_SAMPLES:-15000}"
WINDOW_STRIDE="${WINDOW_STRIDE:-4}"
RUN_TIMEOUT="${RUN_TIMEOUT:-1800}"
mkdir -p "${ROOT_OUT}/_logs"

declare -A NSTEPS=( [physionet_2012]=48 [appliances_energy]=72 [household_power]=168 [citylearn_zone5]=168 [opsd_germany]=168 [etth1]=96 [etth2]=96 [ettm1]=96 [ettm2]=96 [solar]=24 [eld]=24 )
declare -A NFEAT=(  [physionet_2012]=37 [appliances_energy]=32 [household_power]=11 [citylearn_zone5]=20 [opsd_germany]=10 [etth1]=7  [etth2]=7  [ettm1]=7  [ettm2]=7  [solar]=137 [eld]=370 )

run_lane () {
  local gpu="$1"; local datasets="$2"
  echo "Lane GPU${gpu} start $(date -Is) datasets=[${datasets}]"
  local DS NS NF M MR SD SAVE
  for DS in ${datasets}; do
    NS=${NSTEPS[$DS]}; NF=${NFEAT[$DS]}
    for M in ${MODELS}; do for MR in ${MISSING_RATES}; do for SD in ${SEEDS}; do
      SAVE="${ROOT_OUT}/${M}/${DS}/mr${MR}/seed${SD}"
      if [ -f "${SAVE}/.done" ]; then echo "skip ${M}/${DS}/mr${MR}/seed${SD}"; continue; fi
      mkdir -p "${SAVE}"
      CUDA_VISIBLE_DEVICES=${gpu} timeout ${RUN_TIMEOUT} "${PY}" main.py \
        --model "${M}" --dataset_name "${DS}" --missing_rate "${MR}" --random_seed "${SD}" \
        --saving_path "${SAVE}" --n_steps "${NS}" --n_features "${NF}" \
        --epochs "${EPOCH}" --patience "${PATIENCE}" --batch_size "${BATCH_SIZE}" \
        --d_model 64 --d_ffn 128 --n_heads 4 --n_layers 2 --d_k 16 --d_v 16 --n_fod 4 \
        --max_samples "${MAX_SAMPLES}" --window_stride "${WINDOW_STRIDE}" --device cuda:0 \
        > "${SAVE}/run.log" 2>&1 \
        && touch "${SAVE}/.done" && echo "DONE $(date +%H:%M:%S) ${M}/${DS}/mr${MR}/seed${SD}" \
        || echo "FAIL($?) $(date +%H:%M:%S) ${M}/${DS}/mr${MR}/seed${SD}"
    done; done; done
  done
  echo "Lane GPU${gpu} end $(date -Is)"
}

echo "Campaign start $(date -Is) -> ${ROOT_OUT}"
run_lane 4 "household_power appliances_energy"      >> "${ROOT_OUT}/_logs/lane_gpu4.log" 2>&1 &
P4=$!
run_lane 5 "opsd_germany citylearn_zone5 etth1"     >> "${ROOT_OUT}/_logs/lane_gpu5.log" 2>&1 &
P5=$!
run_lane 6 "ettm1 solar"                            >> "${ROOT_OUT}/_logs/lane_gpu6.log" 2>&1 &
P6=$!
run_lane 7 "eld physionet_2012"                     >> "${ROOT_OUT}/_logs/lane_gpu7.log" 2>&1 &
P7=$!
echo "lane pids: gpu4=$P4 gpu5=$P5 gpu6=$P6 gpu7=$P7"
wait
echo "Campaign complete $(date -Is)"
