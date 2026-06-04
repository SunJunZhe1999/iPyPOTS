#!/bin/bash
# Finish the benchmark tail: distribute every PENDING (model,dataset,mr,seed) combo
# one-per-GPU across the idle GPUs (4,5,7), race-free. Resumes via existing .done.
# TimeMixer++ and the known-failing GPT4TS/eld are excluded.
set -u
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
PY="${PYTHON_BIN:-python}"
ROOT_OUT="output/imputation/cuda/paper_campaign"
WORK="${ROOT_OUT}/_rebalance"
mkdir -p "$WORK"
EPOCH=60; PATIENCE=12; BATCH_SIZE=256; MAX_SAMPLES=15000; WINDOW_STRIDE=4; RUN_TIMEOUT=1800
GPUS=(4 5 7)

# 1) generate balanced per-GPU work lists (slow gpt4ts spread evenly, then fast)
"$PY" - "$ROOT_OUT" "$WORK" <<'PY'
import glob, itertools, sys
root, work = sys.argv[1], sys.argv[2]
done=set()
for d in glob.glob(f"{root}/*/*/mr*/seed*/.done"):
    p=d.split("/"); done.add((p[-5],p[-4],p[-3],p[-2]))
models=["mean","median","locf","saits","tefn","tslanet","gpt4ts"]
datasets=["household_power","appliances_energy","opsd_germany","citylearn_zone5","etth1","ettm1","solar","eld","physionet_2012"]
DIMS={"household_power":(168,11),"appliances_energy":(72,32),"opsd_germany":(168,10),"citylearn_zone5":(168,20),
      "etth1":(96,7),"ettm1":(96,7),"solar":(24,137),"eld":(24,370),"physionet_2012":(48,37)}
rates=["mr0.1","mr0.3","mr0.5"]; seeds=["seed42","seed123","seed456"]
pend=[]
for m,ds,r,s in itertools.product(models,datasets,rates,seeds):
    if (m,ds,r,s) in done: continue
    if m=="gpt4ts" and ds=="eld": continue
    pend.append((m,ds,r,s))
gpus=[4,5,7]; lanes={g:[] for g in gpus}
# dataset-disjoint: each dataset on exactly ONE GPU -> no concurrent same-dataset cache contention
assign={"etth1":4,"physionet_2012":5,"citylearn_zone5":7,"solar":7}
for x in pend:
    g=assign.get(x[1])
    if g is not None: lanes[g].append(x)
for g in gpus: lanes[g].sort(key=lambda x: x[0]=="gpt4ts")  # fast jobs first, gpt4ts last
for g in gpus:
    with open(f"{work}/gpu{g}.txt","w") as f:
        for m,ds,r,s in lanes[g]:
            ns,nf=DIMS[ds]; f.write(f"{m} {ds} {r[2:]} {s[4:]} {ns} {nf}\n")
    print(f"gpu{g}: {len(lanes[g])} runs ({sum(1 for x in lanes[g] if x[0]=='gpt4ts')} gpt4ts)")
PY

# 2) one sequential lane per GPU
run_lane(){
  local gpu="$1"; local list="${WORK}/gpu${gpu}.txt"; local log="${WORK}/lane_gpu${gpu}.log"
  echo "Lane GPU${gpu} start $(date -Is) ($(wc -l < "$list") runs)" >> "$log"
  local M DS MR SD NS NF SAVE
  while read -r M DS MR SD NS NF; do
    [ -z "${M:-}" ] && continue
    SAVE="${ROOT_OUT}/${M}/${DS}/mr${MR}/seed${SD}"
    if [ -f "${SAVE}/.done" ]; then echo "skip ${M}/${DS}/mr${MR}/seed${SD}" >>"$log"; continue; fi
    mkdir -p "$SAVE"
    CUDA_VISIBLE_DEVICES=${gpu} timeout ${RUN_TIMEOUT} "$PY" main.py \
      --model "$M" --dataset_name "$DS" --missing_rate "$MR" --random_seed "$SD" \
      --saving_path "$SAVE" --n_steps "$NS" --n_features "$NF" \
      --epochs $EPOCH --patience $PATIENCE --batch_size $BATCH_SIZE \
      --d_model 64 --d_ffn 128 --n_heads 4 --n_layers 2 --d_k 16 --d_v 16 --n_fod 4 \
      --max_samples $MAX_SAMPLES --window_stride $WINDOW_STRIDE --device cuda:0 \
      > "${SAVE}/run.log" 2>&1 \
      && touch "${SAVE}/.done" && echo "DONE $(date +%H:%M:%S) ${M}/${DS}/mr${MR}/seed${SD}" >>"$log" \
      || echo "FAIL($?) $(date +%H:%M:%S) ${M}/${DS}/mr${MR}/seed${SD}" >>"$log"
  done < "$list"
  echo "Lane GPU${gpu} end $(date -Is)" >> "$log"
}

echo "Rebalance start $(date -Is) -> ${ROOT_OUT}"
run_lane 4 & P4=$!
run_lane 5 & P5=$!
run_lane 7 & P7=$!
echo "lane pids: gpu4=$P4 gpu5=$P5 gpu7=$P7"
wait
echo "Rebalance complete $(date -Is)"
