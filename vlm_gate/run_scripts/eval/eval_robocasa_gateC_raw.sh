#!/bin/bash
#SBATCH --job-name=eval_robocasa_gateC_joint_checkpoint_raw_sanity_no_gate_always_k1_24tasks_50ep
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --array=0-7
#SBATCH --output=out/%A_%a-eval_gateC_raw.out
#SBATCH --error=out/%A_%a-eval_gateC_raw.err
#SBATCH --time=2-00:00:00

# Raw sanity eval of the gateC joint checkpoint: gate UNUSED (head=main,
# no compression, K1 always). Checks whether joint training with the gate BCE
# hurt the policy itself. Anchor: baseline 0.657 (327/498-ish, 24t x 50ep).
set -u
CKPT_DIR="${CKPT_DIR:-/rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_cosmos_v1/vla_gateC_60k/checkpoint-60000}"
: "${SLURM_ARRAY_TASK_ID:=0}"
: "${SLURM_ARRAY_JOB_ID:=$$}"
POFF=$(( (SLURM_ARRAY_JOB_ID % 90) * 10 ))
PORT=$((11000 + POFF + SLURM_ARRAY_TASK_ID))
N_EPISODES="${N_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-1500}"

BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
CONDA_PATH="$HOME/miniconda3"
OUTPUT_BASE="${OUTPUT_BASE:-$BASE_DIR/output/robocasa/gateC_raw}"
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NV="$CONDA_PATH/envs/quant_gate/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"

cleanup(){ kill ${SPID:-} 2>/dev/null; sleep 2; kill -9 ${SPID:-} 2>/dev/null; }
trap cleanup EXIT

SERVER_LOG="$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$SERVER_LD" PYTHONUNBUFFERED=1 \
"$CONDA_PATH/envs/quant_gate/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT_DIR" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head main > "$SERVER_LOG" 2>&1 &
SPID=$!

for i in $(seq 1 200); do
    grep -q "Server is ready" "$SERVER_LOG" 2>/dev/null && break
    kill -0 "$SPID" 2>/dev/null || { echo "[ERR] server died"; tail -40 "$SERVER_LOG"; exit 1; }
    sleep 5
done
grep -q "Server is ready" "$SERVER_LOG" || { echo "[ERR] server not ready"; exit 1; }

TASK_NAMES=(
  "TurnSinkSpout" "TurnOnStove" "TurnOnSinkFaucet" "TurnOnMicrowave"
  "TurnOffStove" "TurnOffSinkFaucet" "TurnOffMicrowave" "PnPStoveToCounter"
  "PnPSinkToCounter" "PnPMicrowaveToCounter" "PnPCounterToStove" "PnPCounterToSink"
  "PnPCounterToMicrowave" "PnPCounterToCab" "PnPCabToCounter" "OpenSingleDoor"
  "OpenDrawer" "OpenDoubleDoor" "CoffeeSetupMug" "CoffeeServeMug"
  "CoffeePressButton" "CloseSingleDoor" "CloseDrawer" "CloseDoubleDoor"
)
SELECTED=()
[ $SLURM_ARRAY_TASK_ID -lt 8 ] && SELECTED+=("${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}")
[ $((SLURM_ARRAY_TASK_ID + 8)) -lt 24 ] && SELECTED+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 8))]}")
[ $((SLURM_ARRAY_TASK_ID + 16)) -lt 24 ] && SELECTED+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 16))]}")

MAIN_PIDS=()
for TASK in "${SELECTED[@]}"; do
    ODIR="$OUTPUT_BASE/$TASK"; mkdir -p "$ODIR"
    PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
        "$CONDA_PATH/envs/quant_gate_eval/bin/python" -u \
        "$BASE_DIR/scripts/robocasa_service_compress.py" \
        --port $PORT --host localhost --env_name "$TASK" \
        --video_dir "$ODIR" --seed 42 --n_episodes $N_EPISODES \
        --max_episode_steps $MAX_STEPS --generative_textures \
        --compress-k 1 \
        >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done
for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
echo "[i] Array $SLURM_ARRAY_TASK_ID done (gateC raw sanity, K1)."
