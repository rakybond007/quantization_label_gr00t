#!/bin/bash
# 균일 대조군. **GPU 1** — 판정기를 안 쓰니 한 장이면 된다.
#
#   RATE=1.87 sbatch vlm_gate/run_scripts/eval/uniform_control.sh
#
# 게이트 판의 **실측 `rate_mean`** 을 그대로 넣는다. 예상값이 아니라 실측값이다.
#
# 이것이 바다. 사다리를 바로 쓰지 않는 이유는 사다리가 태스크 단위이고 우리가
# 묻는 것은 시점 단위이기 때문이다. **평균 배속이 같은데 게이트가 나으면 시점을
# 가른 것이고, 아니면 아니다.** 사다리 칸(1.5·2.0·2.5)에 정확히 안 맞는 평균이
# 나오므로 그 사이를 보간해 견주는 것보다 같은 값으로 한 판 더 도는 쪽이 깨끗하다.
#SBATCH --wckey=project-short-name:sub_fast
# job-name 은 **50자 이상**이어야 한다. 짧으면 슬럼이 거부한다 -- 이 저장소에서
# 통과한 이름은 전부 50자를 넘고, 28·29자로 쓴 둘만 거부당했다.
#SBATCH --job-name=uniform_control_matched_mean_ratio_robocasa_24tasks_50ep_arr8
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --array=0-7
#SBATCH --requeue
#SBATCH --output=out/%A_%a-uniform_control.out
#SBATCH --error=out/%A_%a-uniform_control.err
set -u
: "${RATE:?RATE 를 주십시오 -- 게이트 판의 실측 rate_mean}"
CKPT_DIR="$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
CONDA_PATH="$HOME/miniconda3"
: "${SLURM_ARRAY_TASK_ID:=0}"; : "${SLURM_ARRAY_JOB_ID:=$$}"
PORT=$((10000 + (SLURM_ARRAY_JOB_ID % 90) * 10 + SLURM_ARRAY_TASK_ID))
N_EPISODES="${N_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-1500}"
OUTPUT_BASE="${OUTPUT_BASE:-$BASE_DIR/output/robocasa/uniform_$RATE}"
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NV="$CONDA_PATH/envs/quant_gate/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"
cleanup() { kill ${SPID:-} 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT

SERVER_LOG="$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$SERVER_LD" PYTHONUNBUFFERED=1 \
"$CONDA_PATH/envs/quant_gate/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT_DIR" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head main > "$SERVER_LOG" 2>&1 &
SPID=$!
for i in $(seq 1 180); do
    grep -q "Server is ready" "$SERVER_LOG" 2>/dev/null && break
    kill -0 $SPID 2>/dev/null || { echo "[ERR] GR00T 죽음"; tail -40 "$SERVER_LOG"; exit 1; }
    sleep 5
done

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
        --frac-ratio "$RATE" \
        >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done
for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
echo "[i] 균일 $RATE · array $SLURM_ARRAY_TASK_ID 끝"
