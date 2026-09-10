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
# job-name 은 **50자 이상**이어야 한다. 짧으면 슬럼이 거부한다 -- 이 저장소에서
# 통과한 이름은 전부 50자를 넘고, 28·29자로 쓴 둘만 거부당했다.
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --requeue
set -u

# ---------- bundle-sbatch 배열 규약 ----------
# 감싸개는 **부모 뿌리만** 준다. 각 작업이 자기 번호를 검사하고, 자기 디렉터리를
# 만들고, 소유·권한·심링크를 확인한 뒤, 두 변수를 다시 묶고 나서 쓰기 시작한다.
# `#SBATCH --array` 는 금지다 -- 배열은 `bundle-sbatch --array 0-7` 로만 만든다.
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID 가 없다}
case "$task_id" in
  0|1|2|3|4|5|6|7) ;;
  *) echo "예상 밖의 배열 번호 $task_id" >&2; exit 2 ;;
esac
task_dir="${CODE_OUTPUT_DIR:?CODE_OUTPUT_DIR 가 없다 -- bundle-sbatch 로 내십시오}/${task_id}"
mkdir -p -- "$task_dir"
mode=$(stat -c %a -- "$task_dir")
if ! test -d "$task_dir" || test -L "$task_dir" || ! test -O "$task_dir" \
   || test "$mode" != 755; then
  echo "배열 작업 출력 디렉터리가 안전하지 않다: $task_dir" >&2; exit 2
fi
export CODE_OUTPUT_DIR="$task_dir"
export MODEL_OUTPUT_DIR="$task_dir"
: "${RATE:?RATE 를 주십시오 -- 게이트 판의 실측 rate_mean}"
CKPT_DIR="$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
CONDA_PATH="$HOME/miniconda3"
: "${SLURM_ARRAY_TASK_ID:=0}"; : "${SLURM_ARRAY_JOB_ID:=$$}"
PORT=$((10000 + (SLURM_ARRAY_JOB_ID % 90) * 10 + SLURM_ARRAY_TASK_ID))
N_EPISODES="${N_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-1500}"
OUTPUT_BASE="${OUTPUT_BASE:-$CODE_OUTPUT_DIR}"
mkdir -p "$OUTPUT_BASE"
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
