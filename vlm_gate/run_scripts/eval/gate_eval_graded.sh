#!/bin/bash
# 등급표 게이트 평가. **GPU 2** — 0번 VLA, 1번 Cosmos 판정기.
#
# 한 GPU 에 둘 다 올리지 않는다. Cosmos3-Nano(16B) 가 bf16 으로 32GB 쯤이고
# GR00T 가 그 위에 얹히면 활성값까지 빠듯하다. 기존 게이트 평가
# (eval_robocasa_cosmos_tau0p5.sh) 도 둘로 나눠 썼고 그 배치를 그대로 따른다.
#
#   MODE=stage1 sbatch vlm_gate/run_scripts/eval/gate_eval_graded.sh
#   MODE=stage2 sbatch vlm_gate/run_scripts/eval/gate_eval_graded.sh
#
# 두 판이 무엇을 묻는가
#
#   stage1  VLM 이 5문항 등급표로 답한다 -> 신뢰도 -> 압축할까 말까.
#           압축하면 2.5배, 아니면 1.5배.
#   stage2  거기에 더해, 압축하기로 한 청크의 배속을 **현오차**가 고른다.
#
# **압축을 2.5배로 하는 이유**: 1.5배로 압축하면 아무 일도 안 일어난다.
# 균일 1.5 가 실측으로 공짜이고(0.6567 -> 0.6467, 0.7SE), 1.5배속에서는
# 질러간 양이 손상을 전혀 안 가른다(-0.1%). 막을 손상이 없으면 게이트가
# 잘하는지 못하는지 알 수 없다. 손상이 생기는 2배속 위로 올려야 시험이 된다.
#
# **안 할 때를 1.5 로 두는 이유**: 같은 실측이다. 1.0 으로 내리면 측정된
# 여유를 버린다. 지금 접촉 가드가 34.9% 를 1.0 에 박고 있는 그 자리다.
#
# 대조군은 이 판들의 실측 rate_mean 을 보고 uniform_control.sh 로 따로 던진다.
# **사다리를 바로 쓰지 않는다** -- 사다리는 태스크 단위이고 우리가 묻는 것은
# 시점 단위다. 평균 배속이 같은데 게이트가 나으면 시점을 가른 것이다.
#SBATCH --wckey=project-short-name:sub_fast
# job-name 은 **50자 이상**이어야 한다. 짧으면 슬럼이 거부한다 -- 이 저장소에서
# 통과한 이름은 전부 50자를 넘고, 28·29자로 쓴 둘만 거부당했다.
#SBATCH --job-name=gate_eval_graded_vlm_grades_robocasa_24tasks_50ep_arr8
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=background
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --array=0-7
#SBATCH --requeue
#SBATCH --output=out/%A_%a-gate_eval_graded.out
#SBATCH --error=out/%A_%a-gate_eval_graded.err
set -u
MODE="${MODE:-stage1}"
# **두 판이 같은 역치를 쓴다.** 그래야 stage2 가 더한 것이 하나뿐이 된다 --
# 압축하기로 한 청크의 배속을 고정 2.5 로 두느냐, 현오차가 눈금에서 고르느냐.
# 역치까지 같이 바꾸면 무엇이 효과인지 또 못 가린다.
case "$MODE" in
  stage1) TAU=0.517; EXTRA="" ;;
  stage2) TAU=0.517; EXTRA="--rate-by-chord 0.1200 --rate-grid 1.5,2.0,2.5" ;;
  *) echo "MODE 는 stage1 또는 stage2"; exit 1 ;;
esac
# 역치는 pick_thresholds.py 가 라벨 분포에서 낸 값이고 0.383~0.392 봉우리는
# 피했다. **온라인 분포로 다시 잡지 않는다** -- 그것은 시험 대상에 맞춰 선을
# 옮기는 것이다. 스모크에서 이 역치가 49% 를 통과시켰다(라벨 기준 예상 35%).
# 반반에 가까운 쪽이 게이트에 가르는 여지가 제일 크므로 그대로 간다.

CKPT_DIR="$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
COSMOS_VENV="$HOME/quantization_agent_workspace/cosmos_judge_venv"
BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
CONDA_PATH="$HOME/miniconda3"
: "${SLURM_ARRAY_TASK_ID:=0}"; : "${SLURM_ARRAY_JOB_ID:=$$}"
POFF=$(( (SLURM_ARRAY_JOB_ID % 90) * 10 ))
PORT=$((10000 + POFF + SLURM_ARRAY_TASK_ID))
JUDGE_PORT=$((20000 + POFF + SLURM_ARRAY_TASK_ID))
N_EPISODES="${N_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-1500}"
OUTPUT_BASE="${OUTPUT_BASE:-$BASE_DIR/output/robocasa/gate_graded_$MODE}"
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NV="$CONDA_PATH/envs/quant_gate/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"
cleanup() { kill ${SPID:-} ${JPID:-} 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT

SERVER_LOG="$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$SERVER_LD" PYTHONUNBUFFERED=1 \
"$CONDA_PATH/envs/quant_gate/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT_DIR" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head main > "$SERVER_LOG" 2>&1 &
SPID=$!

JUDGE_LOG="$OUTPUT_BASE/judge-$SLURM_ARRAY_TASK_ID.log"
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts" \
"$COSMOS_VENV/bin/python" -u "$BASE_DIR/scripts/vlm_gate_cosmos.py" --serve \
    --model nvidia/Cosmos3-Nano --port $JUDGE_PORT --host 127.0.0.1 > "$JUDGE_LOG" 2>&1 &
JPID=$!

wait_ready() {
    for i in $(seq 1 180); do
        grep -qE "$2" "$1" 2>/dev/null && return 0
        kill -0 "$3" 2>/dev/null || { echo "[ERR] $4 죽음"; tail -40 "$1"; return 1; }
        sleep 5
    done
    echo "[ERR] $4 안 뜸"; tail -40 "$1"; return 1
}
wait_ready "$SERVER_LOG" "Server is ready" "$SPID" "GR00T" || exit 1
wait_ready "$JUDGE_LOG" "JUDGE READY" "$JPID" "Cosmos" || exit 1
echo "[i] 두 서버 준비됨 (array $SLURM_ARRAY_TASK_ID · MODE=$MODE · tau=$TAU)"

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
        --compress-k 2 \
        --judge-url "http://127.0.0.1:$JUDGE_PORT" \
        --judge-checks phase9_checks --judge-threshold $TAU \
        --rate-on 2.5 --rate-off 1.5 --judge-view-parity unflip $EXTRA \
        --gate-subchunk 8 \
        >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done
for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
echo "[i] array $SLURM_ARRAY_TASK_ID 끝. 산출물로 본다 -- gate_format_fail ·"
echo "    grade_A..E · gate_quantize_rate · rate_mean"
