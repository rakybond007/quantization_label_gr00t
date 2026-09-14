#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=eval_ratio_gate_r2_mode_two_tasks_conf_picks_rate_under_measured_task_ceiling
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=debug
#SBATCH --exclude=worker-node100
#SBATCH --array=0-3
#SBATCH --time=2:45:00
#SBATCH --output=out/%A_%a-eval_robocasa_cosmos.out
#SBATCH --error=out/%A_%a-eval_robocasa_cosmos.err
#SBATCH --comment="robocasa base GR00T-N1.5 @60k + zero-shot COSMOS-gated K=2 quantization (nvidia/Cosmos3-Nano judge, tau=0.5). 24 tasks. GPU0=GR00T serve, GPU1=Cosmos judge."

# Cosmos variant of eval_robocasa_vlm_gate_gemma4_tau0p5_mv_guide.sh:
# GPU0 = base GR00T server (quant_gate, head=main); GPU1 = nvidia/Cosmos3-Nano
# judge via the transformers reasoner path (cosmos_judge_venv). Same compress
# client + --judge-guidance + gate-subchunk. Array 0-7 -> 24 tasks (3 per job).
set -u
CKPT_DIR="$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
COSMOS_VENV="$HOME/quantization_agent_workspace/cosmos_judge_venv"
COSMOS_MODEL="nvidia/Cosmos3-Nano"
: "${SLURM_ARRAY_TASK_ID:=0}"
: "${SLURM_ARRAY_JOB_ID:=$$}"
POFF=$(( (SLURM_ARRAY_JOB_ID % 90) * 10 ))
PORT=$((10000 + POFF + SLURM_ARRAY_TASK_ID))
JUDGE_PORT=$((20000 + POFF + SLURM_ARRAY_TASK_ID))
N_EPISODES="${N_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-1500}"
K=2
TAU="${TAU:-0.5}"

BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
CONDA_PATH="$HOME/miniconda3"
GUIDANCE_FILE="${GUIDANCE_FILE:-$BASE_DIR/run_scripts/eval/vlm_gate_guidance.txt}"
OUTPUT_BASE="${OUTPUT_BASE:-$BASE_DIR/output/robocasa/ratio_gate_r2}"
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NV="$CONDA_PATH/envs/quant_gate/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"

cleanup() { kill ${SPID:-} ${JPID:-} 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT

# ---------- GPU0: base GR00T policy server (head=main) ----------
SERVER_LOG="$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$SERVER_LD" PYTHONUNBUFFERED=1 \
"$CONDA_PATH/envs/quant_gate/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT_DIR" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head main > "$SERVER_LOG" 2>&1 &
SPID=$!

# ---------- GPU1: Cosmos3-Nano judge (transformers reasoner) ----------
JUDGE_LOG="$OUTPUT_BASE/judge-$SLURM_ARRAY_TASK_ID.log"
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts" \
"$COSMOS_VENV/bin/python" -u "$BASE_DIR/scripts/vlm_gate_cosmos.py" --serve \
    --model "$COSMOS_MODEL" --port $JUDGE_PORT --host 127.0.0.1 > "$JUDGE_LOG" 2>&1 &
JPID=$!

wait_ready() { # <log> <pat> <pid> <label>
    for i in $(seq 1 180); do
        grep -qE "$2" "$1" 2>/dev/null && return 0
        kill -0 "$3" 2>/dev/null || { echo "[ERR] $4 died"; tail -40 "$1"; return 1; }
        sleep 5
    done
    echo "[ERR] $4 not ready"; tail -40 "$1"; return 1
}
wait_ready "$SERVER_LOG" "Server is ready" "$SPID" "GR00T server" || exit 1
wait_ready "$JUDGE_LOG" "JUDGE READY" "$JPID" "Cosmos judge" || exit 1
echo "[i] both servers ready (array $SLURM_ARRAY_TASK_ID)"

# ---------- 상한이 1.0 인 태스크는 게이트가 할 일이 없어 뺀다 ----------
TASKS_ALL=(OpenDoubleDoor PnPCounterToSink CloseDoubleDoor CoffeeServeMug)
SELECTED=("${TASKS_ALL[$SLURM_ARRAY_TASK_ID]}")
[ ${#SELECTED[@]} -eq 0 ] && { echo "태스크가 안 잡혔다"; exit 1; }
echo "[i] 태스크: ${SELECTED[*]} · 에피 $N_EPISODES"

# ---------- run the 3 tasks as parallel clients against the shared servers ----------
MAIN_PIDS=()
for TASK in "${SELECTED[@]}"; do
    ODIR="$OUTPUT_BASE/$TASK"; mkdir -p "$ODIR"
    PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
        "$CONDA_PATH/envs/quant_gate_eval/bin/python" -u \
        "$BASE_DIR/scripts/robocasa_service_compress.py" \
        --port $PORT --host localhost --env_name "$TASK" \
        --video_dir "$ODIR" --seed 42 --n_episodes $N_EPISODES \
        --max_episode_steps $MAX_STEPS --generative_textures \
        --compress-k $K \
        --judge-url "http://127.0.0.1:$JUDGE_PORT" --judge-threshold $TAU \
        --judge-checks phase9_checks_v7 \
        --task-ceilings "$BASE_DIR/analysis/ratio_prompt/task_ceilings.json" \
        --ceiling-floor 1.0 --ceiling-mode frac \
        --conf-quantiles "$BASE_DIR/analysis/ratio_prompt/conf_quantiles_rollout.json" \
        --judge-guidance "@$GUIDANCE_FILE" --gate-subchunk 8 \
        >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done
for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
# **산출물로 판정한다.** 클라이언트가 0개 돌아도 잡은 COMPLETED 로 끝난다 --
# 실제로 그렇게 56초 만에 끝난 적이 있다.
rc=0
for TASK in "${SELECTED[@]}"; do
    P="$OUTPUT_BASE/$TASK/prediction.txt"
    N=$(wc -l < "$P" 2>/dev/null || echo 0)
    echo "[i] $TASK: $N 에피"
    if [ "$N" -lt "$N_EPISODES" ]; then
        echo "실패: $TASK 이 $N/$N_EPISODES"; tail -15 "$OUTPUT_BASE/$TASK/eval-$SLURM_ARRAY_TASK_ID.log" 2>/dev/null; rc=1
    fi
done
echo "[i] Array $SLURM_ARRAY_TASK_ID done."
exit $rc
