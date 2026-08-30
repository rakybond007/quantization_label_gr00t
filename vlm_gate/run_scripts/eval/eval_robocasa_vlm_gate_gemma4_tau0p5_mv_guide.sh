#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=eval_robocasa_vlm_gate_gemma4_tau0p5_mv_guide_24t_50ep_arr8
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=background
#SBATCH --exclude=worker-node100
#SBATCH --array=0-7
#SBATCH --output=out/%A_%a-eval_robocasa_vlm_gate_gemma4.out
#SBATCH --error=out/%A_%a-eval_robocasa_vlm_gate_gemma4.err
#SBATCH --comment="robocasa base GR00T-N1.5 @60k + zero-shot VLM-gated K=2 quantization (Gemma4-12B judge, tau=0.5). 24 tasks x 50 ep. GPU0=GR00T serve, GPU1=judge."

# Per array job: GPU0 = base GR00T policy server, GPU1 = Gemma4 VLM judge.
# Both loaded once; the 3 robocasa tasks for this array id run as parallel
# clients that query the judge per chunk (quantize K=2 when conf>=tau else raw).
set -u
CKPT_DIR="$HOME/quantization_agent_workspace/vlm_gate/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
JUDGE_MODEL="google/gemma-4-12b-it"
# Overridable for interactive single-job simulation; defaults = real eval.
: "${SLURM_ARRAY_TASK_ID:=0}"
: "${SLURM_ARRAY_JOB_ID:=$$}"
# Per-job port offset so concurrent jobs (e.g. two evals at once) never collide;
# array-task id (0-7) stays within the 10-wide slot per job.
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
OUTPUT_BASE="${OUTPUT_BASE:-$BASE_DIR/output/robocasa/vlm_gate_gemma4_tau0p5_mv_guide}"
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NV="$CONDA_PATH/envs/quant_gate/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"

cleanup() { kill ${SPID:-} ${JPID:-} 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT

# ---------- GPU0: base GR00T policy server ----------
SERVER_LOG="$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$SERVER_LD" PYTHONUNBUFFERED=1 \
"$CONDA_PATH/envs/quant_gate/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT_DIR" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head main > "$SERVER_LOG" 2>&1 &
SPID=$!

# ---------- GPU1: Gemma4 VLM judge server ----------
JUDGE_LOG="$OUTPUT_BASE/judge-$SLURM_ARRAY_TASK_ID.log"
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$CONDA_PATH/envs/vlm_judge/bin/python" -u "$BASE_DIR/scripts/vlm_gate.py" --serve \
    --model "$JUDGE_MODEL" --port $JUDGE_PORT --host 127.0.0.1 > "$JUDGE_LOG" 2>&1 &
JPID=$!

# ---------- wait for both servers ----------
wait_ready() { # <log> <pat> <pid> <label>
    for i in $(seq 1 150); do
        grep -qE "$2" "$1" 2>/dev/null && return 0
        kill -0 "$3" 2>/dev/null || { echo "[ERR] $4 died"; tail -40 "$1"; return 1; }
        sleep 5
    done
    echo "[ERR] $4 not ready"; tail -40 "$1"; return 1
}
wait_ready "$SERVER_LOG" "Server is ready" "$SPID" "GR00T server" || exit 1
wait_ready "$JUDGE_LOG" "JUDGE READY" "$JPID" "VLM judge" || exit 1
echo "[i] both servers ready (array $SLURM_ARRAY_TASK_ID)"

# ---------- map array id -> up to 3 tasks ----------
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
        --judge-guidance "@$GUIDANCE_FILE" --gate-subchunk 8 \
        >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done
for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
echo "[i] Array $SLURM_ARRAY_TASK_ID done."
