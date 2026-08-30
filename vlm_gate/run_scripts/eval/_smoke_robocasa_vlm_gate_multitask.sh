#!/bin/bash
# Probe: how do zero-shot VLM gate confidences look across DIVERSE robocasa tasks?
# Loads base GR00T (GPU0) + VLM judge (GPU1) ONCE, then runs the gated compress
# client over several tasks, logging per-chunk confidence to gate_conf.csv each.
#
# NOT for sbatch. Run on the 2-GPU interactive allocation (tmux 0:0).
# Usage: _smoke_robocasa_vlm_gate_multitask.sh [episodes] [K] [max_steps] [threshold]
set -u
N_EPISODES="${1:-2}"
K="${2:-2}"
MAX_STEPS="${3:-400}"
THRESH="${4:-0.5}"
JUDGE_MODEL="google/gemma-4-12b-it"
GUIDANCE="${GUIDANCE:-@$HOME/quantization_agent_workspace/vlm_gate/run_scripts/eval/vlm_gate_guidance.txt}"

# Task set chosen by compression-safety category (see analysis/eval_gate_protocol.py):
# default spans SAFE (compression doesn't hurt -> gate SHOULD quantize) and
# HARMFUL (compression hurts -> gate SHOULD protect). Override with $SMOKE_TASKS.
read -ra TASKS <<< "${SMOKE_TASKS:-CoffeePressButton CloseDrawer TurnOnStove PnPCounterToCab OpenDrawer PnPCounterToSink OpenDoubleDoor CoffeeServeMug}"

BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
CONDA="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
PORT=10930
JUDGE_PORT=8121
ROOT="${OUTPUT_ROOT:-$BASE_DIR/output/robocasa/_smoke_vlm_gate_multi}"
mkdir -p "$ROOT"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NV="$CONDA/envs/quant_gate/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"

cleanup() { kill ${SPID:-} ${JPID:-} 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT INT TERM

# ---------- GPU0: base GR00T policy server ----------
SERVER_LOG="$ROOT/server.log"
echo "[probe] (GPU0) serving base GR00T on port $PORT"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$SERVER_LD" PYTHONUNBUFFERED=1 \
"$CONDA/envs/quant_gate/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head main > "$SERVER_LOG" 2>&1 &
SPID=$!

# ---------- GPU1: VLM judge server ----------
JUDGE_LOG="$ROOT/judge.log"
echo "[probe] (GPU1) serving VLM judge $JUDGE_MODEL on port $JUDGE_PORT"
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
"$CONDA/envs/vlm_judge/bin/python" -u "$BASE_DIR/scripts/vlm_gate.py" --serve \
    --model "$JUDGE_MODEL" --port $JUDGE_PORT --host 127.0.0.1 > "$JUDGE_LOG" 2>&1 &
JPID=$!

wait_ready() { # <log> <pat> <pid> <label>
    for i in $(seq 1 150); do
        grep -qE "$2" "$1" 2>/dev/null && return 0
        kill -0 "$3" 2>/dev/null || { echo "[ERR] $4 died"; tail -40 "$1"; return 1; }
        sleep 5
    done
    echo "[ERR] $4 not ready"; tail -40 "$1"; return 1
}
wait_ready "$SERVER_LOG" "Server is ready" "$SPID" "GR00T server" || exit 1
echo "[probe] GR00T server ready"
wait_ready "$JUDGE_LOG" "JUDGE READY" "$JPID" "VLM judge" || exit 1
echo "[probe] VLM judge ready"

# ---------- loop tasks ----------
for TASK in "${TASKS[@]}"; do
    ODIR="$ROOT/$TASK"; mkdir -p "$ODIR"
    echo "==================== $TASK ===================="
    PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
    "$CONDA/envs/quant_gate_eval/bin/python" -u \
        "$BASE_DIR/scripts/robocasa_service_compress.py" \
        --port $PORT --host 127.0.0.1 \
        --env_name "$TASK" --video_dir "$ODIR" \
        --seed 42 --n_episodes "$N_EPISODES" --max_episode_steps "$MAX_STEPS" \
        --generative_textures --compress-k "$K" \
        --judge-url "http://127.0.0.1:$JUDGE_PORT" --judge-threshold "$THRESH" \
        --judge-guidance "$GUIDANCE" --gate-subchunk "${GATE_SUBCHUNK:-0}" \
        > "$ODIR/eval.log" 2>&1
    echo "[probe] $TASK done:"; tail -5 "$ODIR/prediction.txt" 2>/dev/null
done

echo "[probe] ALL DONE"
