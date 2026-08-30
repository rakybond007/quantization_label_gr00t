#!/bin/bash
# Smoke: zero-shot COSMOS-gated action quantization on robocasa (base GR00T-N1.5).
# Same as _smoke_robocasa_vlm_gate.sh but the judge is nvidia/Cosmos3-Nano via the
# transformers reasoner path (cosmos_judge_venv), NOT Gemma.
#   GPU0: base GR00T policy server (quant_gate, head=main, shared ckpt)
#   GPU1: Cosmos3-Nano judge (cosmos_judge_venv, vlm_gate_cosmos.py --serve)
#   client: robocasa_service_compress.py (quant_gate_eval) --judge-url -> cosmos
# NOT for sbatch. Run inside the 2-GPU interactive alloc (srun --overlap).
# Usage: _smoke_robocasa_cosmos_gate.sh [task] [episodes] [K] [max_steps]
set -u
TASK="${1:-OpenDrawer}"
N_EPISODES="${2:-1}"
K="${3:-2}"
MAX_STEPS="${4:-300}"

BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
CONDA="$HOME/miniconda3"
COSMOS_VENV="$HOME/quantization_agent_workspace/cosmos_judge_venv"
# Base GR00T checkpoint stays on the shared tree (no copy).
CKPT="$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
PORT="${PORT:-10921}"
JUDGE_PORT="${JUDGE_PORT:-8124}"
OUT="$BASE_DIR/output/robocasa/_smoke_cosmos_gate/$TASK"
mkdir -p "$OUT"; cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NV="$CONDA/envs/quant_gate/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"

timeout 10 fuser -k "${PORT}/tcp" "${JUDGE_PORT}/tcp" 2>/dev/null || true; sleep 1
cleanup() { kill ${SPID:-} ${JPID:-} 2>/dev/null; timeout 10 fuser -k "${PORT}/tcp" "${JUDGE_PORT}/tcp" 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT INT TERM

# ---------- GPU0: base GR00T policy server (head=main) ----------
SERVER_LOG="$OUT/server.log"
echo "[smoke] (GPU0) serving base GR00T on port $PORT"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$SERVER_LD" PYTHONUNBUFFERED=1 \
"$CONDA/envs/quant_gate/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head main > "$SERVER_LOG" 2>&1 &
SPID=$!

# ---------- GPU1: Cosmos3-Nano judge ----------
JUDGE_LOG="$OUT/judge.log"
echo "[smoke] (GPU1) serving Cosmos3-Nano judge on port $JUDGE_PORT"
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts" \
"$COSMOS_VENV/bin/python" -u "$BASE_DIR/scripts/vlm_gate_cosmos.py" --serve \
    --model nvidia/Cosmos3-Nano --port $JUDGE_PORT --host 127.0.0.1 > "$JUDGE_LOG" 2>&1 &
JPID=$!

wait_ready() { # <logfile> <pattern> <pid> <label>
    for i in $(seq 1 150); do
        grep -qE "$2" "$1" 2>/dev/null && return 0
        kill -0 "$3" 2>/dev/null || { echo "[ERR] $4 died"; tail -40 "$1"; return 1; }
        sleep 5
    done
    echo "[ERR] $4 not ready"; tail -40 "$1"; return 1
}
wait_ready "$SERVER_LOG" "Server is ready" "$SPID" "GR00T server" || exit 1
echo "[smoke] GR00T server ready"
wait_ready "$JUDGE_LOG" "JUDGE READY" "$JPID" "Cosmos judge" || exit 1
echo "[smoke] Cosmos judge ready"

echo "[smoke] running cosmos-gated eval: task=$TASK ep=$N_EPISODES K=$K"
PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
"$CONDA/envs/quant_gate_eval/bin/python" -u \
    "$BASE_DIR/scripts/robocasa_service_compress.py" \
    --port $PORT --host 127.0.0.1 \
    --env_name "$TASK" --video_dir "$OUT" \
    --seed 42 --n_episodes "$N_EPISODES" --max_episode_steps "$MAX_STEPS" \
    --generative_textures --compress-k "$K" \
    --judge-url "http://127.0.0.1:$JUDGE_PORT" \
    2>&1 | tee "$OUT/eval.log"
RC=${PIPESTATUS[0]}
echo "[smoke] done rc=$RC"
echo "[smoke] prediction.txt:"; tail -8 "$OUT/prediction.txt" 2>/dev/null
exit $RC
