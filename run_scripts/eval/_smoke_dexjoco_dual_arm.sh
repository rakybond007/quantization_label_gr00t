#!/bin/bash
# Smoke eval for DexJoCo dual-arm: serve a checkpoint + run a few episodes on one
# bimanual task. NOT for sbatch — run directly on an interactive GPU (tmux 0:0).
#
# Usage: _smoke_dexjoco_dual_arm.sh <baseline|moe> <ckpt_path> <port> [task] [episodes]
set -u
MODE="${1:?mode: baseline|moe}"
CKPT="${2:?ckpt path}"
PORT="${3:-9700}"
TASK="${4:-bimanual_assembly}"
N_EPISODES="${5:-2}"

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
DEXJOCO_REPO="$HOME/multigpu_workspace/external_dependencies/dexjoco"
CONFIG="$DEXJOCO_REPO/configs/rand_obj/$TASK.yaml"
OUTPUT_BASE="$BASE_DIR/output/dexjoco/_smoke_dual_arm_${MODE}/$TASK"
mkdir -p "$OUTPUT_BASE"

if [ "$MODE" = "moe" ]; then
    HEAD_ARGS="--head moe --moe-stochastic --moe-confidence-threshold 0.7"
else
    HEAD_ARGS="--head main"
fi

NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

SERVER_LOG="$OUTPUT_BASE/server.log"
echo "[smoke] serving $MODE ckpt=$CKPT port=$PORT task=$TASK"
LD_LIBRARY_PATH="$SERVER_LD" NO_ALBUMENTATIONS_UPDATE=1 \
"$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/serve_policy_dexjoco.py" \
    --port "$PORT" --model-path "$CKPT" \
    --data-config dexjoco_dual_arm_multi_horizon --embodiment-tag new_embodiment \
    $HEAD_ARGS --denoising-steps 4 \
    > "$SERVER_LOG" 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 60); do
    if grep -qE "server listening|Listening on|serve_forever|local_ip" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died"; tail -40 "$SERVER_LOG"; exit 1; fi
    sleep 5
done
[ "$READY" -ne 1 ] && { echo "[ERR] server not ready"; tail -40 "$SERVER_LOG"; kill "$SPID" 2>/dev/null; exit 1; }
echo "[smoke] server ready, launching client"

MUJOCO_GL=egl \
"$CONDA_PATH/envs/dexjoco/bin/python" -u "$BASE_DIR/scripts/dexjoco_eval_gr00t_sync.py" \
    --config "$CONFIG" --port "$PORT" --host 127.0.0.1 \
    --episodes "$N_EPISODES" --max-episode-steps 600 \
    --output "$OUTPUT_BASE" \
    2>&1 | tee "$OUTPUT_BASE/eval.log"
RC=${PIPESTATUS[0]}

kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
echo "[smoke] $MODE done rc=$RC"
exit $RC
