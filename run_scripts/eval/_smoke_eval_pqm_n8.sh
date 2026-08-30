#!/bin/bash
# Eval smoke for MoE4 per_quad+u8: load the 5-step smoke checkpoint, start the
# inference server, run robocasa_service_moe for 1 task × 2 episodes. Goal is to
# catch state_dict mismatch / per_quad inference / chunk-assembly errors BEFORE
# committing a full training run. Not a quality test — just a no-crash check.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/_smoke_moe_per_quad_mask_n8_robocasa/checkpoint-5"
OUT="$BASE_DIR/output/robocasa/_smoke_eval_pqm_n8"
PORT=9123
mkdir -p "$OUT"
cd "$BASE_DIR"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "[$(date '+%T')] === EVAL SMOKE: starting server on smoke ckpt ==="
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT \
    --model_path "$CKPT" \
    --data_config single_panda_gripper \
    --embodiment_tag new_embodiment \
    --denoising_steps 4 \
    --head moe \
    --discrete-action-dims 6 11 \
    > "$OUT/server.log" 2>&1 &
SPID=$!
echo "[$(date '+%T')] server PID=$SPID, waiting for ready..."
READY=0
for i in $(seq 1 60); do
    if grep -q "Server is ready" "$OUT/server.log" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died early"; break; fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then
    echo "[$(date '+%T')] === SERVER FAILED — log tail ==="
    tail -25 "$OUT/server.log"
    kill "$SPID" 2>/dev/null
    exit 1
fi
echo "[$(date '+%T')] server ready. running client (1 task × 2 ep)..."
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
    "$BASE_DIR/scripts/robocasa_service_moe.py" \
    --port $PORT --host localhost \
    --env_name CloseDrawer \
    --video_dir "$OUT/CloseDrawer" \
    --seed 42 \
    --n_episodes 2 \
    --max_episode_steps 300 \
    --generative_textures \
    > "$OUT/client.log" 2>&1
RC=$?
kill "$SPID" 2>/dev/null
echo "[$(date '+%T')] === client exit code: $RC ==="
echo "--- client.log tail ---"
tail -20 "$OUT/client.log"
[ $RC -eq 0 ] && echo "[$(date '+%T')] === EVAL SMOKE PASS ===" || echo "[$(date '+%T')] === EVAL SMOKE FAIL ==="
