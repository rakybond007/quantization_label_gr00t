#!/bin/bash
# Eval smoke for the metaq model: load the 5-step smoke checkpoint via
# inference_service_metaq.py + robocasa_service_moe.py for 1 task × 2 episodes.
# Catches state_dict mismatch and meta_q backbone wrap regressions before
# committing to a full training run.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/_smoke_metaq_robocasa_no_d/checkpoint-5"
OUT="$BASE_DIR/output/robocasa/_smoke_eval_metaq"
PORT=9456
mkdir -p "$OUT"
cd "$BASE_DIR"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "[$(date '+%T')] === EVAL SMOKE (metaq): server starting ==="
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_metaq.py" --server \
    --port $PORT \
    --model_path "$CKPT" \
    --data_config single_panda_gripper \
    --embodiment_tag new_embodiment \
    --denoising_steps 4 \
    --head moe \
    --discrete-action-dims 6 11 \
    > "$OUT/server.log" 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 60); do
    grep -q "Server is ready" "$OUT/server.log" 2>/dev/null && READY=1 && break
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
tail -15 "$OUT/client.log"
[ $RC -eq 0 ] && echo "=== EVAL SMOKE PASS ===" || echo "=== EVAL SMOKE FAIL ==="
