#!/bin/bash
# Smoke: GR00T-N1.5 baseline + fixed-K=2 compression on robocasa.
# 1 task x 2 episodes. Verifies wiring of robocasa_service_compress.py.
set -u
CKPT_DIR="$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
PORT=10100
N_EPISODES=2
K=2
TASK="TurnOnSinkFaucet"

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
OUTPUT_BASE="$BASE_DIR/output/robocasa/_smoke_baseline_compress_K${K}"
mkdir -p "$OUTPUT_BASE/$TASK"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

SERVER_LOG="$OUTPUT_BASE/server.log"
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT_DIR" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head main \
    > "$SERVER_LOG" 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 60); do
    if grep -q "Server is ready" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died"; tail -30 "$SERVER_LOG"; exit 1; fi
    sleep 5
done
[ "$READY" -ne 1 ] && { echo "[ERR] server not ready"; tail -30 "$SERVER_LOG"; kill "$SPID" 2>/dev/null; exit 1; }
echo "[i] Server ready"

ODIR="$OUTPUT_BASE/$TASK"
PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
"$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
    "$BASE_DIR/scripts/robocasa_service_compress.py" \
    --port $PORT --host localhost \
    --env_name "$TASK" \
    --video_dir "$ODIR" --seed 42 --n_episodes $N_EPISODES \
    --max_episode_steps 1500 --generative_textures \
    --compress-k $K \
    > "$ODIR/eval.log" 2>&1
RC=$?
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
[ -f "$ODIR/prediction.txt" ] && cat "$ODIR/prediction.txt"
echo "SMOKE_ROBOCASA_COMPRESS_K${K}_DONE rc=$RC"
exit $RC
