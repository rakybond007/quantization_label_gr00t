#!/bin/bash
# Smoke: original AAC (paper) dynamic-horizon on plain GR00T-N1.5 baseline,
# 1 robocasa task × 2 episodes. Server runs --head main_n --n_samples 10.
# Verifies wiring of new scripts/robocasa_service_aac_original.py.
set -u
CKPT_DIR="$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
PORT=9871
N_EPISODES=2
N_SAMPLES=10
TASK="TurnOnSinkFaucet"

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
OUTPUT_BASE="$BASE_DIR/output/robocasa/_smoke_baseline_aac_original"
mkdir -p "$OUTPUT_BASE/$TASK"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

SERVER_LOG="$OUTPUT_BASE/server.log"
echo "[i] Starting server (gr00t env), port=$PORT, head=main_n, N=$N_SAMPLES, ckpt=$CKPT_DIR"
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT_DIR" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head main_n --n_samples $N_SAMPLES \
    > "$SERVER_LOG" 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 60); do
    if grep -q "Server is ready" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died early"; tail -40 "$SERVER_LOG"; exit 1; fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then echo "[ERR] server not ready in 5 min"; tail -40 "$SERVER_LOG"; kill "$SPID" 2>/dev/null; exit 1; fi
echo "[i] Server ready. Running 2-ep eval on $TASK."

ODIR="$OUTPUT_BASE/$TASK"
PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
"$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
    "$BASE_DIR/scripts/robocasa_service_aac_original.py" \
    --port $PORT --host localhost \
    --env_name "$TASK" \
    --video_dir "$ODIR" --seed 42 --n_episodes $N_EPISODES \
    --max_episode_steps 1500 --generative_textures \
    --aac-xi 1 \
    > "$ODIR/eval.log" 2>&1
RC=$?

kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
echo "[i] Eval exit rc=$RC"
if [ -f "$ODIR/prediction.txt" ]; then
    echo "[i] prediction.txt contents:"
    cat "$ODIR/prediction.txt"
fi
echo "[i] SMOKE_DONE rc=$RC"
exit $RC
