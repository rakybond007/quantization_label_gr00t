#!/bin/bash
# Smoke: trajlog version - 1 task (TurnOnSinkFaucet), 1 ep, baseline ckpt.
# Verifies that trajectory JSONL is dumped per step.
set -u
CKPT_DIR="$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
PORT=10310
N_EPISODES=1
TASK="TurnOnSinkFaucet"
SEED=42

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
OUTPUT_BASE="$BASE_DIR/output/robocasa/_smoke_trajlog/baseline_${TASK}_seed${SEED}"
TRAJ_DIR="$OUTPUT_BASE/traj"
mkdir -p "$OUTPUT_BASE"
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
echo "[i] server ready"

PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
"$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
    "$BASE_DIR/scripts/robocasa_service_trajlog.py" --client \
    --port $PORT --host localhost \
    --env_name "$TASK" \
    --video_dir "$OUTPUT_BASE" --trajlog_dir "$TRAJ_DIR" \
    --seed $SEED --n_episodes $N_EPISODES \
    --max_episode_steps 1500 --generative_textures \
    > "$OUTPUT_BASE/eval.log" 2>&1
RC=$?
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null

echo "[i] eval rc=$RC"
echo "--- trajlog files ---"
ls -la "$TRAJ_DIR" 2>/dev/null
echo "--- first 3 lines of traj_ep00.jsonl ---"
head -3 "$TRAJ_DIR/traj_ep00.jsonl" 2>/dev/null
echo "--- traj_ep00 line count ---"
wc -l "$TRAJ_DIR/traj_ep00.jsonl" 2>/dev/null
echo "SMOKE_TRAJLOG_DONE rc=$RC"
exit $RC
