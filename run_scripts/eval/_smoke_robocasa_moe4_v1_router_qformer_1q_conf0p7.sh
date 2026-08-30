#!/bin/bash
# Smoke: MoE4 v1 K=4 b_only no_metaq balance ON + router=qformer_1q + conf0p7.
# 1 task (TurnOnSinkFaucet), 2 episodes. Verifies HF download + qformer wiring + conf0p7.
set -u
HF_REPO=prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-router-qformer-1q-60k
PORT=10080
N_EPISODES=2
TASK="TurnOnSinkFaucet"

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$("$CONDA_PATH/envs/gr00t/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_REPO', repo_type='model'))")"
OUTPUT_BASE="$BASE_DIR/output/robocasa/_smoke_moe4_v1_router_qformer_1q_conf0p7"
mkdir -p "$OUTPUT_BASE/$TASK"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

SERVER_LOG="$OUTPUT_BASE/server.log"
echo "[i] Starting server, ckpt=$CKPT, port=$PORT"
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_fair_moe.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe --discrete-action-dims 6 11 \
    --moe-stochastic --moe-confidence-threshold 0.7 \
    > "$SERVER_LOG" 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 60); do
    if grep -q "Server is ready" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died"; tail -40 "$SERVER_LOG"; exit 1; fi
    sleep 5
done
[ "$READY" -ne 1 ] && { echo "[ERR] server not ready"; tail -40 "$SERVER_LOG"; kill "$SPID" 2>/dev/null; exit 1; }
echo "[i] Server ready"

ODIR="$OUTPUT_BASE/$TASK"
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
    "$BASE_DIR/scripts/robocasa_service_moe.py" \
    --port $PORT --host localhost \
    --env_name "$TASK" \
    --video_dir "$ODIR" --seed 42 --n_episodes $N_EPISODES \
    --max_episode_steps 1500 --generative_textures \
    > "$ODIR/eval.log" 2>&1
RC=$?
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
echo "[i] eval rc=$RC"
[ -f "$ODIR/prediction.txt" ] && cat "$ODIR/prediction.txt"
echo "SMOKE_QFORMER_1Q_DONE rc=$RC"
exit $RC
