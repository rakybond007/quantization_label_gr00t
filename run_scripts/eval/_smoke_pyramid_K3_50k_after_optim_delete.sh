#!/bin/bash
# Smoke verify: load pyramid_K3 50k ckpt AFTER deleting optimizer.pt/rng/scheduler,
# serve + run 2 eps of one robocasa task. If this works, model.safetensors alone
# is sufficient for eval/serving and the resume-state files are safe to delete
# across all our self-trained ckpts.
set -u
CKPT_DIR="$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_moe_pyramid_K3_raw16_m8_m4_b_only_no_metaq_no_balance/checkpoint-50000"
PORT=9870
N_EPISODES=2
TASK="TurnOnSinkFaucet"

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
OUTPUT_BASE="$BASE_DIR/output/robocasa/_smoke_pyramid_K3_50k_after_optim_delete"
mkdir -p "$OUTPUT_BASE/$TASK"
cd "$BASE_DIR"

echo "[i] Confirming optimizer/rng/scheduler are NOT present:"
ls "$CKPT_DIR" | grep -E "(optimizer|rng_state|scheduler)" && { echo "[ERR] resume files still present"; exit 1; }
echo "[i] OK — only model + config present."
ls -la "$CKPT_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

SERVER_LOG="$OUTPUT_BASE/server.log"
echo "[i] Starting server (gr00t env), port=$PORT, ckpt=$CKPT_DIR"
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_fair_moe.py" --server \
    --port $PORT --model_path "$CKPT_DIR" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe --discrete-action-dims 6 11 \
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
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
    "$BASE_DIR/scripts/robocasa_service_moe.py" \
    --port $PORT --host localhost \
    --env_name "$TASK" \
    --video_dir "$ODIR" --seed 42 --n_episodes $N_EPISODES \
    --max_episode_steps 1500 --generative_textures \
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
