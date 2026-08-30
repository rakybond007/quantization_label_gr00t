#!/bin/bash
# Real-inference smoke: start K=4 pyramid head=moe server + run 1 episode of
# CloseDrawer (fastest task) with the actual robocasa client. This DOES hit the
# decoders[picked] code path that the earlier server-ready smoke missed.
# Interactive GPU node only (tmux 0:0).
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_moe_pyramid_K4_raw16_m8_m4_m2_b_only_no_metaq_no_balance/checkpoint-60000"
LOG_DIR="$BASE_DIR/analysis/_smoke/pyramid_K4_real"
PORT=9788
mkdir -p "$LOG_DIR" "$LOG_DIR/CloseDrawer"

# Server (gr00t env)
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
export NO_ALBUMENTATIONS_UPDATE=1
"$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_fair_moe.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe --discrete-action-dims 6 11 \
    > "$LOG_DIR/server.log" 2>&1 &
SPID=$!
for i in $(seq 1 60); do
    if grep -q "Server is ready" "$LOG_DIR/server.log" 2>/dev/null; then break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "K4PYR_REAL_SMOKE FAIL (server died at boot)"; tail -20 "$LOG_DIR/server.log"; exit 1; fi
    sleep 5
done
echo "[i] server ready"

# Client (robocasa_gr00t env), 1 episode CloseDrawer
"$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u "$BASE_DIR/scripts/robocasa_service_moe.py" \
    --port $PORT --host localhost \
    --env_name CloseDrawer \
    --video_dir "$LOG_DIR/CloseDrawer" --seed 42 --n_episodes 1 \
    --max_episode_steps 1500 --generative_textures \
    > "$LOG_DIR/CloseDrawer/eval.log" 2>&1
RC=$?

kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null

if [ $RC -eq 0 ] && grep -q "^episode " "$LOG_DIR/CloseDrawer/prediction.txt" 2>/dev/null; then
    line=$(head -1 "$LOG_DIR/CloseDrawer/prediction.txt")
    if grep -qE "AttributeError|Traceback" "$LOG_DIR/server.log"; then
        echo "K4PYR_REAL_SMOKE FAIL (server traceback)"
        grep -E "AttributeError|Traceback" "$LOG_DIR/server.log" | head -3
    else
        echo "K4PYR_REAL_SMOKE PASS ($line)"
    fi
else
    echo "K4PYR_REAL_SMOKE FAIL (client rc=$RC, no prediction)"
    tail -15 "$LOG_DIR/server.log"
fi
