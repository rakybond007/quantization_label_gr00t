#!/bin/bash
# Quick smoke: load pyramid K3 ckpt + verify inference_service_fair_moe server starts.
# Interactive GPU only (tmux 0:0). Server stops after first "ready" or 3-min timeout.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT_DIR="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_moe_pyramid_K3_raw16_m8_m4_b_only_no_metaq_no_balance/checkpoint-60000"
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1
LOG="$BASE_DIR/analysis/_smoke/pyramid_K3_server.log"
"$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_fair_moe.py" --server \
    --port 9777 --model_path "$CKPT_DIR" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe_selective --n_samples 10 \
    --discrete-action-dims 6 11 > "$LOG" 2>&1 &
PID=$!
READY=0
for i in $(seq 1 36); do
    if grep -q "Server is ready" "$LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$PID" 2>/dev/null; then echo "PYRAMID_SMOKE FAIL (server died)"; tail -20 "$LOG"; exit 1; fi
    sleep 5
done
kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
if [ "$READY" -eq 1 ]; then echo "PYRAMID_SMOKE PASS"; else echo "PYRAMID_SMOKE FAIL (timeout)"; tail -10 "$LOG"; fi
