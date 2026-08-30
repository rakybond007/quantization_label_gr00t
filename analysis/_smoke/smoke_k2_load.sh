#!/bin/bash
# Smoke: K=2 raw16+merged8 no_balance HF ckpt load + server start (head=moe_selective N=10).
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
HF_REPO=prehj/GR00T-N1.5-robocasa-moe-v1-K2-raw16-merged8-b-only-no-metaq-no-balance-60k
CKPT=$("$CONDA_PATH/envs/gr00t/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_REPO', repo_type='model'))")
echo "[i] CKPT=$CKPT"
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1
LOG="$BASE_DIR/analysis/_smoke/k2_server.log"
"$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_fair_moe.py" --server \
    --port 9778 --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe_selective --n_samples 10 \
    --discrete-action-dims 6 11 > "$LOG" 2>&1 &
PID=$!
READY=0
for i in $(seq 1 36); do
    if grep -q "Server is ready" "$LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$PID" 2>/dev/null; then echo "K2_SMOKE FAIL (died)"; tail -20 "$LOG"; exit 1; fi
    sleep 5
done
kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
[ "$READY" -eq 1 ] && echo "K2_SMOKE PASS" || { echo "K2_SMOKE FAIL (timeout)"; tail -10 "$LOG"; }
