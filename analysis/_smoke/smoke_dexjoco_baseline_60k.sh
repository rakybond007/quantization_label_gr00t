#!/bin/bash
# Smoke: dexjoco baseline checkpoint-60000 + 1 episode each on hammer_nail
# (normal front+wrist) and click_mouse (ego_right camera, modality.json patched).
# Verifies eval pipeline end-to-end on the actual final ckpt before submitting
# the full 6-task array eval. Interactive GPU only (tmux 0:0).
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_baseline/checkpoint-60000"
DEXJOCO_REPO="$HOME/multigpu_workspace/external_dependencies/dexjoco"
LOG_DIR="$BASE_DIR/analysis/_smoke/dexjoco_baseline_60k"
PORT=8095
mkdir -p "$LOG_DIR/hammer_nail" "$LOG_DIR/click_mouse"

# --- server (gr00t env) -------
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
export NO_ALBUMENTATIONS_UPDATE=1
"$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/serve_policy_dexjoco.py" \
    --port "$PORT" --model-path "$CKPT" \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --head main --denoising-steps 4 \
    > "$LOG_DIR/server.log" 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 36); do
    if grep -qE "server listening|Listening on|serve_forever|local_ip" "$LOG_DIR/server.log" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "DEX60K_SMOKE FAIL (server died)"; tail -20 "$LOG_DIR/server.log"; exit 1; fi
    sleep 5
done
[ "$READY" -ne 1 ] && { echo "DEX60K_SMOKE FAIL (server not ready)"; tail -30 "$LOG_DIR/server.log"; kill "$SPID" 2>/dev/null; exit 1; }
echo "[i] server ready"

# --- client: hammer_nail (1 ep, normal cams) ---
MUJOCO_GL=egl "$CONDA_PATH/envs/dexjoco/bin/python" -u \
    "$BASE_DIR/scripts/dexjoco_eval_gr00t.py" \
    --config "$DEXJOCO_REPO/configs/rand_obj/hammer_nail.yaml" \
    --port "$PORT" --host 127.0.0.1 --episodes 1 \
    --output "$LOG_DIR/hammer_nail" \
    > "$LOG_DIR/hammer_nail/eval.log" 2>&1
HRC=$?

# --- client: click_mouse (1 ep, ego_right camera via modality.json patch) ---
MUJOCO_GL=egl "$CONDA_PATH/envs/dexjoco/bin/python" -u \
    "$BASE_DIR/scripts/dexjoco_eval_gr00t.py" \
    --config "$DEXJOCO_REPO/configs/rand_obj/click_mouse.yaml" \
    --port "$PORT" --host 127.0.0.1 --episodes 1 \
    --output "$LOG_DIR/click_mouse" \
    > "$LOG_DIR/click_mouse/eval.log" 2>&1
CRC=$?

kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null

server_err=$(grep -cE "Traceback|AttributeError" "$LOG_DIR/server.log")
if [ "$HRC" -eq 0 ] && [ "$CRC" -eq 0 ] && [ "$server_err" -eq 0 ]; then
    h=$(grep "Success rate" "$LOG_DIR/hammer_nail/eval.log" | tail -1)
    c=$(grep "Success rate" "$LOG_DIR/click_mouse/eval.log" | tail -1)
    echo "DEX60K_SMOKE PASS  hammer_nail=[$h]  click_mouse=[$c]"
else
    echo "DEX60K_SMOKE FAIL  HRC=$HRC CRC=$CRC server_err=$server_err"
    tail -15 "$LOG_DIR/server.log"
fi
