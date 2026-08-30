#!/bin/bash
# Server (gr00t env, head=moe) + client (dexjoco env) probe.
# Reports router-pick distribution per task.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"; CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_moe4_v1_no_balance/checkpoint-60000"
LOG_DIR="$BASE_DIR/analysis/_smoke/dexjoco_pick_probe"; PORT=8103
mkdir -p "$LOG_DIR"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"; export NO_ALBUMENTATIONS_UPDATE=1
"$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/serve_policy_dexjoco.py" \
    --port "$PORT" --model-path "$CKPT" \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --head moe --denoising-steps 4 > "$LOG_DIR/server.log" 2>&1 &
SPID=$!
for i in $(seq 1 36); do
    if grep -qE "server listening|Listening on|serve_forever|local_ip" "$LOG_DIR/server.log" 2>/dev/null; then break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "PICK_PROBE FAIL (server died)"; tail -20 "$LOG_DIR/server.log"; exit 1; fi
    sleep 5
done
echo "[i] server ready"
MUJOCO_GL=egl "$CONDA_PATH/envs/dexjoco/bin/python" -u "$BASE_DIR/scripts/dexjoco_pick_probe_client.py" \
    --port "$PORT" --host 127.0.0.1 --n-queries 12 \
    > "$LOG_DIR/probe.log" 2>&1
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
echo "PICK_PROBE_DONE"
echo ""
echo "=== distribution ==="
grep -E "^(task|---|hammer|click|pick|pinch|fold|water|TOTAL|  k=)" "$LOG_DIR/probe.log"
