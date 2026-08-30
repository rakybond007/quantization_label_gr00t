#!/bin/bash
# Debug: take baseline (proven-working) ckpt, force inference-time block-last
# downsampling at factor=2. Mimics what MoE m8 routing path would emit.
# - If robot moves + task succeeds: robot CAN handle 2x compression → MoE
#   head=moe path is the bug.
# - If robot frozen: dexjoco controller fundamentally can't handle compressed
#   abs execution → MoE compression speedup is infeasible on this robot.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"; CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_baseline/checkpoint-60000"
DEXJOCO_REPO="$HOME/multigpu_workspace/external_dependencies/dexjoco"
LOG_DIR="$BASE_DIR/analysis/_smoke/dexjoco_baseline_compress2"; PORT=8098
mkdir -p "$LOG_DIR/hammer_nail"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"; export NO_ALBUMENTATIONS_UPDATE=1
"$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/serve_policy_dexjoco.py" \
    --port "$PORT" --model-path "$CKPT" \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --head main --denoising-steps 4 --compress-factor 2 \
    > "$LOG_DIR/server.log" 2>&1 &
SPID=$!
for i in $(seq 1 36); do
    if grep -qE "server listening|Listening on|serve_forever|local_ip" "$LOG_DIR/server.log" 2>/dev/null; then break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "COMPRESS2_SMOKE FAIL (server died)"; tail -30 "$LOG_DIR/server.log"; exit 1; fi
    sleep 5
done
echo "[i] server ready"
MUJOCO_GL=egl "$CONDA_PATH/envs/dexjoco/bin/python" -u "$BASE_DIR/scripts/dexjoco_eval_gr00t.py" \
    --config "$DEXJOCO_REPO/configs/rand_obj/hammer_nail.yaml" \
    --port "$PORT" --host 127.0.0.1 --episodes 2 --output "$LOG_DIR/hammer_nail" \
    > "$LOG_DIR/hammer_nail/eval.log" 2>&1
RC=$?
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
sr=$(grep "Success rate" "$LOG_DIR/hammer_nail/eval.log" | tail -1)
echo "COMPRESS2_SMOKE PASS  $sr  (rc=$RC)"
