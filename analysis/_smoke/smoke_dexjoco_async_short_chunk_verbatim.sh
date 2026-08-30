#!/bin/bash
# Smoke: verify async client + new --short-chunk-verbatim-threshold flag handles
# MoE short chunks (8-step m8, 4-step m4) without freezing the robot.
# Compares: (a) async vanilla (expected: frozen → 0/2), (b) async + threshold 12 (expected: moves → 2/2).
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"; CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_moe4_v1_balance/checkpoint-60000"
DEXJOCO_REPO="$HOME/multigpu_workspace/external_dependencies/dexjoco"
ROOT="$BASE_DIR/analysis/_smoke/dexjoco_async_short_chunk_verbatim"
PORT_A=8200; PORT_B=8201
mkdir -p "$ROOT/vanilla" "$ROOT/threshold12"

unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

start_server() {
    local PORT=$1 LOG=$2
    "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/serve_policy_dexjoco.py" \
        --port "$PORT" --model-path "$CKPT" \
        --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
        --head moe --denoising-steps 4 --moe-stochastic > "$LOG" 2>&1 &
    SPID=$!
    for i in $(seq 1 60); do
        if grep -qE "server listening|Listening on|serve_forever|local_ip" "$LOG" 2>/dev/null; then return 0; fi
        if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died"; tail -20 "$LOG"; exit 1; fi
        sleep 5
    done
    echo "[ERR] server timed out"; tail -20 "$LOG"; kill "$SPID" 2>/dev/null; exit 1
}

run_eval() {
    local PORT=$1 OUT=$2 THRESH=$3 LOG=$4
    EXTRA=""
    if [ "$THRESH" -gt 0 ]; then EXTRA="--short-chunk-verbatim-threshold $THRESH"; fi
    MUJOCO_GL=egl "$CONDA_PATH/envs/dexjoco/bin/python" -u \
        -m dexjoco_openpi_client.eval_dexjoco_openpi \
        --config "$DEXJOCO_REPO/configs/rand_obj/hammer_nail.yaml" \
        --port "$PORT" --host 127.0.0.1 --episodes 2 --seed 0 \
        --output "$OUT" $EXTRA > "$LOG" 2>&1
}

echo "=== (a) async vanilla (no threshold) ==="
start_server "$PORT_A" "$ROOT/vanilla/server.log"; SPID_A=$SPID
run_eval "$PORT_A" "$ROOT/vanilla" 0 "$ROOT/vanilla/eval.log"
sr_a=$(grep -E "Success!|Failed" "$ROOT/vanilla/eval.log" | awk '{print}' | tr '\n' ' ')
echo "  result: $sr_a"
kill "$SPID_A" 2>/dev/null; wait "$SPID_A" 2>/dev/null

echo "=== (b) async + --short-chunk-verbatim-threshold 12 ==="
start_server "$PORT_B" "$ROOT/threshold12/server.log"; SPID_B=$SPID
run_eval "$PORT_B" "$ROOT/threshold12" 12 "$ROOT/threshold12/eval.log"
sr_b=$(grep -E "Success!|Failed" "$ROOT/threshold12/eval.log" | awk '{print}' | tr '\n' ' ')
echo "  result: $sr_b"
kill "$SPID_B" 2>/dev/null; wait "$SPID_B" 2>/dev/null

echo "=== SUMMARY ==="
echo "vanilla:      $sr_a"
echo "threshold=12: $sr_b"
echo "SHORT_CHUNK_SMOKE_DONE"
