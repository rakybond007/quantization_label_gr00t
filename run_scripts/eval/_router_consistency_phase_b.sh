#!/bin/bash
# Phase B: forced compression schedule (B: from prior, C: uniform random)
# using baseline ckpt + K=2. Resumable.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
DEXJOCO_REPO="$HOME/multigpu_workspace/external_dependencies/dexjoco"
CKPT="$BASE_DIR/ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_baseline/checkpoint-60000"
OUTPUT_BASE="$BASE_DIR/output/dexjoco/_router_consistency/phase_b"
PRIOR="$BASE_DIR/output/dexjoco/_router_consistency/phase_a/rollouts_A.jsonl"
mkdir -p "$OUTPUT_BASE"
PORT=11020
EPISODES=${EPISODES:-50}
TASK="hammer_nail"
COMPRESS_K=${COMPRESS_K:-2}

NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

SERVER_LOG="$OUTPUT_BASE/server.log"
LD_LIBRARY_PATH="$SERVER_LD" NO_ALBUMENTATIONS_UPDATE=1 \
"$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/serve_policy_dexjoco.py" \
    --port "$PORT" --model-path "$CKPT" \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --head main --denoising-steps 4 \
    > "$SERVER_LOG" 2>&1 &
SPID=$!
for i in $(seq 1 60); do
    if grep -qE "server listening|Listening on|serve_forever|local_ip" "$SERVER_LOG" 2>/dev/null; then break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died"; tail -30 "$SERVER_LOG"; exit 1; fi
    sleep 5
done
echo "[i] Server ready (phase_b)"

MUJOCO_GL=egl \
"$CONDA_PATH/envs/dexjoco/bin/python" -u "$BASE_DIR/scripts/dexjoco_router_consistency.py" \
    --config "$DEXJOCO_REPO/configs/rand_obj/$TASK.yaml" \
    --output-jsonl "$OUTPUT_BASE/rollouts_B.jsonl" \
    --mode phase_b \
    --prior-jsonl "$PRIOR" \
    --episodes "$EPISODES" --seed 0 \
    --port "$PORT" --host 127.0.0.1 \
    --max-episode-steps 1500 \
    --compress-k "$COMPRESS_K"
RC=$?
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
echo "PHASE_B_DONE rc=$RC"
exit $RC
