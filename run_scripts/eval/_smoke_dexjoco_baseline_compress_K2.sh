#!/bin/bash
# Smoke: GR00T-N1.5 baseline + fixed-K=2 compression on dexjoco hammer_nail.
# 1 task x 2 episodes. Verifies dexjoco_eval_gr00t_sync.py --compress-k wiring.
set -u
CKPT="$HOME/multigpu_workspace/Isaac-GR00T/ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_baseline/checkpoint-60000"
PORT=10110
N_EPISODES=2
K=2
TASK="hammer_nail"

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
DEXJOCO_REPO="$HOME/multigpu_workspace/external_dependencies/dexjoco"
OUTPUT_BASE="$BASE_DIR/output/dexjoco/_smoke_baseline_compress_K${K}"
mkdir -p "$OUTPUT_BASE/$TASK"

NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

SERVER_LOG="$OUTPUT_BASE/$TASK/server.log"
LD_LIBRARY_PATH="$SERVER_LD" NO_ALBUMENTATIONS_UPDATE=1 \
"$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/serve_policy_dexjoco.py" \
    --port "$PORT" --model-path "$CKPT" \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --head main --denoising-steps 4 \
    > "$SERVER_LOG" 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 60); do
    if grep -qE "server listening|Listening on|serve_forever|local_ip" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died"; tail -30 "$SERVER_LOG"; exit 1; fi
    sleep 5
done
[ "$READY" -ne 1 ] && { echo "[ERR] server not ready"; tail -30 "$SERVER_LOG"; kill "$SPID" 2>/dev/null; exit 1; }
echo "[i] Server ready"

MUJOCO_GL=egl \
"$CONDA_PATH/envs/dexjoco/bin/python" -u "$BASE_DIR/scripts/dexjoco_eval_gr00t_sync.py" \
    --config "$DEXJOCO_REPO/configs/rand_obj/$TASK.yaml" \
    --port "$PORT" --host 127.0.0.1 \
    --episodes $N_EPISODES --max-episode-steps 1500 \
    --compress-k $K \
    --output "$OUTPUT_BASE/$TASK" \
    > "$OUTPUT_BASE/$TASK/eval.log" 2>&1
RC=$?
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
echo "[i] Eval rc=$RC"
grep -E "Success|Failed|Success rate" "$OUTPUT_BASE/$TASK/eval.log" | tail -8
echo "SMOKE_DEXJOCO_COMPRESS_K${K}_DONE rc=$RC"
exit $RC
