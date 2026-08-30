#!/bin/bash
# Smoke: dexjoco MoE no_balance with --moe-stochastic (sample by router probs).
set -u
TASK=hammer_nail
N_EPISODES=5
PORT=9502

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
DEXJOCO_REPO="$HOME/multigpu_workspace/external_dependencies/dexjoco"
CKPT="$BASE_DIR/ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_moe4_v1_no_balance/checkpoint-60000"
CONFIG="$DEXJOCO_REPO/configs/rand_obj/$TASK.yaml"
OUT="$BASE_DIR/analysis/_smoke/_stochastic_smoke/$TASK"
mkdir -p "$OUT"

NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

SERVER_LOG="$OUT/server.log"
echo "[i] Starting server (stochastic=True) on port $PORT"
LD_LIBRARY_PATH="$SERVER_LD" NO_ALBUMENTATIONS_UPDATE=1 \
"$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/serve_policy_dexjoco.py" \
    --port "$PORT" --model-path "$CKPT" \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --head moe --denoising-steps 4 --moe-stochastic \
    > "$SERVER_LOG" 2>&1 &
SPID=$!

READY=0
for i in $(seq 1 60); do
    if grep -qE "server listening|Listening on|serve_forever|local_ip" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died"; tail -30 "$SERVER_LOG"; exit 1; fi
    sleep 5
done
[ "$READY" -ne 1 ] && { echo "[ERR] server not ready"; tail -40 "$SERVER_LOG"; kill "$SPID" 2>/dev/null; exit 1; }
grep -E "stochastic|head=" "$SERVER_LOG"
echo "[i] Server ready, running 5 ep on $TASK"

MUJOCO_GL=egl \
"$CONDA_PATH/envs/dexjoco/bin/python" -u "$BASE_DIR/scripts/dexjoco_eval_gr00t_sync.py" \
    --config "$CONFIG" --port "$PORT" --host 127.0.0.1 \
    --episodes $N_EPISODES --max-episode-steps 1500 \
    --output "$OUT" > "$OUT/eval.log" 2>&1
RC=$?

kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
echo
echo "=== eval log (last 20) ==="
tail -20 "$OUT/eval.log"
echo
echo "=== success summary ==="
ls "$OUT" | grep -E "^episode_" | sed 's/^episode_[0-9]*_//' | sort | uniq -c
echo
echo "SMOKE_STOCHASTIC_DONE rc=$RC"
exit $RC
