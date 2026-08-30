#!/bin/bash
# Smoke 1 RoboTwin 2.0 task end-to-end:
#   - boot our GR00T zmq inference server (gr00t env, GPU)
#   - run RoboTwin's script/eval_policy.py (robotwin env, sim)
#     via policy_name=gr00t_zmq.deploy_policy (this repo's adapter)
#
# Usage:
#   bash _smoke_eval_robotwin.sh <ckpt_path> <variant> <task_name> [task_config] [port]
#
#   variant:     "baseline" (inference_service.py)  OR  "fair_moe" (inference_service_fair_moe.py)
#   task_config: demo_clean | demo_randomized (default demo_clean)
#   port:        zmq port (default 5555)
#
# Example:
#   bash _smoke_eval_robotwin.sh \
#     $HOME/multigpu_workspace/Isaac-GR00T/ckpt/robotwin/groot_n1_5_bs64_baseline_clean50/checkpoint-50000 \
#     baseline beat_block_hammer

set -u
CKPT=${1:?need ckpt path}
VARIANT=${2:?need variant (baseline|fair_moe)}
TASK=${3:?need task name}
TASK_CONFIG=${4:-demo_clean}
PORT=${5:-5555}

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
RT_DIR="$HOME/multigpu_workspace/external_dependencies/RoboTwin"
OUT="$BASE_DIR/output/robotwin/_smoke_${VARIANT}_${TASK}_${TASK_CONFIG}"
mkdir -p "$OUT"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

case "$VARIANT" in
    baseline) SVC="scripts/inference_service.py";     SVC_ARGS="" ;;
    fair_moe) SVC="scripts/inference_service_fair_moe.py"; SVC_ARGS="--head moe --discrete-action-dims 6 13" ;;
    *) echo "unknown variant: $VARIANT"; exit 1 ;;
esac

# --- 1) boot server (background, gr00t env, GPU0) ---
echo "[$(date '+%T')] [smoke] booting GR00T server: $SVC port=$PORT ckpt=$CKPT"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/$SVC" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config robotwin_agilex --embodiment_tag new_embodiment \
    $SVC_ARGS \
    > "$OUT/server.log" 2>&1 &
SPID=$!
trap "kill $SPID 2>/dev/null; pkill -P $SPID 2>/dev/null" EXIT INT TERM

READY=0
for i in $(seq 1 90); do
    grep -q "Server is ready" "$OUT/server.log" 2>/dev/null && READY=1 && break
    if ! kill -0 "$SPID" 2>/dev/null; then
        echo "[ERR] server died early"; tail -30 "$OUT/server.log"; exit 1
    fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then
    echo "[ERR] server not ready in 7.5 min"; tail -30 "$OUT/server.log"; exit 1
fi
sleep 5

# --- 2) run eval_policy.py (robotwin env, sim, GPU1) ---
echo "[$(date '+%T')] [smoke] running eval_policy.py task=$TASK config=$TASK_CONFIG"
cd "$RT_DIR"
CUDA_VISIBLE_DEVICES=1 \
"$CONDA_PATH/envs/robotwin/bin/python" "$RT_DIR/script/eval_policy.py" \
    --config "$RT_DIR/policy/gr00t_zmq/deploy_policy.yml" \
    --overrides \
    --task_name "$TASK" \
    --task_config "$TASK_CONFIG" \
    --ckpt_setting smoke \
    --seed 0 \
    --policy_name "gr00t_zmq.deploy_policy" \
    --instruction_type unseen \
    --port "$PORT" \
    >& "$OUT/eval.log"
RC=$?

kill $SPID 2>/dev/null
echo "[$(date '+%T')] [smoke] client exit=$RC"
if [ $RC -eq 0 ]; then
    echo "[$(date '+%T')] === smoke PASS ==="
else
    echo "[$(date '+%T')] === smoke FAIL (exit=$RC) ==="
    tail -30 "$OUT/eval.log"
fi
