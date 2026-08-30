#!/bin/bash
# RoboTwin 2.0 minimal smoke: boot our zmq server with baseline ckpt-50000
# and run 1 task (adjust_bottle, demo_clean, seed 0) through RoboTwin sim.
# Intended for tmux 0:0 (interactive worker node with GPUs).
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA="$HOME/miniconda3"
RT="$HOME/multigpu_workspace/external_dependencies/RoboTwin"
OUT="$BASE_DIR/output/robotwin/_smoke_baseline_50k"
mkdir -p "$OUT"

CKPT="$BASE_DIR/ckpt/robotwin/groot_n1_5_bs64_baseline_clean50/checkpoint-50000"
PORT=5755
TASK=adjust_bottle

# ---- Server (gr00t env, our zmq inference_service) ----
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA/envs/gr00t/bin:$PATH"
cd "$BASE_DIR"
echo "[$(date +%T)] booting server: $CKPT (port $PORT)..."
PYTHONUNBUFFERED=1 "$CONDA/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config robotwin_agilex --embodiment_tag new_embodiment \
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
    echo "[ERR] server not ready in 7.5 min"; tail -30 "$OUT/server.log"
    kill $SPID 2>/dev/null; exit 1
fi
echo "[$(date +%T)] server up. running eval on $TASK..."

# ---- Eval client (robotwin env, RoboTwin sim) ----
cd "$RT"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
PYTHONUNBUFFERED=1 "$CONDA/envs/robotwin/bin/python" -u script/eval_policy.py \
    --config policy/gr00t_zmq/deploy_policy.yml \
    --overrides \
        --task_name $TASK \
        --task_config demo_clean \
        --ckpt_setting smoke \
        --seed 0 \
        --policy_name gr00t_zmq.deploy_policy \
    > "$OUT/eval.log" 2>&1
RC=$?
kill $SPID 2>/dev/null
echo "[$(date +%T)] eval exit=$RC"
if [ $RC -eq 0 ]; then
    echo "=== SMOKE PASS ==="
else
    echo "=== SMOKE FAIL (rc=$RC) ==="
    tail -40 "$OUT/eval.log"
fi
