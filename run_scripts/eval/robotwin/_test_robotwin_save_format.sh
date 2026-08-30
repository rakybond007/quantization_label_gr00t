#!/bin/bash
# Quick test of the new result-saving format (summary.txt + per_ep.csv +
# episode{N}_{success|fail}.mp4 in ROBOTWIN_EVAL_SAVE_DIR). 1 task x 3 ep.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA="$HOME/miniconda3"
RT="$HOME/multigpu_workspace/external_dependencies/RoboTwin"
OUT="$BASE_DIR/output/robotwin/_test_save_format"
TASK_OUT="$OUT/adjust_bottle"
rm -rf "$OUT"; mkdir -p "$TASK_OUT"

CKPT="$BASE_DIR/ckpt/robotwin/groot_n1_5_bs64_baseline_clean50/checkpoint-60000"
PORT=5756
TASK=adjust_bottle

unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA/envs/gr00t/bin:$PATH"
cd "$BASE_DIR"
echo "[$(date +%T)] booting server..."
PYTHONUNBUFFERED=1 "$CONDA/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config robotwin_agilex --embodiment_tag new_embodiment \
    > "$OUT/server.log" 2>&1 &
SPID=$!
trap "kill $SPID 2>/dev/null; pkill -P $SPID 2>/dev/null" EXIT INT TERM
READY=0
for i in $(seq 1 90); do
    grep -q "Server is ready" "$OUT/server.log" 2>/dev/null && READY=1 && break
    kill -0 "$SPID" 2>/dev/null || { echo "[ERR] server died"; tail -20 "$OUT/server.log"; exit 1; }
    sleep 5
done
[ "$READY" -ne 1 ] && { echo "[ERR] server not ready"; tail -20 "$OUT/server.log"; exit 1; }
echo "[$(date +%T)] server up. running 3-ep $TASK with new save format..."

cd "$RT"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
ROBOTWIN_EVAL_SAVE_DIR="$TASK_OUT" PYTHONUNBUFFERED=1 "$CONDA/envs/robotwin/bin/python" -u script/eval_policy.py \
    --config policy/gr00t_zmq/deploy_policy.yml \
    --overrides \
        --task_name $TASK \
        --task_config demo_clean \
        --ckpt_setting test_save \
        --seed 0 \
        --policy_name gr00t_zmq.deploy_policy \
        --port $PORT \
        --eval_test_num 3 \
    > "$TASK_OUT/eval.log" 2>&1
RC=$?
kill $SPID 2>/dev/null
echo "[$(date +%T)] eval exit=$RC"
echo "=== TASK_OUT listing ==="
ls -la "$TASK_OUT"
echo "=== summary.txt ==="
cat "$TASK_OUT/summary.txt" 2>/dev/null
echo "=== per_ep.csv ==="
cat "$TASK_OUT/per_ep.csv" 2>/dev/null
echo "=== TEST_DONE rc=$RC ==="
