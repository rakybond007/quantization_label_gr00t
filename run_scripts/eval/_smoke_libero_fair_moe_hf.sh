#!/bin/bash
# Smoke libero fair_moe HF ckpt: serve_policy_fair_moe.py + libero eval client.
# 1 task suite (libero_10) × task_idx=0 × 3 ep.
# Usage: bash _smoke_libero_fair_moe_hf.sh <HF_REPO> <TAG> [PORT] [SERVE_SCRIPT]
# SERVE_SCRIPT: serve_policy_fair_moe.py (v1/v2, default) | serve_policy_fair_moe_v3.py (v3a)
set -u
HF_REPO=${1:?need HF repo}
TAG=${2:?need tag}
PORT=${3:-8801}
SERVE=${4:-serve_policy_fair_moe.py}

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
OUT="$BASE_DIR/output/libero/_smoke_hf_${TAG}"
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$BASE_DIR"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

echo "[$(date '+%T')] [$TAG] downloading $HF_REPO ..."
CKPT=$("$CONDA_PATH/envs/gr00t/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_REPO', repo_type='model'))")
echo "[$(date '+%T')] [$TAG] CKPT=$CKPT"

"$CONDA_PATH/envs/gr00t/bin/python" "$BASE_DIR/scripts/$SERVE" \
    --port=$PORT --model-path=$CKPT --head=moe --n-samples=1 \
    --discrete-action-dims 6 \
    > "$OUT/server.log" 2>&1 &
SPID=$!
trap "kill $SPID 2>/dev/null; pkill -P $SPID 2>/dev/null" EXIT INT TERM

READY=0
for i in $(seq 1 90); do
    grep -q "Creating server" "$OUT/server.log" 2>/dev/null && READY=1 && break
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died"; tail -25 "$OUT/server.log"; exit 1; fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then echo "[ERR] server not ready"; tail -25 "$OUT/server.log"; exit 1; fi
sleep 20  # extra bind time
echo "[$(date '+%T')] [$TAG] server up, running 3-ep libero_10 task0..."

OPENPI_SRC="$HOME/multigpu_workspace/openpi/packages/openpi-client/src"
PYTHONPATH="$OPENPI_SRC" "$CONDA_PATH/envs/libero/bin/python" "$BASE_DIR/gr00t/eval/libero/eval_taskwise_gr00t_moe.py" \
    --args.task-suite-name libero_10 \
    --args.video-out-path "$OUT" \
    --args.task_idx=0 \
    --args.port=$PORT \
    --args.num_trials_per_task=3 \
    --args.decision_mode=as_is \
    >& "$OUT/eval.log"
RC=$?
kill $SPID 2>/dev/null
echo "[$(date '+%T')] [$TAG] client exit=$RC"
if [ $RC -eq 0 ]; then echo "[$TAG PASS]"; else echo "[$TAG FAIL]"; tail -20 "$OUT/eval.log"; fi
