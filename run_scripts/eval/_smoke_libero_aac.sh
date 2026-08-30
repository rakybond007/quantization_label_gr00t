#!/bin/bash
# Smoke libero AAC cliff on baseline (head=main_n, main_sumpair) OR MoE
# (head=moe_selective, m8). 1 suite (libero_10) x task0 x 3 ep.
# Usage: bash _smoke_libero_aac.sh <HF_REPO> <TAG> <SERVE> <HEAD> <COMPRESS_SRC> [PORT]
set -u
HF_REPO=${1:?repo}; TAG=${2:?tag}; SERVE=${3:?serve}; HEAD=${4:?head}; CSRC=${5:?compress_src}; PORT=${6:-8841}
N_SAMPLES=10

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
OUT="$BASE_DIR/output/libero/_smoke_aac_${TAG}"
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$BASE_DIR"; unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

echo "[$(date +%T)] [$TAG] downloading $HF_REPO ..."
CKPT=$("$CONDA_PATH/envs/gr00t/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_REPO', repo_type='model'))")
echo "[$(date +%T)] [$TAG] serve=$SERVE head=$HEAD csrc=$CSRC CKPT=$CKPT"

"$CONDA_PATH/envs/gr00t/bin/python" "$BASE_DIR/scripts/$SERVE" \
    --port=$PORT --model-path=$CKPT --head=$HEAD --n-samples=$N_SAMPLES \
    --discrete-action-dims 6 \
    > "$OUT/server.log" 2>&1 &
SPID=$!
trap "kill $SPID 2>/dev/null; pkill -P $SPID 2>/dev/null" EXIT INT TERM
READY=0
for i in $(seq 1 90); do
    grep -q "Creating server" "$OUT/server.log" 2>/dev/null && READY=1 && break
    kill -0 "$SPID" 2>/dev/null || { echo "[ERR] server died"; tail -25 "$OUT/server.log"; exit 1; }
    sleep 5
done
[ "$READY" -ne 1 ] && { echo "[ERR] server not ready"; tail -25 "$OUT/server.log"; exit 1; }
sleep 20
echo "[$(date +%T)] [$TAG] server up; running aac_cliff 3-ep libero_10 task0..."

OPENPI_SRC="$HOME/multigpu_workspace/openpi/packages/openpi-client/src"
PYTHONPATH="$OPENPI_SRC" "$CONDA_PATH/envs/libero/bin/python" "$BASE_DIR/gr00t/eval/libero/eval_taskwise_gr00t_moe.py" \
    --args.task-suite-name libero_10 --args.video-out-path "$OUT" \
    --args.task_idx=0 --args.port=$PORT --args.num_trials_per_task=3 \
    --args.decision_mode=aac_cliff --args.aac_compress_src=$CSRC --args.aac_xi=1 \
    >& "$OUT/eval.log"
RC=$?
kill $SPID 2>/dev/null
echo "[$(date +%T)] [$TAG] client exit=$RC"
if [ $RC -eq 0 ]; then echo "[$TAG PASS]"; else echo "[$TAG FAIL]"; tail -25 "$OUT/eval.log"; fi
echo "--- results ---"; head -4 "$OUT/0_results.txt" 2>/dev/null
echo "--- chunk kinds (compression evidence) ---"; grep -iE "Chunk kind|Pick counts" "$OUT/eval.log" 2>/dev/null | tail -3
