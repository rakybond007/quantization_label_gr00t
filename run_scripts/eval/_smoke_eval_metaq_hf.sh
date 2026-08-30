#!/bin/bash
# Smoke-eval a metaq HF checkpoint: snapshot_download → inference_service_metaq
# server → 1-task × 2-ep robocasa client.  Verifies the checkpoint structure
# (state_dict matches metaq backbone+head, meta_q_emb included).
# Usage: bash _smoke_eval_metaq_hf.sh <hf_repo_id> <tag>
set -u
HF_REPO=${1:?"need hf repo id e.g. prehj/gr00t-n1.5-robocasa-metaq-no_shared_t"}
TAG=${2:?"need a short tag for output dir"}
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
OUT="$BASE_DIR/output/robocasa/_smoke_eval_hf_${TAG}"
PORT=$((9700 + RANDOM % 100))
mkdir -p "$OUT"
cd "$BASE_DIR"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "[$(date '+%T')] === SMOKE-EVAL HF ($TAG): $HF_REPO ==="
echo "[$(date '+%T')] downloading ckpt..."
"$CONDA_PATH/envs/gr00t/bin/python" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$HF_REPO', repo_type='model')
print('downloaded:', p)
" > "$OUT/download.log" 2>&1 || { echo "[ERR] download failed"; cat "$OUT/download.log"; exit 1; }
LOCAL=$(grep -oE '/[^ ]+models--[^ ]+/snapshots/[^ ]+' "$OUT/download.log" | head -1)
[ -z "$LOCAL" ] && LOCAL=$(awk '/^downloaded:/ {print $2}' "$OUT/download.log")
echo "[$(date '+%T')] local path: $LOCAL"

echo "[$(date '+%T')] starting server (port=$PORT)..."
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_metaq.py" --server \
    --port $PORT --model_path "$LOCAL" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe --discrete-action-dims 6 11 \
    > "$OUT/server.log" 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 60); do
    grep -q "Server is ready" "$OUT/server.log" 2>/dev/null && READY=1 && break
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died early"; break; fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then
    echo "[$(date '+%T')] === SERVER FAILED — log tail ==="
    tail -25 "$OUT/server.log"
    kill "$SPID" 2>/dev/null
    exit 1
fi
echo "[$(date '+%T')] server ready. running client..."
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
    "$BASE_DIR/scripts/robocasa_service_moe.py" \
    --port $PORT --host localhost --env_name CloseDrawer \
    --video_dir "$OUT/CloseDrawer" --seed 42 --n_episodes 2 \
    --max_episode_steps 300 --generative_textures \
    > "$OUT/client.log" 2>&1
RC=$?
kill "$SPID" 2>/dev/null
echo "[$(date '+%T')] === client exit: $RC ==="
tail -10 "$OUT/client.log"
[ $RC -eq 0 ] && echo "=== $TAG: PASS ===" || echo "=== $TAG: FAIL ==="
echo "$LOCAL" > "$OUT/local_path.txt"
