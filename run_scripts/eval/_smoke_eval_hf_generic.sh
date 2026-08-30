#!/bin/bash
# Generic HF smoke: snapshot_download → 1 task × 2 ep × CloseDrawer.
# Usage: bash _smoke_eval_hf_generic.sh <HF_REPO> <TAG> <SERVICE_PY> <PORT> <GPU>
set -u
HF_REPO=${1:?need hf repo}
TAG=${2:?need tag}
SERVICE=${3:?need service py path relative to BASE_DIR}
PORT=${4:?need port}
GPU=${5:-0}

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
OUT="$BASE_DIR/output/robocasa/_smoke_eval_hf_${TAG}"
mkdir -p "$OUT"
cd "$BASE_DIR"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES=$GPU
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "[$(date '+%T')] [$TAG/gpu$GPU] downloading $HF_REPO ..."
"$CONDA_PATH/envs/gr00t/bin/python" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$HF_REPO', repo_type='model')
print('downloaded:', p)
" > "$OUT/download.log" 2>&1 || { echo "[$TAG ERR] download failed"; tail -15 "$OUT/download.log"; exit 1; }
LOCAL=$(awk '/^downloaded:/ {print $2}' "$OUT/download.log")
echo "[$(date '+%T')] [$TAG] LOCAL=$LOCAL"

PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/$SERVICE" --server \
    --port $PORT --model_path "$LOCAL" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe --discrete-action-dims 6 11 \
    > "$OUT/server.log" 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 60); do
    grep -q "Server is ready" "$OUT/server.log" 2>/dev/null && READY=1 && break
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[$TAG ERR] server died"; break; fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then
    echo "[$TAG ERR] server not ready"; tail -25 "$OUT/server.log"; kill "$SPID" 2>/dev/null; exit 1
fi
echo "[$(date '+%T')] [$TAG] server ready, running 2-ep CloseDrawer..."
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
    "$BASE_DIR/scripts/robocasa_service_moe.py" \
    --port $PORT --host localhost --env_name CloseDrawer \
    --video_dir "$OUT/CloseDrawer" --seed 42 --n_episodes 2 \
    --max_episode_steps 300 --generative_textures \
    > "$OUT/client.log" 2>&1
RC=$?
kill "$SPID" 2>/dev/null
echo "$LOCAL" > "$OUT/local_path.txt"
if [ $RC -eq 0 ]; then
    echo "[$(date '+%T')] === $TAG: PASS (client exit=0) ==="
else
    echo "[$(date '+%T')] === $TAG: FAIL (client exit=$RC) ==="
    tail -15 "$OUT/client.log"
fi
