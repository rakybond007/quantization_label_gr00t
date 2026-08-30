#!/bin/bash
# Smoke robocasa MoE v1 + AAC cliff: fair_moe server (head=moe_selective N=10) +
# robocasa_service_selective (aac_cliff, xi=0, curv=2.0, compress_src=m8 trained
# decoder). Validates the new combination (fair_moe v1 + moe_selective + curvature).
# Usage: bash _smoke_robocasa_moe_aac.sh [TASK] [N_EP] [PORT]
set -u
TASK=${1:-CloseDoubleDoor}
N_EP=${2:-3}
PORT=${3:-9366}
CKPT_IN=${4:-prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-60k}
CSRC=${5:-m8}
N_SAMPLES=10

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA="$HOME/miniconda3"
OUT="$BASE_DIR/output/robocasa/_smoke_moe_aac/$TASK"
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$BASE_DIR"
export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

if [ -d "$CKPT_IN" ]; then CKPT="$CKPT_IN"; else CKPT=$("$CONDA/envs/gr00t/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$CKPT_IN', repo_type='model'))"); fi
echo "[$(date +%T)] CKPT=$CKPT; booting fair_moe server (head=moe_selective N=$N_SAMPLES)..."
PYTHONUNBUFFERED=1 "$CONDA/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_fair_moe.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe_selective --n_samples $N_SAMPLES \
    --discrete-action-dims 6 11 \
    > "$OUT/server.log" 2>&1 &
SPID=$!
trap "kill $SPID 2>/dev/null; pkill -P $SPID 2>/dev/null" EXIT INT TERM
READY=0
for i in $(seq 1 90); do
    grep -q "Server is ready" "$OUT/server.log" 2>/dev/null && READY=1 && break
    kill -0 "$SPID" 2>/dev/null || { echo "[ERR] server died"; tail -25 "$OUT/server.log"; exit 1; }
    sleep 5
done
[ "$READY" -ne 1 ] && { echo "[ERR] server not ready"; tail -25 "$OUT/server.log"; exit 1; }
echo "[$(date +%T)] server up; running MoE+aac (xi=0,curv2.0,m8) on $TASK x $N_EP ep..."

PYTHONUNBUFFERED=1 "$CONDA/envs/robocasa_gr00t/bin/python" -u \
    "$BASE_DIR/scripts/robocasa_service_selective.py" \
    --port $PORT --host localhost --env_name "$TASK" \
    --video_dir "$OUT" --seed 42 --n_episodes $N_EP --max_episode_steps 1500 \
    --generative_textures --score-mode entropy --decision-rule aac_cliff \
    --aac-xi 0 --aac-compress-src $CSRC --curvature-weight 2.0 \
    > "$OUT/client.log" 2>&1
RC=$?
kill $SPID 2>/dev/null
echo "[$(date +%T)] client exit=$RC"
if [ $RC -eq 0 ]; then echo "=== MoE_AAC PASS ==="; else echo "=== FAIL ==="; tail -30 "$OUT/client.log"; fi
echo "--- prediction ---"; grep -hE "^episode|^is_success:" "$OUT/prediction.txt" 2>/dev/null
echo "--- exec/picks ---"; grep -hiE "mean exec len|router pick|picks" "$OUT/client.log" 2>/dev/null | tail -3
