#!/bin/bash
# Smoke baseline (non-MoE) robocasa + AAC cliff via main_sumpair (no m8 decoder
# needed). server head=selective (N main samples for entropy) on the plain
# baseline ckpt; client runs aac_cliff with --aac-compress-src main_sumpair.
# Validates that inference-time sum-pair compression + entropy cliff runs
# end-to-end on the base model and actually compresses (check h_star / chunk log).
# Usage: bash _smoke_robocasa_baseline_aac_cliff.sh [TASK] [N_EP] [PORT]
set -u
TASK=${1:-CloseDrawer}
N_EP=${2:-3}
PORT=${3:-9360}
AAC_XI=${4:-1}
CURV=${5:-0.0}
N_SAMPLES=10

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
OUT="$BASE_DIR/output/robocasa/_smoke_baseline_aac_cliff/$TASK"
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "[$(date +%T)] booting BASELINE server (head=selective N=$N_SAMPLES)..."
PYTHONUNBUFFERED=1 "$CONDA/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head main_n --n_samples $N_SAMPLES \
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
echo "[$(date +%T)] server up. running aac_cliff (main_sumpair) on $TASK x $N_EP ep..."

PYTHONUNBUFFERED=1 "$CONDA/envs/robocasa_gr00t/bin/python" -u \
    "$BASE_DIR/scripts/robocasa_service_selective.py" \
    --port $PORT --host localhost --env_name "$TASK" \
    --video_dir "$OUT" --seed 42 --n_episodes $N_EP --max_episode_steps 1500 \
    --generative_textures --score-mode entropy --decision-rule aac_cliff \
    --aac-xi $AAC_XI --aac-compress-src main_sumpair --curvature-weight $CURV \
    > "$OUT/client.log" 2>&1
RC=$?
kill $SPID 2>/dev/null
echo "[$(date +%T)] client exit=$RC"
if [ $RC -eq 0 ]; then echo "=== BASELINE_AAC_CLIFF PASS ==="; else echo "=== FAIL ==="; tail -30 "$OUT/client.log"; fi
echo "--- prediction.txt ---"; grep -hE "^episode|^is_success:" "$OUT/prediction.txt" 2>/dev/null
echo "--- compression evidence (h_star / compressed) ---"; grep -hiE "h_star|compress|cliff|chunk" "$OUT/client.log" 2>/dev/null | tail -8
