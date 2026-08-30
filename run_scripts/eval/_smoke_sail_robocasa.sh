#!/bin/bash
# Smoke SAIL robocasa eval on the BASE GR00T-N1.5 model (no MoE).
# server (gr00t env, inference_service --head main) + client robocasa_service_sail.
# Runs the SAME task twice on one server: (1) baseline 1-by-1 (--sail off),
# (2) SAIL aggregation (--sail). Compares action_steps (speedup) + success.
# Usage: bash _smoke_sail_robocasa.sh [TASK] [N_EP] [PORT]
set -u
TASK=${1:-PnPCounterToSink}
N_EP=${2:-2}
PORT=${3:-8560}

BASE=$HOME/multigpu_workspace/Isaac-GR00T
CONDA=$HOME/miniconda3
CKPT=$BASE/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000
OUT=$BASE/output/robocasa/_smoke_sail/$TASK
rm -rf "$OUT"; mkdir -p "$OUT/baseline" "$OUT/sail"

NVIDIA_PKG_DIR="$CONDA/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

echo "[$(date '+%T')] starting base-model server (head=main) on port $PORT"
PYTHONUNBUFFERED=1 $CONDA/envs/gr00t/bin/python -u $BASE/scripts/inference_service.py --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head main \
    > "$OUT/server.log" 2>&1 &
SPID=$!
trap "kill $SPID 2>/dev/null; pkill -P $SPID 2>/dev/null" EXIT INT TERM

for i in $(seq 1 120); do
    grep -q "Server is ready" "$OUT/server.log" 2>/dev/null && break
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died"; tail -30 "$OUT/server.log"; exit 1; fi
    sleep 5
done
grep -q "Server is ready" "$OUT/server.log" 2>/dev/null || { echo "[ERR] server not ready"; tail -30 "$OUT/server.log"; exit 1; }
echo "[$(date '+%T')] server ready"

run_client () {  # $1=outdir  $2.. = extra flags
    local odir=$1; shift
    PYTHONUNBUFFERED=1 $CONDA/envs/robocasa_gr00t/bin/python -u $BASE/scripts/robocasa_service_sail.py \
        --port $PORT --host localhost --env_name "$TASK" \
        --video_dir "$odir" --seed 42 --n_episodes $N_EP \
        --max_episode_steps 1500 --generative_textures "$@" \
        2>&1 | tee "$odir/client.log"
}

echo "[$(date '+%T')] === (1) baseline 1-by-1 (sail off) ==="
run_client "$OUT/baseline"
echo "[$(date '+%T')] === (2) SAIL aggregation (sail on) ==="
run_client "$OUT/sail" --sail

kill $SPID 2>/dev/null
echo "[$(date '+%T')] === SMOKE done ==="
echo "--- baseline action_steps ---"; grep -hE "^episode" "$OUT/baseline/prediction.txt" 2>/dev/null
echo "--- sail action_steps ---";     grep -hE "^episode" "$OUT/sail/prediction.txt" 2>/dev/null
