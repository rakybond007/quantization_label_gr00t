#!/bin/bash
# Sweep curvature-weight on 2 contrasting robocasa tasks to pick the value.
# One baseline server (head=main_n N=10); clients loop task x curvature-weight.
# Reports mean chunk exec length (compression amount; 16=none) + SR per cell.
# Usage: bash _sweep_curvature_robocasa.sh [PORT] [N_EP]
set -u
PORT=${1:-9370}
N_EP=${2:-3}
N_SAMPLES=10
TASKS=(PnPCounterToSink TurnOnStove)
CURVS=(0 0.5 1.0 2.0)

BASE=$HOME/multigpu_workspace/Isaac-GR00T
CONDA=$HOME/miniconda3
CKPT=$BASE/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000
OUT=$BASE/output/robocasa/_sweep_curv
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$BASE"
export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "[$(date +%T)] booting baseline server (head=main_n N=$N_SAMPLES)..."
PYTHONUNBUFFERED=1 "$CONDA/envs/gr00t/bin/python" -u "$BASE/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head main_n --n_samples $N_SAMPLES \
    --discrete-action-dims 6 11 \
    > "$OUT/server.log" 2>&1 &
SPID=$!
trap "kill $SPID 2>/dev/null; pkill -P $SPID 2>/dev/null" EXIT INT TERM
for i in $(seq 1 90); do
    grep -q "Server is ready" "$OUT/server.log" 2>/dev/null && break
    kill -0 "$SPID" 2>/dev/null || { echo "[ERR] server died"; tail -20 "$OUT/server.log"; exit 1; }
    sleep 5
done
grep -q "Server is ready" "$OUT/server.log" 2>/dev/null || { echo "[ERR] not ready"; exit 1; }
echo "[$(date +%T)] server up. sweeping ${#TASKS[@]} tasks x ${#CURVS[@]} weights, $N_EP ep each..."

for TASK in "${TASKS[@]}"; do
    for CURV in "${CURVS[@]}"; do
        ODIR="$OUT/${TASK}_curv${CURV}"; mkdir -p "$ODIR"
        PYTHONUNBUFFERED=1 "$CONDA/envs/robocasa_gr00t/bin/python" -u \
            "$BASE/scripts/robocasa_service_selective.py" \
            --port $PORT --host localhost --env_name "$TASK" \
            --video_dir "$ODIR" --seed 42 --n_episodes $N_EP --max_episode_steps 1200 \
            --generative_textures --score-mode entropy --decision-rule aac_cliff \
            --aac-xi 0 --aac-compress-src main_sumpair --curvature-weight $CURV \
            > "$ODIR/client.log" 2>&1
        EXEC=$(grep -hoE "mean exec len so far=[0-9.]+" "$ODIR/client.log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
        SR=$(grep -hoE "success rate = [0-9.]+" "$ODIR/client.log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
        echo "  [$TASK  curv=$CURV]  exec_len=${EXEC:-?}  SR=${SR:-?}"
    done
done
kill $SPID 2>/dev/null
echo "[$(date +%T)] === SWEEP done ==="
