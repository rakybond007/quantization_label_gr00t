#!/bin/bash
set -u
BASE=$HOME/multigpu_workspace/Isaac-GR00T
CONDA=$HOME/miniconda3
PORT=8500
NVIDIA_PKG_DIR="$CONDA/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

OUT=$BASE/output/robocasa/_smoke_selective/_perstep_check
rm -rf "$OUT"; mkdir -p "$OUT"

echo "[$(date '+%T')] starting server N=10..."
PYTHONUNBUFFERED=1 $CONDA/envs/gr00t/bin/python -u $BASE/scripts/inference_service.py --server \
    --port $PORT \
    --model_path "$BASE/ckpt/robocasa/groot/groot_n1_5_bs64_mh_m8_discfix/checkpoint-60000" \
    --data_config single_panda_gripper \
    --embodiment_tag new_embodiment \
    --denoising_steps 4 \
    --head selective \
    --n_samples 10 \
    --discrete-action-dims 6 11 \
    > $OUT/server.log 2>&1 &
SPID=$!
trap "kill $SPID 2>/dev/null; pkill -P $SPID 2>/dev/null" EXIT INT TERM
for i in $(seq 1 60); do
    grep -q "Server is ready" $OUT/server.log 2>/dev/null && break
    sleep 5
done
grep -q "Server is ready" $OUT/server.log || { echo "[ERROR] server not ready"; tail -20 $OUT/server.log; exit 1; }
echo "[$(date '+%T')] server ready, running 1 task × 5 ep..."

mkdir -p $OUT/CoffeeSetupMug
PYTHONUNBUFFERED=1 $CONDA/envs/robocasa_gr00t/bin/python -u \
    $BASE/scripts/robocasa_service_selective.py \
    --port $PORT --host localhost \
    --env_name "CoffeeSetupMug" \
    --video_dir "$OUT/CoffeeSetupMug" \
    --seed 42 \
    --n_episodes 5 \
    --max_episode_steps 1500 \
    --generative_textures \
    --score-mode entropy \
    --dump-scores-only \
    > $OUT/CoffeeSetupMug/eval.log 2>&1
echo "[$(date '+%T')] done"
kill $SPID 2>/dev/null
