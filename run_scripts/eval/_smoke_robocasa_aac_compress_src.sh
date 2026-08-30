#!/bin/bash
# Smoke the aac_compress_src option on robocasa selective AAC eval.
# server head=moe_selective (N=10) once; client runs aac_cliff twice with
# --aac-compress-src main_sumpair and m8. 1 task (CloseDrawer) x 2 ep each.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_moe4_per_expert_body/checkpoint-60000"
PORT=9355
OUT="$BASE_DIR/output/robocasa/_smoke_aac_compress_src"
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "[$(date +%T)] booting server (head=moe_selective N=10)..."
PYTHONUNBUFFERED=1 "$CONDA/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe_selective --n_samples 10 \
    --discrete-action-dims 6 11 \
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
echo "[$(date +%T)] server up."

for SRC in main_sumpair m8; do
    ODIR="$OUT/$SRC"; mkdir -p "$ODIR"
    echo "[$(date +%T)] === aac_cliff compress_src=$SRC ==="
    PYTHONUNBUFFERED=1 "$CONDA/envs/robocasa_gr00t/bin/python" -u \
        "$BASE_DIR/scripts/robocasa_service_selective.py" \
        --port $PORT --host localhost --env_name CloseDrawer \
        --video_dir "$ODIR" --seed 42 --n_episodes 2 --max_episode_steps 300 \
        --generative_textures --score-mode entropy --decision-rule aac_cliff \
        --aac-xi 1 --aac-compress-src "$SRC" \
        > "$ODIR/client.log" 2>&1
    RC=$?
    echo "[$(date +%T)] [$SRC] client exit=$RC"
    if [ $RC -eq 0 ]; then echo "=== $SRC PASS ==="; else echo "=== $SRC FAIL ==="; tail -25 "$ODIR/client.log"; fi
done
kill $SPID 2>/dev/null
echo "SMOKE_AAC_DONE"
