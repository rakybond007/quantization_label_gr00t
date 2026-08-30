#!/bin/bash
# Smoke: fair_moe_v2_b_only HF ckpt, head=moe_selective, N=10 samples,
# Beta-sampled timestep schedule, AAC cliff (entropy). 1 task × 2 ep.
set -u
BASE=$HOME/multigpu_workspace/Isaac-GR00T
CONDA=$HOME/miniconda3
PORT=9690
N_SAMPLES=10
AAC_XI=1
HF_REPO=prehj/gr00t-n1.5-robocasa-fair-moe-v2-b-only

OUT="$BASE/output/robocasa/_smoke_fair_moe_v2_hf_aac_beta"
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$BASE"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "[$(date '+%T')] === downloading HF ckpt ==="
CKPT=$("$CONDA/envs/gr00t/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_REPO', repo_type='model'))")
echo "[$(date '+%T')] CKPT=$CKPT"

echo "[$(date '+%T')] === SMOKE fair_moe_v2 HF + AAC + Beta: starting server ==="
PYTHONUNBUFFERED=1 "$CONDA/envs/gr00t/bin/python" -u "$BASE/scripts/inference_service_fair_moe_v2.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe_selective --n_samples $N_SAMPLES \
    --beta_sample_t \
    --discrete-action-dims 6 11 \
    > "$OUT/server.log" 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 60); do
    grep -q "Server is ready" "$OUT/server.log" 2>/dev/null && READY=1 && break
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died"; tail -25 "$OUT/server.log"; exit 1; fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then echo "[ERR] server not ready"; tail -25 "$OUT/server.log"; kill "$SPID" 2>/dev/null; exit 1; fi
echo "[$(date '+%T')] server ready. running 2-ep CloseDrawer..."

PYTHONUNBUFFERED=1 "$CONDA/envs/robocasa_gr00t/bin/python" -u \
    "$BASE/scripts/robocasa_service_selective.py" \
    --port $PORT --host localhost --env_name CloseDrawer \
    --video_dir "$OUT/CloseDrawer" --seed 42 --n_episodes 2 --max_episode_steps 300 --generative_textures \
    --score-mode entropy --decision-rule aac_cliff --aac-xi $AAC_XI \
    > "$OUT/client.log" 2>&1
RC=$?
kill "$SPID" 2>/dev/null
echo "[$(date '+%T')] === client exit=$RC ==="
if [ $RC -eq 0 ]; then
    echo "=== SMOKE PASS ==="
    grep -E "Beta-sampled|Server is ready" "$OUT/server.log" | head -3
else
    echo "=== SMOKE FAIL ==="; tail -25 "$OUT/client.log"
fi
