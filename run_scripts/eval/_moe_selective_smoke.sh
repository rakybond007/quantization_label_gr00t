#!/bin/bash
# Validate head=moe_selective end-to-end on per_expert_moe ckpt.
# 1 task × 3 ep, both AAC rules.
set -u
BASE=$HOME/multigpu_workspace/Isaac-GR00T
CONDA=$HOME/miniconda3
PORT=8600
NVIDIA_PKG_DIR="$CONDA/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

OUT=$BASE/output/robocasa/_smoke_selective/_moe_sel_val
rm -rf "$OUT"; mkdir -p "$OUT"

echo "[$(date '+%T')] starting per_expert_moe + head=moe_selective + N=10..."
PYTHONUNBUFFERED=1 $CONDA/envs/gr00t/bin/python -u $BASE/scripts/inference_service.py --server \
    --port $PORT \
    --model_path "$BASE/ckpt/robocasa/groot/groot_n1_5_bs64_moe4_per_expert_body/checkpoint-60000" \
    --data_config single_panda_gripper \
    --embodiment_tag new_embodiment \
    --denoising_steps 4 \
    --head moe_selective \
    --n_samples 10 \
    --discrete-action-dims 6 11 \
    > $OUT/server.log 2>&1 &
SPID=$!
trap "kill $SPID 2>/dev/null; pkill -P $SPID 2>/dev/null" EXIT INT TERM
for i in $(seq 1 60); do
    grep -q "Server is ready" $OUT/server.log 2>/dev/null && break
    sleep 5
done
grep -q "Server is ready" $OUT/server.log || { echo "[ERROR] server not ready"; tail -30 $OUT/server.log; exit 1; }
echo "[$(date '+%T')] server ready"

run_one() {
    local LABEL=$1; local ARGS=$2
    local DIR=$OUT/$LABEL
    mkdir -p $DIR
    echo "[$(date '+%T')]   $LABEL: $ARGS"
    PYTHONUNBUFFERED=1 $CONDA/envs/robocasa_gr00t/bin/python -u \
        $BASE/scripts/robocasa_service_selective.py \
        --port $PORT --host localhost \
        --env_name "CoffeeSetupMug" \
        --video_dir "$DIR" \
        --seed 42 \
        --n_episodes 3 \
        --max_episode_steps 1500 \
        --generative_textures \
        --score-mode entropy \
        $ARGS \
        > $DIR/eval.log 2>&1
    succ=$(grep -E "^episode .* True" "$DIR/prediction.txt" 2>/dev/null | wc -l)
    ep=$(grep -c "^episode" "$DIR/prediction.txt" 2>/dev/null)
    meanlen=$(grep "mean_exec_chunk_len" "$DIR/prediction.txt" 2>/dev/null | tail -1 | tr -d '\n')
    picks=$(grep "moe_pick_counts" "$DIR/prediction.txt" 2>/dev/null | tail -1 | tr -d '\n')
    err=$(grep -c "Traceback" "$DIR/eval.log" 2>/dev/null)
    echo "[$(date '+%T')]     -> succ=$succ/$ep  $meanlen  $picks  errors=$err"
}

run_one cliff  "--decision-rule aac_cliff --aac-xi 1"
run_one binary "--decision-rule aac_chunk_binary --aac-xi 4"

echo "[$(date '+%T')] done"
kill $SPID 2>/dev/null
