#!/bin/bash
# Diagnostic: isolate whether per-dim magnitude cap (not direction gate) blocks
# SAIL aggregation. Runs 1 ep with direction gate disabled (dot=-1.1) and
# gripper gate off, so ONLY the per-dim 0.05 cap can stop a merge.
set -u
TASK=${1:-PnPCounterToSink}; PORT=${2:-8564}
BASE=$HOME/multigpu_workspace/Isaac-GR00T; CONDA=$HOME/miniconda3
CKPT=$BASE/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000
OUT=$BASE/output/robocasa/_smoke_sail/_diag_$TASK; rm -rf "$OUT"; mkdir -p "$OUT"
NVIDIA_PKG_DIR="$CONDA/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1
PYTHONUNBUFFERED=1 $CONDA/envs/gr00t/bin/python -u $BASE/scripts/inference_service.py --server \
    --port $PORT --model_path "$CKPT" --data_config single_panda_gripper \
    --embodiment_tag new_embodiment --denoising_steps 4 --head main > "$OUT/server.log" 2>&1 &
SPID=$!; trap "kill $SPID 2>/dev/null" EXIT INT TERM
for i in $(seq 1 90); do grep -q "Server is ready" "$OUT/server.log" 2>/dev/null && break; sleep 10; done
echo "[diag] server up; running 1 ep, dot gate OFF, gripper gate OFF"
PYTHONUNBUFFERED=1 $CONDA/envs/robocasa_gr00t/bin/python -u $BASE/scripts/robocasa_service_sail.py \
    --port $PORT --host localhost --env_name "$TASK" --video_dir "$OUT" --seed 42 \
    --n_episodes 1 --max_episode_steps 1500 --generative_textures \
    --sail --no_gripper_gate --agg_dot_thresh -1.1 2>&1 | tee "$OUT/client.log" | grep -E "group_sizes|success"
kill $SPID 2>/dev/null
