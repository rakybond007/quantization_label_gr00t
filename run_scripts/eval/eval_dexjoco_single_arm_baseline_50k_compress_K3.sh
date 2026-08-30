#!/bin/bash
#SBATCH --job-name=eval_dexjoco_single_arm_baseline_50k_compress_K3_6t_50ep_arr6
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --exclude=worker-node100
#SBATCH --array=0-5
#SBATCH --output=out/%A_%a-eval_dexjoco_single_arm_baseline.out
#SBATCH --error=out/%A_%a-eval_dexjoco_single_arm_baseline.err
#SBATCH --time=4:00:00
#SBATCH --comment="DexJoCo single-arm 6-task baseline GR00T-N1.5 @60k, 50 ep/task."

set -u
TASKS=(hammer_nail click_mouse pick_bucket pinch_tongs fold_glasses water_plant)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
PORT=$((10340 + SLURM_ARRAY_TASK_ID))
N_EPISODES=50

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
DEXJOCO_REPO="$HOME/multigpu_workspace/external_dependencies/dexjoco"
CKPT="$BASE_DIR/ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_baseline/checkpoint-50000"
CONFIG="$DEXJOCO_REPO/configs/rand_obj/$TASK.yaml"
OUTPUT_BASE="$BASE_DIR/output/dexjoco/single_arm_multitask_baseline_50k_compress_K3"
mkdir -p out "$OUTPUT_BASE/$TASK"

NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

SERVER_LOG="$OUTPUT_BASE/$TASK/server-$SLURM_ARRAY_TASK_ID.log"
LD_LIBRARY_PATH="$SERVER_LD" NO_ALBUMENTATIONS_UPDATE=1 \
"$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/serve_policy_dexjoco.py" \
    --port "$PORT" --model-path "$CKPT" \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --head main --denoising-steps 4 \
    > "$SERVER_LOG" 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 60); do  # up to 5 min
    if grep -qE "server listening|Listening on|serve_forever|local_ip" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died"; tail -30 "$SERVER_LOG"; exit 1; fi
    sleep 5
done
[ "$READY" -ne 1 ] && { echo "[ERR] server not ready"; tail -40 "$SERVER_LOG"; kill "$SPID" 2>/dev/null; exit 1; }

MUJOCO_GL=egl \
"$CONDA_PATH/envs/dexjoco/bin/python" -u "$BASE_DIR/scripts/dexjoco_eval_gr00t_sync.py" \
    --config "$CONFIG" --port "$PORT" --host 127.0.0.1 \
    --episodes $N_EPISODES --compress-k 3 --max-episode-steps 1500 \
    --output "$OUTPUT_BASE/$TASK" \
    > "$OUTPUT_BASE/$TASK/eval-$SLURM_ARRAY_TASK_ID.log" 2>&1
RC=$?

kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
echo "[i] Array $SLURM_ARRAY_TASK_ID ($TASK) done rc=$RC."
exit $RC
