#!/bin/bash
#SBATCH --job-name=eval_dexjoco_single_arm_multitask_6tasks_pi0_baseline_50k_50ep_arr6
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --exclude=worker-node100
#SBATCH --array=0-5
#SBATCH --output=out/%A_%a-eval_dexjoco_pi0_baseline_50k.out
#SBATCH --error=out/%A_%a-eval_dexjoco_pi0_baseline_50k.err
#SBATCH --time=4:00:00
#SBATCH --comment="DexJoCo single-arm 6-task pi0 baseline @50k, 50 ep/task."

set -u
TASKS=(hammer_nail click_mouse pick_bucket pinch_tongs fold_glasses water_plant)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
PORT=$((9820 + SLURM_ARRAY_TASK_ID))
N_EPISODES=50

OPENPI_DIR="$HOME/multigpu_workspace/openpi"
OPENPI_VENV="$OPENPI_DIR/.venv"
DEXJOCO_REPO="$HOME/multigpu_workspace/external_dependencies/dexjoco"
GR00T_BASE="$HOME/multigpu_workspace/Isaac-GR00T"

CKPT="$OPENPI_DIR/checkpoints/pi0_dexjoco_single_arm_baseline/pi0_dexjoco_single_arm_baseline_h100_2gpu_perGPU32_bs64_60k/50000"
CONFIG="$DEXJOCO_REPO/configs/rand_obj/$TASK.yaml"
OUTPUT_BASE="$GR00T_BASE/output/dexjoco/single_arm_multitask_pi0_baseline_50k"
mkdir -p out "$OUTPUT_BASE/$TASK"

SERVER_LOG="$OUTPUT_BASE/$TASK/server-$SLURM_ARRAY_TASK_ID.log"
EVAL_LOG="$OUTPUT_BASE/$TASK/eval-$SLURM_ARRAY_TASK_ID.log"

unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "[i] $TASK arr=$SLURM_ARRAY_TASK_ID port=$PORT ckpt=$CKPT"
cd "$OPENPI_DIR"
"$OPENPI_VENV/bin/python" -u scripts/serve_policy.py \
    --port $PORT \
    policy:checkpoint \
    --policy.config=pi0_dexjoco_single_arm_baseline \
    --policy.dir="$CKPT" > "$SERVER_LOG" 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 120); do
    if grep -qE "server listening|Listening on|serve_forever|local_ip" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died early"; tail -40 "$SERVER_LOG"; exit 1; fi
    sleep 5
done
[ "$READY" -ne 1 ] && { echo "[ERR] server timed out"; tail -40 "$SERVER_LOG"; kill "$SPID" 2>/dev/null; exit 1; }
echo "[i] server ready"

MUJOCO_GL=egl /sjw_alinlab2/home/hojin2/miniconda3/envs/dexjoco/bin/python -u \
    "$GR00T_BASE/scripts/dexjoco_eval_gr00t_sync.py" \
    --config "$CONFIG" \
    --port $PORT --host 127.0.0.1 \
    --episodes $N_EPISODES --seed 0 \
    --output "$OUTPUT_BASE/$TASK" > "$EVAL_LOG" 2>&1
RC=$?
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
echo "[i] $TASK done rc=$RC"
grep -E "Success rate" "$EVAL_LOG" | tail -3
exit $RC
