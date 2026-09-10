#!/bin/bash
#SBATCH --job-name=libero_demospeedup_closed_loop_clip_bounds_released
#SBATCH --qos=eval
#SBATCH --gres=gpu:1
#SBATCH --time=3:45:00
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --output=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/liberods_%A_%a.out
#SBATCH --error=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/liberods_%A_%a.out
set -u
# LIBERO closed loop for the DemoSpeedup policy, controller clip bounds released.
# The policy was trained on summed deltas (up to 4x a single step), so it emits
# magnitudes the +-1 normalisation range would truncate for no physical reason.
SUITES=(libero_spatial libero_object libero_goal libero_10)
IDX=${SLURM_ARRAY_TASK_ID:-0}
SUITE=${SUITES[$((IDX / 10))]}
TASK_IDX=$((IDX % 10))
N_EP=${N_EP:-50}
CLIP=${CLIP_SCALE:-8}
PORT=$((12000 + IDX))
DATA=/s3data/libero_gr00t_delta/v1                   # source of the retimed training set
# /ckpt is per-AZ: the checkpoint was written by a training job on a100, so an eval
# scheduled on l40s cannot see it. Read it through the S3 mirror, which every node has.
CKPT=$(ls -d /s3ckpt/$USER/libero/demospeedup_*/checkpoint-60000 2>/dev/null | tail -1)
OUT=/ckpt/$USER/eval/libero_ds_clip${CLIP}_n${N_EP}/$SUITE
mkdir -p "$OUT"
[ -n "$CKPT" ] || { echo "[ERR] no libero demospeedup checkpoint"; exit 1; }

REPO=$HOME/quant_label_workspace/quantization_label_gr00t
CONDA=$HOME/miniconda3
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$HOME/sim/libero/server_overlay:${PYTHONPATH:-}"
cd "$REPO"
echo "[$(date '+%T')] node=$(hostname) suite=$SUITE task=$TASK_IDX clip=$CLIP ckpt=$CKPT"

"$CONDA/envs/quant_gate/bin/python" -u scripts/serve_policy.py --port=$PORT --model-path="$CKPT" \
  > "$OUT/server-$IDX.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; pkill -P $SPID 2>/dev/null' EXIT INT TERM
for i in $(seq 1 90); do grep -qiE "server|ready|listening" "$OUT/server-$IDX.log" 2>/dev/null && break; sleep 5; done
sleep 30

"$CONDA/envs/libero/bin/python" -u gr00t/eval/libero/eval_taskwise_gr00t.py \
    --args.task-suite-name $SUITE --args.video-out-path "$OUT" \
    --args.task-idx $TASK_IDX --args.num-trials-per-task $N_EP \
    --args.port $PORT --args.clip-scale $CLIP > "$OUT/eval-$IDX.log" 2>&1
echo "  client exit=$?"
kill $SPID 2>/dev/null
echo "[$(date '+%T')] === $SUITE task $TASK_IDX ==="
grep -E "clip\]|Total success rate|Success-only mean steps" "$OUT/eval-$IDX.log" "$OUT/${TASK_IDX}_results.txt" 2>/dev/null | tail -4
