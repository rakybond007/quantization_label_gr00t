#!/bin/bash
#SBATCH --job-name=libero_forty_tasks_closed_loop_clip_released
#SBATCH --qos=eval
#SBATCH --gres=gpu:1
#SBATCH --time=3:45:00
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --output=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/lb40_%A_%a.out
#SBATCH --error=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/lb40_%A_%a.out
set -uo pipefail
# All 40 LIBERO tasks, laid out the way run_scripts/eval/eval_libero_baseline.sh does it:
# array 0-9 selects the task index within each suite, and one policy server serves four
# suite clients running in parallel. 10 elements x 4 suites = 40 tasks, and the ~2 min
# model load is paid ten times instead of forty.
#
# Clip bounds released: the DemoSpeedup policy is trained on summed deltas and emits
# magnitudes the +-1 normalisation range would truncate for no physical reason.
: "${SLURM_ARRAY_TASK_ID:=0}"; : "${SLURM_ARRAY_JOB_ID:=$$}"
METHOD=${METHOD:-demospeedup}
N_EP=${N_EP:-50}
CLIP=${CLIP_SCALE:-8}
PORT=$((14000 + (SLURM_ARRAY_JOB_ID % 80) * 10 + SLURM_ARRAY_TASK_ID))

REPO=$HOME/quant_label_workspace/quantization_label_gr00t
CONDA=$HOME/miniconda3
DATA=/s3data/libero_gr00t_delta/v1
# Eval artefacts go to the shared home filesystem, not /ckpt. /ckpt is per-AZ and a
# job landing on l40s hit "No space left on device" on that partition's copy while
# a100 had 690G free -- and results written on one AZ are invisible from the other.
# Home is visible from every node and these artefacts are small.
BASE=$HOME/eval_out/lb40_clip${CLIP}_n${N_EP}/$METHOD
mkdir -p "$BASE"

if [ "$METHOD" = "demospeedup" ]; then
  # /ckpt is per-AZ; only the S3 mirror is visible from whichever partition we land on
  CKPT=$(ls -d /s3ckpt/$USER/libero/demospeedup_*/checkpoint-60000 2>/dev/null | tail -1)
else
  CKPT=$(find "$HOME/.cache/huggingface/hub/models--prehj--GR00T-N1.5-libero-baseline-bs32-60k/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)
fi
[ -n "$CKPT" ] || { echo "[ERR] no checkpoint for METHOD=$METHOD"; exit 1; }

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$HOME/sim/libero/server_overlay:${PYTHONPATH:-}"
cd "$REPO"
SLOG="$BASE/server-$SLURM_ARRAY_TASK_ID.log"
echo "[$(date '+%T')] node=$(hostname) array=$SLURM_ARRAY_TASK_ID method=$METHOD clip=$CLIP port=$PORT"
echo "  ckpt=$CKPT"

"$CONDA/envs/quant_gate/bin/python" -u scripts/serve_policy.py --port=$PORT --model-path="$CKPT" \
  > "$SLOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; pkill -P $SPID 2>/dev/null' EXIT INT TERM
for i in $(seq 1 120); do
  kill -0 $SPID 2>/dev/null || { echo "[ERR] server died"; tail -25 "$SLOG"; exit 1; }
  grep -qiE "ready|listening|serving" "$SLOG" 2>/dev/null && break
  sleep 5
done
sleep 45
echo "[$(date '+%T')] server up"

SUITES=("libero_10" "libero_goal" "libero_object" "libero_spatial")
PIDS=()
for SUITE in "${SUITES[@]}"; do
  ODIR="$BASE/$SUITE"; mkdir -p "$ODIR"
  # Clear this task's videos as well as its results file. The client skips any episode
  # whose mp4 already exists and, on that path, bumps the totals without appending to
  # ep_records -- so a requeued element silently reports a success rate with
  # "Success-only mean steps: 0.00 (over 0 ep)". Removing only the results file is not
  # enough, and --requeue makes this reachable at any time.
  rm -f "$ODIR/${SLURM_ARRAY_TASK_ID}_results.txt"
  find "$ODIR" -maxdepth 1 -name "rollout_*_${SLURM_ARRAY_TASK_ID}_*.mp4" -delete 2>/dev/null || true
  "$CONDA/envs/libero/bin/python" -u gr00t/eval/libero/eval_taskwise_gr00t.py \
      --args.task-suite-name $SUITE --args.video-out-path "$ODIR" \
      --args.task-idx $SLURM_ARRAY_TASK_ID --args.num-trials-per-task $N_EP \
      --args.port $PORT --args.clip-scale $CLIP \
      > "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
kill $SPID 2>/dev/null
echo "[$(date '+%T')] === array $SLURM_ARRAY_TASK_ID / $METHOD ==="
for SUITE in "${SUITES[@]}"; do
  r="$BASE/$SUITE/${SLURM_ARRAY_TASK_ID}_results.txt"
  echo "  $SUITE task $SLURM_ARRAY_TASK_ID: $(grep -hE 'Total success rate|Success-only mean steps' "$r" 2>/dev/null | tr '\n' ' ')"
done
