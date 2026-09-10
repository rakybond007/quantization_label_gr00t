#!/bin/bash
#SBATCH --job-name=robocasa_twentyfour_tasks_closed_loop_clip_released
#SBATCH --qos=eval
#SBATCH --gres=gpu:1
#SBATCH --time=3:45:00
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --output=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/rc24_%A_%a.out
#SBATCH --error=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/rc24_%A_%a.out
set -uo pipefail
# All 24 RoboCasa Kitchen tasks, laid out the way vlm_gate/run_scripts/eval/eval_robocasa_naiveK.sh
# does it: array 0-7, one policy server per element, three task clients in parallel against
# it. That amortises the ~2 min model load over three tasks instead of paying it per task.
#
# Controller clip bounds are released (clip_scale): the +-1 range is a normalisation
# artefact, and truncating a merged command the arm can execute only manufactures OOD.
# A K-step merge reaches exactly K in magnitude and SAIL merges up to max_group=8, so
# every method uses 8 and therefore identical physics.
TASK_NAMES=(
  "TurnSinkSpout" "TurnOnStove" "TurnOnSinkFaucet" "TurnOnMicrowave"
  "TurnOffStove" "TurnOffSinkFaucet" "TurnOffMicrowave" "PnPStoveToCounter"
  "PnPSinkToCounter" "PnPMicrowaveToCounter" "PnPCounterToStove" "PnPCounterToSink"
  "PnPCounterToMicrowave" "PnPCounterToCab" "PnPCabToCounter" "OpenSingleDoor"
  "OpenDrawer" "OpenDoubleDoor" "CoffeeSetupMug" "CoffeeServeMug"
  "CoffeePressButton" "CloseSingleDoor" "CloseDrawer" "CloseDoubleDoor"
)
: "${SLURM_ARRAY_TASK_ID:=0}"; : "${SLURM_ARRAY_JOB_ID:=$$}"
METHOD=${METHOD:?set METHOD: uncompressed|k2|k3|sail|demospeedup}
N_EP=${N_EP:-50}
CLIP=${CLIP_SCALE:-8}
PORT=$((13000 + (SLURM_ARRAY_JOB_ID % 80) * 10 + SLURM_ARRAY_TASK_ID))

REPO=$HOME/quant_label_workspace/quantization_label_gr00t
CONDA=$HOME/miniconda3
DATA=/s3data/robocasa_mg_gr00t_300/v1
# Eval artefacts go to the shared home filesystem, not /ckpt. /ckpt is per-AZ and a
# job landing on l40s hit "No space left on device" on that partition's copy while
# a100 had 690G free -- and results written on one AZ are invisible from the other.
# Home is visible from every node and these artefacts are small.
BASE=$HOME/eval_out/rc24_clip${CLIP}_n${N_EP}/$METHOD
mkdir -p "$BASE"

if [ "$METHOD" = "demospeedup" ]; then
  # /ckpt is per-AZ; the trained checkpoint is reachable everywhere only via the S3 mirror
  CKPT=$(ls -d /s3ckpt/$USER/robocasa/demospeedup_*/checkpoint-60000 2>/dev/null | tail -1)
else
  CKPT=$(find "$HOME/.cache/huggingface/hub/models--prehj--GR00T-N1.5-robocasa-baseline/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)
fi
[ -n "$CKPT" ] || { echo "[ERR] no checkpoint for METHOD=$METHOD"; exit 1; }

export NO_ALBUMENTATIONS_UPDATE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1
cd "$REPO"
SLOG="$BASE/server-$SLURM_ARRAY_TASK_ID.log"
echo "[$(date '+%T')] node=$(hostname) array=$SLURM_ARRAY_TASK_ID method=$METHOD clip=$CLIP port=$PORT"
echo "  ckpt=$CKPT"

"$CONDA/envs/quant_gate/bin/python" -u scripts/inference_service.py --server \
    --port $PORT --model_path "$CKPT" --data_config single_panda_gripper \
    --embodiment_tag new_embodiment --denoising_steps 4 --head main > "$SLOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; pkill -P $SPID 2>/dev/null' EXIT INT TERM
for i in $(seq 1 200); do
  grep -q "Server is ready" "$SLOG" 2>/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "[ERR] server died"; tail -25 "$SLOG"; exit 1; }
  sleep 5
done
grep -q "Server is ready" "$SLOG" || { echo "[ERR] server not ready"; exit 1; }
echo "[$(date '+%T')] server ready"

# three tasks per array element: i, i+8, i+16
SELECTED=()
for off in 0 8 16; do
  idx=$((SLURM_ARRAY_TASK_ID + off))
  [ $idx -lt 24 ] && SELECTED+=("${TASK_NAMES[$idx]}")
done
echo "  tasks: ${SELECTED[*]}"

PIDS=()
for TASK in "${SELECTED[@]}"; do
  ODIR="$BASE/$TASK"; rm -rf "$ODIR"; mkdir -p "$ODIR"
  if [ "$METHOD" = "sail" ]; then
    "$CONDA/envs/robocasa_eval/bin/python" -u scripts/robocasa_service_sail.py \
      --port $PORT --host localhost --env_name "$TASK" --video_dir "$ODIR" --seed 42 \
      --n_episodes $N_EP --max_episode_steps 1500 --generative_textures --no_record_video \
      --sail --agg_mag_thresh 1.0 --agg_rot_thresh 1.0 --clip-scale $CLIP \
      > "$ODIR/client.log" 2>&1 &
  else
    K=1; [ "$METHOD" = "k2" ] && K=2; [ "$METHOD" = "k3" ] && K=3
    PYTHONPATH="$REPO/vlm_gate/scripts:${PYTHONPATH:-}" \
    "$CONDA/envs/robocasa_eval/bin/python" -u vlm_gate/scripts/robocasa_service_compress.py \
      --port $PORT --host localhost --env_name "$TASK" --video_dir "$ODIR" --seed 42 \
      --n_episodes $N_EP --max_episode_steps 1500 --generative_textures \
      --compress-k $K --clip-scale $CLIP > "$ODIR/client.log" 2>&1 &
  fi
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
kill $SPID 2>/dev/null
echo "[$(date '+%T')] === array $SLURM_ARRAY_TASK_ID / $METHOD ==="
for TASK in "${SELECTED[@]}"; do
  f="$BASE/$TASK/prediction.txt"
  echo "  $TASK: $([ -f "$f" ] && grep -cE '^episode' "$f" || echo 0) eps  $(grep -E '^is_success' "$f" 2>/dev/null | tail -1)"
done
