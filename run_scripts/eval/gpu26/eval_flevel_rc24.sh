#!/bin/bash
#SBATCH --job-name=flevel_ratio_robocasa_twentyfour_tasks_closed_loop_clip_released
#SBATCH --gres=gpu:1
#SBATCH --qos=eval
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=12
#SBATCH --array=0-7
#SBATCH --output=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/flrc24_%A_%a.out
#SBATCH --error=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/flrc24_%A_%a.out
set -uo pipefail

# 24 RoboCasa tasks over 8 array elements (i, i+8, i+16), same grouping the
# DemoSpeedup and ISR arms used, so the numbers line up task for task.
#
# MODE   = ratio_future | ratio_current   (which trained arm)
# ARM    = readout | fine                 (readout lets the classifier pick the
#                                          speed; fine forces K=1 -- the
#                                          all-fine reference inside the same
#                                          checkpoint, per docs/ratio_flevel)
# Controller clip bounds released (x8) to match the other arms: a merged row of
# a K>1 stream routinely leaves +-1 and that range is a normalisation artefact.
MODE=${MODE:-ratio_future}
ARM=${ARM:-readout}
N_EP=${N_EP:-50}
CLIP=${CLIP_SCALE:-8}
STEP=${STEP:-60000}

TASK_NAMES=(TurnSinkSpout TurnOnStove TurnOnSinkFaucet TurnOnMicrowave TurnOffStove
            TurnOffSinkFaucet TurnOffMicrowave PnPStoveToCounter PnPSinkToCounter
            PnPMicrowaveToCounter PnPCounterToStove PnPCounterToSink PnPCounterToMicrowave
            PnPCounterToCab PnPCabToCounter OpenSingleDoor OpenDrawer OpenDoubleDoor
            CoffeeSetupMug CoffeeServeMug CoffeePressButton CloseSingleDoor CloseDrawer
            CloseDoubleDoor)

CKPT=/s3ckpt/$USER/flevel/ratio_comparison_seed42/$MODE/checkpoint-$STEP
FL=/fsx/home/hojin/quant_label_workspace/F_level
QR=/fsx/home/hojin/quant_label_workspace/quantization_label_gr00t
CONDA=$HOME/miniconda3
FORCE=0; [ "$ARM" = "fine" ] && FORCE=1
BASE=$HOME/eval_out/flrc24_${MODE}_${ARM}_clip${CLIP}_n${N_EP}
PORT=$((17000 + (SLURM_ARRAY_JOB_ID % 1000) * 10 + SLURM_ARRAY_TASK_ID))

echo "[$(date '+%T')] node=$(hostname) array=$SLURM_ARRAY_TASK_ID mode=$MODE arm=$ARM clip=$CLIP port=$PORT"
test -d "$CKPT" || { echo "[ERR] no checkpoint: $CKPT"; exit 1; }
mkdir -p "$BASE"

SLOG="$BASE/server_${SLURM_ARRAY_TASK_ID}.log"
PYTHONPATH="$FL/papers/reproducing/FLARE:$FL" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
NO_ALBUMENTATIONS_UPDATE=1 \
"/fsx/home/hojin/quant_label_workspace/environments/flevel_hj/bin/python" -u \
  "$FL/papers/reproducing/FLARE/scripts/serve_flevel_robocasa.py" \
  --model-path "$CKPT" --port $PORT --mode "$MODE" --force-level $FORCE > "$SLOG" 2>&1 &
SPID=$!
# Loading the 3 checkpoint shards straight off the /s3ckpt S3 mount was
# measured at ~211 s per shard (~11 min); the old 7.5 min wait gave up on a
# server that was still loading fine.
for i in $(seq 1 360); do
  grep -q "Server is ready" "$SLOG" 2>/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "[ERR] server died"; tail -25 "$SLOG"; exit 1; }
  [ $((i % 24)) -eq 0 ] && echo "[wait] server loading... $((i*5))s"
  [ $((i % 24)) -eq 0 ] && echo "[wait] server still loading... $((i*5))s"
  sleep 5
done
grep -q "Server is ready" "$SLOG" || { echo "[ERR] server not ready"; tail -40 "$SLOG"; kill $SPID; exit 1; }
echo "[$(date '+%T')] server ready"

SELECTED=()
for off in 0 8 16; do
  idx=$((SLURM_ARRAY_TASK_ID + off))
  [ $idx -lt 24 ] && SELECTED+=("${TASK_NAMES[$idx]}")
done
echo "  tasks: ${SELECTED[*]}"

cd "$QR"
PIDS=()
for TASK in "${SELECTED[@]}"; do
  ODIR="$BASE/$TASK"
  # A requeue must not resume a partially-recorded task: the client counts
  # completed episodes from prediction.txt, so a stale file would silently
  # shorten the run.  Start each task's record clean.
  rm -rf "$ODIR"; mkdir -p "$ODIR"
  MUJOCO_GL=egl NO_ALBUMENTATIONS_UPDATE=1 \
  "$CONDA/envs/robocasa_eval/bin/python" -u scripts/robocasa_service_flevel.py \
    --port $PORT --host localhost --env_name "$TASK" --video_dir "$ODIR" --seed 42 \
    --n_episodes $N_EP --max_episode_steps 1500 --generative_textures --no_record_video \
    --clip-scale $CLIP > "$ODIR/client.log" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
kill $SPID 2>/dev/null

echo "[$(date '+%T')] === array $SLURM_ARRAY_TASK_ID / $MODE / $ARM ==="
for TASK in "${SELECTED[@]}"; do
  f="$BASE/$TASK/prediction.txt"
  echo "  $TASK: $([ -f "$f" ] && grep -cE '^episode' "$f" || echo 0) eps  $(grep -E '^is_success' "$f" 2>/dev/null | tail -1)  $(grep -E '^flevel_levels' "$f" 2>/dev/null | tail -1)"
done
