#!/bin/bash
#SBATCH --job-name=flevel_two_decoder_conf_robocasa_twentyfour_tasks_closed_loop
#SBATCH --gres=gpu:1
#SBATCH --qos=eval
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=12
#SBATCH --array=0-7
#SBATCH --output=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/flconf24_%A_%a.out
#SBATCH --error=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/flconf24_%A_%a.out
set -uo pipefail

# Closed-loop RoboCasa eval for the TWO-DECODER conf checkpoints.
#
# Difference from eval_flevel_rc24.sh (the 4-decoder ratio arms):
#   * the readout regresses a conf scalar instead of a 4-way class, so the
#     server runs `--mode conf` (data config `robocasa_conf`, which drops the
#     class-range check on the carrier), and
#   * the compress/fine decision is `conf >= TAU`.  TAU is an eval knob, not a
#     trained parameter, so one checkpoint sweeps the whole speed/success curve
#     -- pass TAU to move along it.
#
# VARIANT = vlm | vlm_contact | contact   (which label the run trained on)
# DEC     = k2 | k25                      (compression decoder: K=2 / K=2.5)
# ARM     = readout | fine                (fine forces K=1: the all-fine
#                                          reference inside the same checkpoint)
# TAU     = threshold; empty keeps the checkpoint's own value (0.5)
# TASK_LIST = space-separated task names; set it to run a smoke on a subset
#             instead of this array element's (i, i+8, i+16) triple.
VARIANT=${VARIANT:-vlm}
DEC=${DEC:-k2}
ARM=${ARM:-readout}
TAU=${TAU:-}
N_EP=${N_EP:-50}
# NOCLIP=1 removes arm delta saturation outright instead of widening the bound.
# Those 6 axes (eef_pos + eef_rot) are the only ones compression sums: RoboCasa
# base_motion is identically zero in the data (stats min=max=0) and gripper /
# control_mode latch the block's last value rather than summing, so with the arm
# cap gone nothing that moves is clipped and CLIP stays at 1 (bounds untouched).
NOCLIP=${NOCLIP:-1}
CLIP=${CLIP_SCALE:-1.0}
EXTRA=(); [ "$NOCLIP" = 1 ] && EXTRA+=(--no-action-clip)
STEP=${STEP:-60000}

TASK_NAMES=(TurnSinkSpout TurnOnStove TurnOnSinkFaucet TurnOnMicrowave TurnOffStove
            TurnOffSinkFaucet TurnOffMicrowave PnPStoveToCounter PnPSinkToCounter
            PnPMicrowaveToCounter PnPCounterToStove PnPCounterToSink PnPCounterToMicrowave
            PnPCounterToCab PnPCabToCounter OpenSingleDoor OpenDrawer OpenDoubleDoor
            CoffeeSetupMug CoffeeServeMug CoffeePressButton CloseSingleDoor CloseDrawer
            CloseDoubleDoor)

CKPT=/s3ckpt/$USER/flevel/conf_${DEC}_${VARIANT}_seed42/checkpoint-$STEP
FL=/fsx/home/hojin/quant_label_workspace/F_level
QR=/fsx/home/hojin/quant_label_workspace/quantization_label_gr00t
CONDA=$HOME/miniconda3
FORCE=0; [ "$ARM" = "fine" ] && FORCE=1
TAG=${TAU:+_tau$TAU}
# RESUME=1 keeps whatever a task already recorded and only runs the episodes
# still missing -- the client counts completed episodes out of prediction.txt.
# OUT_BASE points the run at an existing result set (e.g. one started under a
# different clip setting); without it the name is derived as usual.
RESUME=${RESUME:-0}
CTAG=$([ "$NOCLIP" = 1 ] && echo noclip || echo clip$CLIP)
BASE=${OUT_BASE:-$HOME/eval_out/flconf24_${DEC}_${VARIANT}_${ARM}${TAG}_${CTAG}_n${N_EP}}
PORT=$((18000 + (SLURM_ARRAY_JOB_ID % 1000) * 10 + SLURM_ARRAY_TASK_ID))

echo "[$(date '+%T')] node=$(hostname) array=$SLURM_ARRAY_TASK_ID dec=$DEC variant=$VARIANT arm=$ARM tau=${TAU:-ckpt} noclip=$NOCLIP clip=$CLIP port=$PORT"
test -d "$CKPT" || { echo "[ERR] no checkpoint: $CKPT"; exit 1; }
mkdir -p "$BASE"

SLOG="$BASE/server_${SLURM_ARRAY_TASK_ID}.log"
PYTHONPATH="$FL/papers/reproducing/FLARE:$FL" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
NO_ALBUMENTATIONS_UPDATE=1 \
"/fsx/home/hojin/quant_label_workspace/environments/flevel_hj/bin/python" -u \
  "$FL/papers/reproducing/FLARE/scripts/serve_flevel_robocasa.py" \
  --model-path "$CKPT" --port $PORT --mode conf --force-level $FORCE \
  ${TAU:+--conf-threshold $TAU} > "$SLOG" 2>&1 &
SPID=$!
# Shards load off the /s3ckpt S3 mount at ~211 s each (~11 min all in).
for i in $(seq 1 360); do
  grep -q "Server is ready" "$SLOG" 2>/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "[ERR] server died"; tail -25 "$SLOG"; exit 1; }
  [ $((i % 24)) -eq 0 ] && echo "[wait] server loading... $((i*5))s"
  sleep 5
done
grep -q "Server is ready" "$SLOG" || { echo "[ERR] server not ready"; tail -40 "$SLOG"; kill $SPID; exit 1; }
echo "[$(date '+%T')] server ready"
grep -E "conf mode:|conf_threshold" "$SLOG"    # what tau actually took effect

SELECTED=()
if [ -n "${TASK_LIST:-}" ]; then
  read -r -a SELECTED <<< "$TASK_LIST"
else
  for off in 0 8 16; do
    idx=$((SLURM_ARRAY_TASK_ID + off))
    [ $idx -lt 24 ] && SELECTED+=("${TASK_NAMES[$idx]}")
  done
fi
echo "  tasks: ${SELECTED[*]}"

cd "$QR"
PIDS=()
for TASK in "${SELECTED[@]}"; do
  ODIR="$BASE/$TASK"
  if [ "$RESUME" = 1 ]; then
    # Record how much of this task predates the resume, so a mixed result set
    # can still say which episodes ran under which setting.
    mkdir -p "$ODIR"
    done_n=$(grep -cE "^episode" "$ODIR/prediction.txt" 2>/dev/null || echo 0)
    printf '%s resume: %s episodes already recorded; the rest run with noclip=%s clip_scale=%s\n' \
      "$(date '+%F %T')" "$done_n" "$NOCLIP" "$CLIP" >> "$ODIR/clip_provenance.txt"
    echo "  [resume] $TASK: $done_n done, running up to $N_EP"
    REDIR=append
  else
    # A requeue must not resume a partially-recorded task: the client counts
    # completed episodes from prediction.txt, so a stale file would silently
    # shorten the run.
    rm -rf "$ODIR"; mkdir -p "$ODIR"
    REDIR=truncate
  fi
  # Truncating client.log on a resume would throw away the per-episode
  # `levels=` lines the finished episodes wrote -- that is the only record of
  # what speed they ran at.
  if [ "$REDIR" = append ]; then
    MUJOCO_GL=egl NO_ALBUMENTATIONS_UPDATE=1 \
    "$CONDA/envs/robocasa_eval/bin/python" -u scripts/robocasa_service_flevel.py \
      --port $PORT --host localhost --env_name "$TASK" --video_dir "$ODIR" --seed 42 \
      --n_episodes $N_EP --max_episode_steps 1500 --generative_textures --no_record_video \
      --clip-scale $CLIP ${EXTRA[@]+"${EXTRA[@]}"} >> "$ODIR/client.log" 2>&1 &
  else
    MUJOCO_GL=egl NO_ALBUMENTATIONS_UPDATE=1 \
    "$CONDA/envs/robocasa_eval/bin/python" -u scripts/robocasa_service_flevel.py \
      --port $PORT --host localhost --env_name "$TASK" --video_dir "$ODIR" --seed 42 \
      --n_episodes $N_EP --max_episode_steps 1500 --generative_textures --no_record_video \
      --clip-scale $CLIP ${EXTRA[@]+"${EXTRA[@]}"} > "$ODIR/client.log" 2>&1 &
  fi
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
kill $SPID 2>/dev/null

echo "[$(date '+%T')] === array $SLURM_ARRAY_TASK_ID / $DEC / $VARIANT / $ARM / tau=${TAU:-ckpt} ==="
for TASK in "${SELECTED[@]}"; do
  f="$BASE/$TASK/prediction.txt"
  echo "  $TASK: $([ -f "$f" ] && grep -cE '^episode' "$f" || echo 0) eps  $(grep -E '^is_success' "$f" 2>/dev/null | tail -1)  $(grep -E '^flevel_levels' "$f" 2>/dev/null | tail -1)"
done
