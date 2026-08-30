#!/bin/bash
# Generic selective-compression eval. Configured via env vars:
#   CKPT_TAG    : "mh_m8" or "per_expert_moe"
#   SCORE_MODE  : "self_agree" | "entropy" | "hybrid"
#   N_SAMPLES   : N for head=selective (1 for self_agree; >=2 for entropy/hybrid)
#   TAU_VALUE   : (Mode α) absolute threshold; pair k compressed iff score_k < TAU
#   K_PREFIX    : (Mode β) compress first K pairs unconditionally (overrides TAU_VALUE)
#   ALPHA       : weight on entropy in hybrid (default 0.5)
#
# Pass exactly one of TAU_VALUE or K_PREFIX. K_PREFIX takes precedence.
#
# 24 tasks split into 8 array jobs × 3 tasks each. Same task assignment as
# eval_robocasa_moe4_per_expert.sh.
#SBATCH --job-name=eval_selective
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --array=0-7
#SBATCH --output=out/%A_%a-eval_selective.out
#SBATCH --error=out/%A_%a-eval_selective.err
#SBATCH --time=2-00:00:00
#SBATCH --comment="Selective per-pair compression eval (variable-length 8..16-step chunks)"

set -u

: "${CKPT_TAG:?need CKPT_TAG}"
: "${SCORE_MODE:?need SCORE_MODE}"
: "${N_SAMPLES:?need N_SAMPLES}"
TAU_VALUE=${TAU_VALUE:-}
K_PREFIX=${K_PREFIX:-}
if [ -z "$TAU_VALUE" ] && [ -z "$K_PREFIX" ]; then
    echo "[ERROR] need one of TAU_VALUE or K_PREFIX"; exit 1
fi
ALPHA=${ALPHA:-0.5}

# Map ckpt tag to path. PORT base is shifted per (CKPT_TAG, SCORE_MODE) so 6 sbatch
# arrays can run concurrently without colliding.
case "$CKPT_TAG" in
    mh_m8)
        CKPT="$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_mh_m8_discfix/checkpoint-60000"
        PORT_BASE_CKPT=9000
        ;;
    per_expert_moe)
        CKPT="$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_moe4_per_expert_body/checkpoint-60000"
        PORT_BASE_CKPT=9100
        ;;
    *)
        echo "[ERROR] unknown CKPT_TAG=$CKPT_TAG"; exit 1 ;;
esac
case "$SCORE_MODE" in
    self_agree) PORT_OFF=0 ;;
    entropy)    PORT_OFF=20 ;;
    hybrid)     PORT_OFF=40 ;;
    *) echo "[ERROR] unknown SCORE_MODE=$SCORE_MODE"; exit 1 ;;
esac

PORT=$((PORT_BASE_CKPT + PORT_OFF + SLURM_ARRAY_TASK_ID))
N_EPISODES=50

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
if [ -n "$K_PREFIX" ]; then
    OUTPUT_BASE="$BASE_DIR/output/robocasa/selective_${CKPT_TAG}_kpref${K_PREFIX}"
    HP_FLAG="--k-prefix $K_PREFIX"
else
    TAU_LBL=${TAU_VALUE//./p}
    OUTPUT_BASE="$BASE_DIR/output/robocasa/selective_${CKPT_TAG}_${SCORE_MODE}_tau${TAU_LBL}"
    HP_FLAG="--tau $TAU_VALUE"
fi
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "[i] Array $SLURM_ARRAY_TASK_ID | $CKPT_TAG / $SCORE_MODE / N=$N_SAMPLES / hp=$HP_FLAG | port=$PORT"

PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT \
    --model_path "$CKPT" \
    --data_config single_panda_gripper \
    --embodiment_tag new_embodiment \
    --denoising_steps 4 \
    --head selective \
    --n_samples "$N_SAMPLES" \
    --discrete-action-dims 6 11 \
    > "$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log" 2>&1 &
SPID=$!
sleep 60
for i in $(seq 1 60); do
    grep -q "Server is ready" "$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log" 2>/dev/null && break
    sleep 5
done

TASK_NAMES=(
  "TurnSinkSpout" "TurnOnStove" "TurnOnSinkFaucet" "TurnOnMicrowave"
  "TurnOffStove" "TurnOffSinkFaucet" "TurnOffMicrowave" "PnPStoveToCounter"
  "PnPSinkToCounter" "PnPMicrowaveToCounter" "PnPCounterToStove" "PnPCounterToSink"
  "PnPCounterToMicrowave" "PnPCounterToCab" "PnPCabToCounter" "OpenSingleDoor"
  "OpenDrawer" "OpenDoubleDoor" "CoffeeSetupMug" "CoffeeServeMug"
  "CoffeePressButton" "CloseSingleDoor" "CloseDrawer" "CloseDoubleDoor"
)
SELECTED=()
[ $SLURM_ARRAY_TASK_ID -lt 8 ] && SELECTED+=("${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}")
[ $((SLURM_ARRAY_TASK_ID + 8)) -lt 24 ] && SELECTED+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 8))]}")
[ $((SLURM_ARRAY_TASK_ID + 16)) -lt 24 ] && SELECTED+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 16))]}")

echo "[i] Tasks: ${SELECTED[*]}"

MAIN_PIDS=()
for TASK in "${SELECTED[@]}"; do
    ODIR="$OUTPUT_BASE/$TASK"
    mkdir -p "$ODIR"
    PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
        "$BASE_DIR/scripts/robocasa_service_selective.py" \
        --port $PORT --host localhost \
        --env_name "$TASK" \
        --video_dir "$ODIR" \
        --seed 42 \
        --n_episodes $N_EPISODES \
        --max_episode_steps 1500 \
        --generative_textures \
        --score-mode "$SCORE_MODE" \
        $HP_FLAG \
        --alpha "$ALPHA" \
        >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done

for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
kill "$SPID" 2>/dev/null
echo "[i] Array $SLURM_ARRAY_TASK_ID done."
