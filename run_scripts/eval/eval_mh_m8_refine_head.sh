#!/bin/bash
#SBATCH --job-name=eval_mh_m8_refine
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --comment="m8 model (no consist) eval with m8 head (native 8-step compressed)"
#SBATCH --partition=background
#SBATCH --array=0-7
#SBATCH --output=out/%A_%a-eval_mh_m8_refine.out
#SBATCH --error=out/%A_%a-eval_mh_m8_refine.err
#SBATCH --time=2-00:00:00

# m8 (no econsist) checkpoint, head=m8 (native 8-step decoder, model-side compression).

set -u

PORT=$((8930 + SLURM_ARRAY_TASK_ID))
N_EPISODES=50

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_mh_m8_econsist_discfix_refine/checkpoint-10000"
OUTPUT_BASE="$BASE_DIR/output/robocasa/mh_m8_refine/m8_refined"
mkdir -p out "$OUTPUT_BASE"

cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "[i] Array $SLURM_ARRAY_TASK_ID | mh_m8_refine m8_refined head | port=$PORT"

"$CONDA_PATH/envs/gr00t/bin/python" "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT \
    --model_path "$CKPT" \
    --data_config single_panda_gripper \
    --embodiment_tag new_embodiment \
    --denoising_steps 4 \
    --head m8_refined &
SPID=$!
sleep 60

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
    "$CONDA_PATH/envs/robocasa_gr00t/bin/python" "$BASE_DIR/scripts/robocasa_service.py" --client \
        --port $PORT --host localhost \
        --env_name "$TASK" \
        --video_dir "$ODIR" \
        --seed 42 \
        --n_episodes $N_EPISODES \
        --max_episode_steps 1500 \
        --action_horizon 8 \
        --generative_textures \
        >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done

for pid in "${MAIN_PIDS[@]}"; do
    wait "$pid"
done

kill "$SPID" 2>/dev/null
echo "[i] Array $SLURM_ARRAY_TASK_ID done."
