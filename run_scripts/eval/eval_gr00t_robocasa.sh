#!/bin/bash
#SBATCH --job-name=eval_rc_groot_robocasa
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --comment="GR00T N1.5 evaluation on Robocasa"
#SBATCH --partition=background
#SBATCH --array=0-7
#SBATCH --output=out/%j-eval_groot_robocasa.out
#SBATCH --error=out/%j-eval_groot_robocasa.err

HOME_DIR=$(pwd)
BASE_DIR=/virtual_lab/sjw_alinlab/taeyoung/workspace
CONDA_PATH=/virtual_lab/sjw_alinlab/taeyoung/miniconda3

CKPT_DIR="$BASE_DIR/ckpt/robocasa/groot"
CKPT_STEP=60000

CKPT_NAME="groot_n1_5_bs32"
CKPT_PATH="$CKPT_DIR/$CKPT_NAME/checkpoint-$CKPT_STEP"

echo "[i] Evaluating: CKPT_NAME=$CKPT_NAME, CKPT_STEP=$CKPT_STEP..."

"$CONDA_PREFIX"/envs/robocasa/bin/python $BASE_DIR/Isaac-GR00T/scripts/inference_service.py --server \
    --port 897$SLURM_ARRAY_TASK_ID \
    --model_path $CKPT_PATH \
    --data_config single_panda_gripper \
    --embodiment_tag new_embodiment &
SERVE_PID=$!



TASK_NAMES=(
  "TurnSinkSpout"
  "TurnOnStove"
  "TurnOnSinkFaucet"
  "TurnOnMicrowave"
  "TurnOffStove"
  "TurnOffSinkFaucet"
  "TurnOffMicrowave"
  "PnPStoveToCounter" #7
  "PnPSinkToCounter"
  "PnPMicrowaveToCounter"
  "PnPCounterToStove"
  "PnPCounterToSink"
  "PnPCounterToMicrowave"
  "PnPCounterToCab"
  "PnPCabToCounter"
  "OpenSingleDoor"
  "OpenDrawer"
  "OpenDoubleDoor"
  "CoffeeSetupMug"
  "CoffeeServeMug"
  "CoffeePressButton"
  "CloseSingleDoor"
  "CloseDrawer"
  "CloseDoubleDoor" # 23
) # 24 tasks in total

MAIN_PIDS=()

# Select tasks at specific indices based on array task ID
SELECTED_TASKS=()
if [ $SLURM_ARRAY_TASK_ID -lt 8 ]; then
    SELECTED_TASKS+=("${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}")
fi
if [ $((SLURM_ARRAY_TASK_ID + 8)) -lt ${#TASK_NAMES[@]} ]; then
    SELECTED_TASKS+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 8))]}")
fi
if [ $((SLURM_ARRAY_TASK_ID + 16)) -lt ${#TASK_NAMES[@]} ]; then
    SELECTED_TASKS+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 16))]}")
fi
TASK_NAMES=("${SELECTED_TASKS[@]}")

echo "[i] Running tasks: ${TASK_NAMES[@]}"
for TASK_NAME in "${TASK_NAMES[@]}"; do
    OUTPUT_DIR="$BASE_DIR/output/groot/robocasa/$CKPT_NAME/$CKPT_STEP/$TASK_NAME"
    mkdir -p "$OUTPUT_DIR"
    "$CONDA_PREFIX"/envs/robocasa/bin/python $BASE_DIR/Isaac-GR00T/scripts/robocasa_service.py --client \
        --port 897$SLURM_ARRAY_TASK_ID \
        --env_name $TASK_NAME \
        --video_dir $OUTPUT_DIR \
        --max_episode_steps 750 \
        --n_episodes 50 \
        --generative_textures \
        >& "$OUTPUT_DIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done


# Wait on just the main.py processes
for pid in "${MAIN_PIDS[@]}"; do
    wait "$pid"
done

# Kill serve_policy once those tasks finish
kill "$SERVE_PID"
echo "[i] Finished CKPT_STEP=$CKPT_STEP on GPU $SLURM_ARRAY_TASK_ID."
