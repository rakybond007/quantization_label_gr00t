#!/bin/bash
#SBATCH --job-name=eval_groot
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --comment="Evaluate paligemma-backbone Gr00T on libero tasks"
#SBATCH --array=0-9
#SBATCH --output=out/%j-eval_groot.out
#SBATCH --error=out/%j-eval_groot.err

HOME_DIR=$(pwd)
BASE_DIR=/virtual_lab/sjw_alinlab/taeyoung/workspace
CONDA_PATH=/virtual_lab/sjw_alinlab/taeyoung/miniconda3

export WANDB_PROJECT=GR00T

CKPT_DIR="$BASE_DIR/ckpt/groot"
CKPT_STEP=60000

CKPT_PATH="$CKPT_DIR/groot_n1_5_libero_paligemma12_bs32/checkpoint-$CKPT_STEP"


echo "[i] Evaluating: CKPT_STEP=$CKPT_STEP..."

"$CONDA_PREFIX"/envs/gr00t/bin/python Isaac-GR00T/scripts/serve_policy.py \
    --port=875$SLURM_ARRAY_TASK_ID \
    --model-path=$CKPT_PATH \
    --backbone-model-type="paligemma" &
SERVE_PID=$!



TASK_NAMES=("libero_10" "libero_goal" "libero_object" "libero_spatial")
# TASK_NAMES=("libero_goal_new")
# TASK_NAMES=("libero_spatial_new")
# TASK_NAMES=("libero_spatial")
MAIN_PIDS=()

sleep 10

for TASK_NAME in "${TASK_NAMES[@]}"; do
    OUTPUT_DIR="$BASE_DIR/output/groot/$TASK_NAME/groot_n1_5_libero_paligemma12_bs32/$CKPT_STEP"
    mkdir -p "$OUTPUT_DIR"
    "$CONDA_PREFIX"/envs/libero/bin/python Isaac-GR00T/gr00t/eval/libero/eval_taskwise_gr00t.py \
        --args.task-suite-name $TASK_NAME \
        --args.video-out-path $OUTPUT_DIR \
        --args.task_idx=$SLURM_ARRAY_TASK_ID \
        --args.port=875$SLURM_ARRAY_TASK_ID \
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