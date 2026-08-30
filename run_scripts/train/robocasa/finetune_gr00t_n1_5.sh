#!/bin/bash
#SBATCH --job-name=groot_n1_5_robocasa
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="GR00T N1.5 finetune on Robocasa dataset"
#SBATCH --partition=batch
#SBATCH --output=out/%j-groot_n1_5_libero.out
#SBATCH --error=out/%j-groot_n1_5_libero.err

HOME_DIR=$(pwd)
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

export WANDB_PROJECT=GR00T-robocasa

CKPT_DIR="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64"

# DATA_DIR=/virtual_lab/sjw_alinlab/taeyoung/LVLA/data/bridge_orig_lerobot
# DATA_DIR=$BASE_DIR/Isaac-GR00T/demo_data/robot_sim.PickNPlace
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"

source $CONDA_PATH/bin/activate gr00t

python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config single_panda_gripper \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-robocasa-fromPT \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 60000 \
    --save-steps 10000 
