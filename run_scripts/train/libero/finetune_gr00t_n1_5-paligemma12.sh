#!/bin/bash
#SBATCH --job-name=groot_n1_5_libero_paligemma12_bs32
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=batch
#SBATCH --comment="Finetune GR00T N1.5 with libero dataset using paligemma backbone"
#SBATCH --output=out/%j-groot_n1_5_libero_paligemma12_bs32.out
#SBATCH --error=out/%j-groot_n1_5_libero_paligemma12_bs32.err

HOME_DIR=$(pwd)
BASE_DIR=/virtual_lab/sjw_alinlab/taeyoung/workspace
CONDA_PATH=/virtual_lab/sjw_alinlab/taeyoung/miniconda3

export WANDB_PROJECT=GR00T

CKPT_DIR="$BASE_DIR/ckpt/groot/groot_n1_5_libero_paligemma12_bs32"

# DATA_DIR=/virtual_lab/sjw_alinlab/taeyoung/LVLA/data/bridge_orig_lerobot
# DATA_DIR=$BASE_DIR/Isaac-GR00T/demo_data/robot_sim.PickNPlace
DATA_DIR=/virtual_lab/sjw_alinlab/taeyoung/LVLA/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta

source $CONDA_PATH/bin/activate gr00t

python $BASE_DIR/Isaac-GR00T/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --data-config libero \
    --embodiment-tag libero \
    --run-name GR00T-N1.5-libero-paligemma12_bs32 \
    --num-gpus 2 \
    --batch-size 16 \
    --max-steps 60000 \
    --save-steps 10000 \
    --backbone-model-type paligemma \
    --backbone-select-layer 12 \
