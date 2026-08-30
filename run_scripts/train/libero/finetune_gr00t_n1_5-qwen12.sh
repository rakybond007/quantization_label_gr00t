#!/bin/bash
#SBATCH --job-name=groot_n1_5_libero_qwen12
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=batch
#SBATCH --comment="Finetune GR00T N1.5 with libero dataset using Qwen backbone"
#SBATCH --output=out/%j-groot_n1_5_libero_qwen12.out
#SBATCH --error=out/%j-groot_n1_5_libero_qwen12.err

HOME_DIR=$(pwd)
BASE_DIR=/virtual_lab/sjw_alinlab/taeyoung/workspace
CONDA_PATH=/virtual_lab/sjw_alinlab/taeyoung/miniconda3

export WANDB_PROJECT=GR00T

CKPT_DIR="$BASE_DIR/ckpt/groot/groot_n1_5_libero_qwen-12"

# DATA_DIR=/virtual_lab/sjw_alinlab/taeyoung/LVLA/data/bridge_orig_lerobot
# DATA_DIR=$BASE_DIR/Isaac-GR00T/demo_data/robot_sim.PickNPlace
DATA_DIR=/virtual_lab/sjw_alinlab/taeyoung/workspace/data/huggingface/lerobot/kimtaey/libero_gr00t_delta

source $CONDA_PATH/bin/activate gr00t
# --backbone-model-type qwen2_5_vl qwen2_5_vl_7b paligemma
# --backbone-path <PATH>
python $BASE_DIR/Isaac-GR00T/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --data-config libero \
    --embodiment-tag libero \
    --run-name GR00T-N1.5-libero-qwen-12 \
    --num-gpus 2 \
    --batch-size 16 \
    --max-steps 60000 \
    --save-steps 10000 \
    --backbone-model-type qwen2_5_vl \
    --backbone-select-layer 12 \
