#!/bin/bash
#SBATCH --job-name=groot_n1_5_robocasa_base
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="GR00T N1.5 finetune on Robocasa (baseline, no multi-horizon)"
#SBATCH --partition=batch
#SBATCH --output=out/%j-groot_n1_5_robocasa_base.out
#SBATCH --error=out/%j-groot_n1_5_robocasa_base.err

# Baseline training: identical hyperparameters to
# finetune_gr00t_n1_5_multi_horizon.sh, but WITHOUT the multi-horizon
# auxiliary decoder heads or extended action fetch.
# Global batch = 2 * 32 = 64

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

export WANDB_PROJECT=GR00T-robocasa
mkdir -p out

CKPT_DIR="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_baseline"
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"

source $CONDA_PATH/bin/activate gr00t

python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config single_panda_gripper \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-robocasa-baseline-fromPT-bs64 \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 60000 \
    --save-steps 10000 \
    --report-to wandb
