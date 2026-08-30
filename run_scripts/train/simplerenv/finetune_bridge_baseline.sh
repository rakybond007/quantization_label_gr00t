#!/bin/bash
#SBATCH --job-name=groot_n1_5_bridge_base
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="GR00T N1.5 finetune on Bridge (SimplerEnv WidowX, baseline)"
#SBATCH --partition=batch
#SBATCH --output=out/%j-groot_n1_5_bridge_base.out
#SBATCH --error=out/%j-groot_n1_5_bridge_base.err

# Baseline training on Bridge / WidowX (SimplerEnv variant) with no
# multi-horizon / m8 / consistency losses. Mirrors the robocasa baseline.
# Global batch = 2 * 32 = 64.

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

export WANDB_PROJECT=GR00T-simplerenv
mkdir -p out

CKPT_DIR="$BASE_DIR/ckpt/simplerenv/bridge/groot_n1_5_bs64_baseline"
DATA_DIR="$HOME/multigpu_workspace/data/simplerenv/bridge_orig_lerobot"

source $CONDA_PATH/bin/activate gr00t

python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config simplerenv_bridge \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-bridge-baseline-fromPT-bs64 \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 60000 \
    --save-steps 10000 \
    --report-to wandb
