#!/bin/bash
#SBATCH --job-name=groot_n1_5_fractal_moe4_per_expert_body
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="GR00T N1.5 finetune on Fractal (SimplerEnv Google): 4-expert MoE (per_expert_h)"
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-groot_n1_5_fractal_moe4_per_expert_body.out
#SBATCH --error=out/%j-groot_n1_5_fractal_moe4_per_expert_body.err

# 4-expert MoE on Fractal / Google (SimplerEnv variant). Global batch = 2 GPU × 32 = 64.
# Discrete dim for fractal: 6 (gripper, single dim). 7-DoF action layout same as bridge:
#   x(0) y(1) z(2) roll(3) pitch(4) yaw(5) gripper(6)

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

export WANDB_PROJECT=GR00T-simplerenv
mkdir -p out

CKPT_DIR="$BASE_DIR/ckpt/simplerenv/fractal/groot_n1_5_bs64_moe4_per_expert_body"
DATA_DIR="$HOME/multigpu_workspace/data/simplerenv/fractal20220817_data_lerobot"

source $CONDA_PATH/bin/activate gr00t

python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config simplerenv_fractal_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-fractal-moe4-per-expert-body-fromPT-bs64 \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 60000 \
    --save-steps 10000 \
    --report-to wandb \
    --discrete-action-dims 6 \
    --use-merged-8-head \
    --merged-8-weight 1.0 \
    --use-merged-4-head \
    --merged-4-weight 1.0 \
    --use-native-8-head \
    --native-8-weight 1.0 \
    --use-moe-routing \
    --moe-body-mode=per_expert_h \
    --moe-num-experts=4 \
    --moe-expert-n-layers=2 \
    --moe-router-temp=0.5 \
    --moe-target-temp=0.3 \
    --moe-balance-weight=0.05 \
    --moe-supervise-weight=0.1 \
    --moe-router-warmup-steps=5000
