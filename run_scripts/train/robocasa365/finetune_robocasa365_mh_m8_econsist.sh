#!/bin/bash
#SBATCH --job-name=groot_n1_5_robocasa365_mh_m8_econsist
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="GR00T N1.5 finetune on RoboCasa365: multi-horizon + discrete-fix + merged_8 + ensemble-consistency"
#SBATCH --partition=batch
#SBATCH --output=out/%j-groot_n1_5_robocasa365_mh_m8_econsist.out
#SBATCH --error=out/%j-groot_n1_5_robocasa365_mh_m8_econsist.err

# RoboCasa365 mh_m8 + ensemble-consistency finetune.
# Mirrors robocasa/finetune_gr00t_n1_5_mh_m8_econsist.sh.
# Loss: 1.0*L_main16 + 1.0*L_m8 + warmup*(1.0*L_f2 + 1.0*L_f4) + c_warmup*0.1*L_consist
# where L_consist = MSE(pair_sum(v_main), v_f2[:8]) + MSE(v_m8, v_f2[:8]), shared-t.
#
# Global batch = 2 * 32 = 64
# NOTE: dataset path currently points to the 24-atomic-task subset
#       (kimtaey/robocasa_mg_gr00t_300). Swap to the LeRobot-converted full
#       365-task dataset (nvidia/robocasa365-datasets) when it is staged.

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

export WANDB_PROJECT=GR00T-robocasa365
mkdir -p out

CKPT_DIR="$BASE_DIR/ckpt/robocasa365/groot/groot_n1_5_bs64_mh_m8_econsist_discfix"

# TODO: swap to full 365-task LeRobot dataset when available.
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"

source $CONDA_PATH/bin/activate gr00t

python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config single_panda_gripper_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-robocasa365-mh-m8-econsist-discfix-fromPT-bs64 \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 60000 \
    --save-steps 10000 \
    --report-to wandb \
    --use-multi-horizon-loss \
    --multi-horizon-factors 2 4 \
    --multi-horizon-loss-weights 1.0 1.0 \
    --multi-horizon-main-weight 1.0 \
    --aux-grad-scale-to-body 0.1 \
    --aux-loss-warmup-steps 5000 \
    --discrete-action-dims 6 11 \
    --use-merged-8-head \
    --merged-8-weight 1.0 \
    --use-ensemble-consistency-loss \
    --ensemble-consistency-weight 0.1 \
    --ensemble-consistency-warmup-steps 2000
