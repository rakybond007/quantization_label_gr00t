#!/bin/bash
#SBATCH --job-name=groot_n1_5_robocasa_mh_discfix
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="GR00T N1.5 finetune on Robocasa with multi-horizon + discrete-dim fix"
#SBATCH --partition=batch
#SBATCH --output=out/%j-groot_n1_5_robocasa_mh_discfix.out
#SBATCH --error=out/%j-groot_n1_5_robocasa_mh_discfix.err

# Final 2-GPU multi-horizon training with discrete-dim fix.
# Global batch = num_gpus * batch_size = 2 * 32 = 64
# Difference vs. the original mh script: pass --discrete-action-dims 6 11 so
# _compress_actions uses last-of-group (not sum) for the binary gripper_close
# (idx 6) and control_mode (idx 11) dims. Without this fix the f2/f4 targets
# get values like 2 (sum of two 1's) which are nonsense for binary signals.

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

export WANDB_PROJECT=GR00T-robocasa
mkdir -p out

CKPT_DIR="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_multi_horizon_discfix"
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"

source $CONDA_PATH/bin/activate gr00t

python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config single_panda_gripper_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-robocasa-multi-horizon-discfix-fromPT-bs64 \
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
    --resume
