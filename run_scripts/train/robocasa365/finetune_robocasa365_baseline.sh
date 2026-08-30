#!/bin/bash
#SBATCH --job-name=groot_n1_5_robocasa365_base
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="GR00T N1.5 finetune on RoboCasa365 (baseline, no multi-horizon)"
#SBATCH --partition=batch
#SBATCH --output=out/%j-groot_n1_5_robocasa365_base.out
#SBATCH --error=out/%j-groot_n1_5_robocasa365_base.err

# RoboCasa365 baseline finetune (mirror of robocasa/finetune_gr00t_n1_5_baseline.sh).
# RoboCasa365 (ICLR 2026, Nasiriany et al.) = 65 atomic + 300 composite tasks,
# 2,500 kitchen scenes, ~2,000+ hours demos. Built on robocasa v1.0 (Feb 2026).
#
# Global batch = 2 * 32 = 64
# NOTE: dataset path below currently points to the 24-atomic-task subset
#       (kimtaey/robocasa_mg_gr00t_300, 7200 episodes). When the full 365-task
#       LeRobot-converted dataset is staged locally, swap DATA_DIR to it.
#       Candidate source: nvidia/robocasa365-datasets on HuggingFace
#         hf download nvidia/robocasa365-datasets --repo-type dataset \
#             --local-dir /sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/nvidia/robocasa365
#       (Note: nvidia/robocasa365-datasets is the upstream raw release; may need
#        LeRobot-format conversion before pointing gr00t_finetune.py at it.)

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

export WANDB_PROJECT=GR00T-robocasa365
mkdir -p out

CKPT_DIR="$BASE_DIR/ckpt/robocasa365/groot/groot_n1_5_bs64_baseline"

# TODO: swap to full 365-task LeRobot dataset when available, e.g.
#   DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/nvidia/robocasa365"
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"

source $CONDA_PATH/bin/activate gr00t

python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config single_panda_gripper \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-robocasa365-baseline-fromPT-bs64 \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 60000 \
    --save-steps 10000 \
    --report-to wandb
