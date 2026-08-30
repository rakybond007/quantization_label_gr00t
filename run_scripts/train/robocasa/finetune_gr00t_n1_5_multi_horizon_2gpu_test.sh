#!/bin/bash
# 2-GPU short test of multi-horizon training (verifies torchrun launch + DDP).
# Run interactively on a 2-GPU node:
#   bash run_scripts/train/robocasa/finetune_gr00t_n1_5_multi_horizon_2gpu_test.sh
# Verifies: 2-GPU launch via torchrun, DDP works, multi-horizon loss across ranks.

set -e

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

export WANDB_MODE=offline
export WANDB_PROJECT=GR00T-robocasa

CKPT_DIR="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_multi_horizon_2gpu_test"
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"

# Clean previous test checkpoint to avoid resume confusion
rm -rf "$CKPT_DIR"

source $CONDA_PATH/bin/activate gr00t

# Per-device batch 4 * 2 GPUs = global batch 8 (small for quick test)
python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 4 \
    --data-config single_panda_gripper_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-robocasa-mh-2gpu-test \
    --batch-size 4 \
    --num-gpus 2 \
    --max-steps 20 \
    --save-steps 10 \
    --report-to tensorboard \
    --use-multi-horizon-loss \
    --multi-horizon-factors 2 4 \
    --multi-horizon-loss-weights 1.0 1.0 \
    --multi-horizon-main-weight 1.0 \
    --aux-grad-scale-to-body 0.1 \
    --aux-loss-warmup-steps 5
