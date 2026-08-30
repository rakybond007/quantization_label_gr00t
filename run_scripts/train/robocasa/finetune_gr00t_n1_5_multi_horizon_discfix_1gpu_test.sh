#!/bin/bash
# 1-GPU short test of multi-horizon + discrete-dim-fix training.
# Verifies: model loads, dataloader works, 1-step training succeeds, checkpoint
# saves, and that _compress_actions does not corrupt the binary discrete dims
# (gripper_close at idx 6, control_mode at idx 11) when summing for f2/f4.

set -e

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

# Avoid interactive wandb prompts during smoke test
export WANDB_MODE=offline
export WANDB_PROJECT=GR00T-robocasa

CKPT_DIR="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_multi_horizon_discfix_1gpu_test"
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"

# Clean previous test checkpoint to avoid resume confusion
rm -rf "$CKPT_DIR"

source $CONDA_PATH/bin/activate gr00t

python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 4 \
    --data-config single_panda_gripper_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-robocasa-mh-1gpu-test \
    --batch-size 4 \
    --num-gpus 1 \
    --max-steps 20 \
    --save-steps 10 \
    --report-to tensorboard \
    --use-multi-horizon-loss \
    --multi-horizon-factors 2 4 \
    --multi-horizon-loss-weights 1.0 1.0 \
    --multi-horizon-main-weight 1.0 \
    --aux-grad-scale-to-body 0.1 \
    --aux-loss-warmup-steps 5 \
    --discrete-action-dims 6 11
