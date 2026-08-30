#!/bin/bash
# 2-GPU smoke test of multi-horizon + discrete-fix + merged_8 + ensemble-consistency loss.
# Expected: loss_main, loss_m8, loss_f2, loss_f4, loss_consist all present, no NaN.

set -e

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

export WANDB_MODE=offline
export WANDB_PROJECT=GR00T-robocasa

CKPT_DIR="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_mh_m8_econsist_2gpu_test"
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"

rm -rf "$CKPT_DIR"

source $CONDA_PATH/bin/activate gr00t

python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 4 \
    --data-config single_panda_gripper_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-mh-m8-econsist-2gpu-test \
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
    --aux-loss-warmup-steps 5 \
    --discrete-action-dims 6 11 \
    --use-merged-8-head \
    --merged-8-weight 1.0 \
    --use-ensemble-consistency-loss \
    --ensemble-consistency-weight 0.1 \
    --ensemble-consistency-warmup-steps 5
