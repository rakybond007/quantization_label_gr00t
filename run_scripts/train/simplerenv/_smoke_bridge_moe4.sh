#!/bin/bash
# Smoke for SimplerEnv Bridge MoE training. 100 step, same global batch (64).
set -u

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
mkdir -p out

CKPT_DIR="$BASE_DIR/ckpt/simplerenv/bridge/_smoke_moe4"
DATA_DIR="$HOME/multigpu_workspace/data/simplerenv/bridge_orig_lerobot"

source $CONDA_PATH/bin/activate gr00t

PYTHONUNBUFFERED=1 python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 16 \
    --data-config simplerenv_bridge_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE_bridge_moe4_per_expert \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 100 \
    --save-steps 200 \
    --report-to tensorboard \
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
