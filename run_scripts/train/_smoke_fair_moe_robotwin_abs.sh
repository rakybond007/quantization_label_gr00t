#!/bin/bash
# Smoke fair_moe v1 + RoboTwin Agilex (abs action). 5 steps, 1 GPU, batch 2.
# Uses adjust_bottle task only (smallest path); switch to full robotwin mix
# once the dataset finishes downloading.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
OUT_DIR="$BASE_DIR/ckpt/_smoke_fair_moe_robotwin_abs"
DATASET_PATH=${1:-"$HOME/multigpu_workspace/datasets/RoboTwin/Clean/adjust_bottle"}
mkdir -p "$BASE_DIR/out"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
export WANDB_PROJECT=GR00T-robotwin
cd "$BASE_DIR"

echo "[$(date '+%T')] === SMOKE fair_moe + RoboTwin (abs action, K=4 v1 layout, [B] only) ==="
echo "[$(date '+%T')] DATASET_PATH=$DATASET_PATH"

"$CONDA_PATH/envs/gr00t/bin/python" $BASE_DIR/scripts/gr00t_finetune_fair_moe.py \
    --dataset-path "$DATASET_PATH" \
    --output-dir $OUT_DIR \
    --dataloader-num-workers 4 \
    --data-config robotwin_agilex \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE-fair-moe-robotwin-abs-B-only \
    --batch-size 2 --num-gpus 1 --max-steps 5 --save-steps 5 \
    --report-to wandb \
    --video-backend torchvision_av \
    --action-mode abs \
    --discrete-action-dims 6 13 \
    --use-merged-8-head --merged-8-weight 1.0 \
    --use-merged-4-head --merged-4-weight 1.0 \
    --use-native-8-head --native-8-weight 1.0 \
    --use-moe-routing \
    --moe-body-mode=per_expert_h \
    --moe-num-experts=4 \
    --moe-routing-mode=single_pick \
    --moe-expert-n-layers=2 \
    --moe-router-temp=0.5 --moe-target-temp=0.3 \
    --moe-balance-weight=0.05 --moe-supervise-weight=0.1 \
    --moe-router-warmup-steps=2 \
    --moe-normalize-loss-for-kl --moe-loss-ema-momentum=0.99 \
    --save-total-limit 2

echo "[$(date '+%T')] === SMOKE done ==="
