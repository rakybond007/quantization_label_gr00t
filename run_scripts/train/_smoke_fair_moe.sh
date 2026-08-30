#!/bin/bash
# Smoke fair_moe fork: MoE4 per_expert + [B] EMA-normalize KL only.
# 5 steps, 2 GPU.
set -u
DATA_CFG=${1:-single_panda_gripper_multi_horizon}   # or _state_aug variant
BAL=${2:-0.05}                                       # moe-balance-weight (0 = balance off)

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
OUT_DIR="$BASE_DIR/ckpt/_smoke_fair_moe_robocasa"
mkdir -p "$BASE_DIR/out"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
export WANDB_PROJECT=GR00T-robocasa
cd "$BASE_DIR"

echo "[$(date '+%T')] === SMOKE fair_moe ([B] only, data=$DATA_CFG) ==="

"$CONDA_PATH/envs/gr00t/bin/python" $BASE_DIR/scripts/gr00t_finetune_fair_moe.py \
    --dataset-path /sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
    --output-dir $OUT_DIR \
    --dataloader-num-workers 8 \
    --data-config $DATA_CFG \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE-fair-moe-B-only \
    --batch-size 2 --num-gpus 2 --max-steps 5 --save-steps 5 \
    --report-to wandb \
    --discrete-action-dims 6 11 \
    --use-merged-8-head --merged-8-weight 1.0 \
    --use-merged-4-head --merged-4-weight 1.0 \
    --use-native-8-head --native-8-weight 1.0 \
    --use-moe-routing \
    --moe-body-mode=per_expert_h \
    --moe-num-experts=4 \
    --moe-routing-mode=single_pick \
    --moe-expert-n-layers=2 \
    --moe-router-temp=0.5 --moe-target-temp=0.3 \
    --moe-balance-weight=$BAL --moe-supervise-weight=0.1 \
    --moe-router-warmup-steps=2 \
    --moe-normalize-loss-for-kl --moe-loss-ema-momentum=0.99 \
    --save-total-limit 2

echo "[$(date '+%T')] === SMOKE done ==="
