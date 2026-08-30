#!/bin/bash
# Smoke fair_moe PYRAMID layout {1x,2x,4x,8x} full-horizon, balance off.
#   K=4: {raw16, merged8(2x), merged4(4x), merged2(8x)}
#   K=3: {raw16, merged8(2x), merged4(4x)}
# 5 steps, 2 GPU. Usage: bash _smoke_fair_moe_pyramid.sh [K] [DATA_CFG]
set -u
K=${1:-4}
DATA_CFG=${2:-single_panda_gripper_multi_horizon}

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
OUT_DIR="$BASE_DIR/ckpt/_smoke_fair_moe_pyramid_K${K}_robocasa"
mkdir -p "$BASE_DIR/out"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
export WANDB_PROJECT=GR00T-robocasa
cd "$BASE_DIR"

echo "[$(date '+%T')] === SMOKE fair_moe PYRAMID K=$K, balance off, data=$DATA_CFG ==="

"$CONDA_PATH/envs/gr00t/bin/python" $BASE_DIR/scripts/gr00t_finetune_fair_moe.py \
    --dataset-path /sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
    --output-dir $OUT_DIR \
    --dataloader-num-workers 8 \
    --data-config $DATA_CFG \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE-fair-moe-pyramid-K${K} \
    --batch-size 2 --num-gpus 2 --max-steps 5 --save-steps 5 \
    --report-to wandb \
    --discrete-action-dims 6 11 \
    --use-merged-8-head --merged-8-weight 1.0 \
    --moe-pyramid \
    --use-moe-routing \
    --moe-body-mode=per_expert_h \
    --moe-num-experts=$K \
    --moe-routing-mode=single_pick \
    --moe-expert-n-layers=2 \
    --moe-router-temp=0.5 --moe-target-temp=0.3 \
    --moe-balance-weight=0.0 --moe-supervise-weight=0.1 \
    --moe-router-warmup-steps=2 \
    --no-moe-shared-t \
    --moe-normalize-loss-for-kl --moe-loss-ema-momentum=0.99 \
    --moe-balance-type=mean_sq \
    --save-total-limit 2

echo "[$(date '+%T')] === SMOKE done ==="
