#!/bin/bash
# Smoke MoE4 (main+m8+m4+n8) with length-cost (option 1) + critic head (option 2).
# 5 steps, 2 GPU, no wandb. Usage: bash _smoke_moe4_lc_critic.sh [robocasa|libero]
set -u
TARGET=${1:?"need: robocasa | libero"}

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
mkdir -p "$BASE_DIR/out"

case "$TARGET" in
    robocasa)
        DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
        DATA_CFG="single_panda_gripper_multi_horizon"
        EMB_TAG="new_embodiment"
        DDIMS="--discrete-action-dims 6 11"
        export WANDB_PROJECT=GR00T-robocasa
        ;;
    libero)
        DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"
        DATA_CFG="libero_multi_horizon"
        EMB_TAG="libero"
        DDIMS="--discrete-action-dims 6"
        export WANDB_PROJECT=GR00T-libero
        ;;
    *) echo "[ERROR] unknown target $TARGET"; exit 1 ;;
esac

OUT_DIR="$BASE_DIR/ckpt/_smoke_moe4_zscore_critic_${TARGET}"
# Strip any active venv so nested torchrun finds the conda gr00t binaries.
unset VIRTUAL_ENV PYTHONHOME
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
PYBIN="$CONDA_PATH/envs/gr00t/bin/python"

echo "[$(date '+%T')] === SMOKE: $TARGET MoE4 length-cost + critic (5 steps) ==="

"$PYBIN" $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $OUT_DIR \
    --dataloader-num-workers 8 \
    --data-config $DATA_CFG \
    --embodiment-tag $EMB_TAG \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE-$TARGET-moe4-lc-critic \
    --batch-size 2 --num-gpus 2 --max-steps 5 --save-steps 100000 \
    --report-to wandb \
    $DDIMS \
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
    --moe-target-normalize \
    --moe-target-normalize-ema=0.01 \
    --moe-length-cost-weight=0.3 \
    --moe-critic-weight=0.3 \
    --moe-critic-hidden=256

echo "[$(date '+%T')] === SMOKE done: $TARGET ==="
