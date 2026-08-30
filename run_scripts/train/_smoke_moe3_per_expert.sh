#!/bin/bash
# Smoke 3-expert MoE per_expert training (5 steps, 2 GPUs, no wandb, no save).
# Usage: bash _smoke_moe3_per_expert.sh [robocasa|libero]
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
        ;;
    libero)
        DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"
        DATA_CFG="libero_multi_horizon"
        EMB_TAG="libero"
        DDIMS="--discrete-action-dims 6"
        ;;
    *) echo "[ERROR] unknown target $TARGET"; exit 1 ;;
esac

OUT_DIR="$BASE_DIR/ckpt/_smoke_moe3_${TARGET}"

source $CONDA_PATH/bin/activate gr00t

# wandb project per target (matches the full sbatch scripts).
case "$TARGET" in
    robocasa) export WANDB_PROJECT=GR00T-robocasa ;;
    libero)   export WANDB_PROJECT=GR00T-libero ;;
esac

echo "[$(date '+%T')] === SMOKE: $TARGET MoE3 per_expert (5 steps) ==="

python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $OUT_DIR \
    --dataloader-num-workers 8 \
    --data-config $DATA_CFG \
    --embodiment-tag $EMB_TAG \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE-$TARGET-moe3-per-expert \
    --batch-size 2 \
    --num-gpus 2 \
    --max-steps 5 \
    --save-steps 100000 \
    --report-to wandb \
    $DDIMS \
    --use-merged-8-head \
    --merged-8-weight 1.0 \
    --use-merged-4-head \
    --merged-4-weight 1.0 \
    --use-moe-routing \
    --moe-body-mode=per_expert_h \
    --moe-num-experts=3 \
    --moe-expert-n-layers=2 \
    --moe-router-temp=0.5 \
    --moe-target-temp=0.3 \
    --moe-balance-weight=0.05 \
    --moe-supervise-weight=0.1 \
    --moe-router-warmup-steps=2

echo "[$(date '+%T')] === SMOKE done: $TARGET ==="
