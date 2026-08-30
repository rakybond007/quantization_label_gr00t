#!/bin/bash
# Smoke MoE4 per_expert single_pick + meta-query (N=8) + [A][B][C] + V2 decoder
# set (u16 / c8 / u24 / c12).  5 steps, 2 GPU, robocasa.
# Usage: bash _smoke_metaq_v2.sh [robocasa] [with_d|no_d]
set -u
TARGET=${1:?"need: robocasa"}
DMODE=${2:-no_d}

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
    *) echo "[ERROR] only robocasa supported for now"; exit 1 ;;
esac

case "$DMODE" in
    with_d) DARG="--moe-length-cost-weight=0.10" ;;
    no_d)   DARG="--moe-length-cost-weight=0.0" ;;
    *) echo "[ERROR] need with_d|no_d"; exit 1 ;;
esac

OUT_DIR="$BASE_DIR/ckpt/_smoke_metaq_v2_${TARGET}_${DMODE}"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
PYBIN="$CONDA_PATH/envs/gr00t/bin/python"
cd "$BASE_DIR"

echo "[$(date '+%T')] === SMOKE METAQ V2: $TARGET $DMODE (5 steps) ==="

"$PYBIN" $BASE_DIR/scripts/gr00t_finetune_metaq_v2.py \
    --dataset-path $DATA_DIR \
    --output-dir $OUT_DIR \
    --dataloader-num-workers 8 \
    --data-config $DATA_CFG \
    --embodiment-tag $EMB_TAG \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE-metaq-v2-$TARGET-$DMODE \
    --batch-size 2 --num-gpus 2 --max-steps 5 --save-steps 5 \
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
    --n-meta-q=8 \
    --moe-shared-t \
    --moe-normalize-loss-for-kl --moe-loss-ema-momentum=0.99 \
    --moe-balance-type=switch \
    $DARG

echo "[$(date '+%T')] === SMOKE done: $TARGET $DMODE ==="
