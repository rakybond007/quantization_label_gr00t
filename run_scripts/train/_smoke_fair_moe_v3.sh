#!/bin/bash
# Smoke fair_moe_v3 (K=4 v3a OR K=6 v3b). 5 steps, 1 GPU, batch 2.
# Usage: bash _smoke_fair_moe_v3.sh <a|b>
set -u
VARIANT=${1:-a}   # 'a' = K=4 (v3a, {u32,c16,u16,c8}); 'b' = K=6 (v3b, +{u8,c4})

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
OUT_DIR="$BASE_DIR/ckpt/_smoke_fair_moe_v3${VARIANT}_robocasa"
mkdir -p "$BASE_DIR/out"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
export WANDB_PROJECT=GR00T-robocasa
cd "$BASE_DIR"

case "$VARIANT" in
    a) K=4; EXTRA_FLAGS="" ;;
    b) K=6; EXTRA_FLAGS="--use-u8-extra-head --use-c4-extra-head" ;;
    *) echo "[ERR] VARIANT must be 'a' or 'b'"; exit 1 ;;
esac

echo "[$(date '+%T')] === SMOKE fair_moe_v3${VARIANT} (K=$K, [B] only) ==="
"$CONDA_PATH/envs/gr00t/bin/python" $BASE_DIR/scripts/gr00t_finetune_fair_moe_v3.py \
    --dataset-path /sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
    --output-dir $OUT_DIR \
    --dataloader-num-workers 4 \
    --data-config single_panda_gripper_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE-fair-moe-v3${VARIANT}-B-only \
    --batch-size 2 --num-gpus 1 --max-steps 5 --save-steps 5 \
    --report-to wandb \
    --discrete-action-dims 6 11 \
    --use-merged-8-head --merged-8-weight 1.0 \
    --use-merged-4-head --merged-4-weight 1.0 \
    --use-native-8-head --native-8-weight 1.0 \
    $EXTRA_FLAGS \
    --use-moe-routing \
    --moe-body-mode=per_expert_h \
    --moe-num-experts=$K \
    --moe-routing-mode=single_pick \
    --moe-expert-n-layers=2 \
    --moe-router-temp=0.5 --moe-target-temp=0.3 \
    --moe-balance-weight=0.05 --moe-supervise-weight=0.1 \
    --moe-router-warmup-steps=2 \
    --moe-normalize-loss-for-kl --moe-loss-ema-momentum=0.99 \
    --save-total-limit 2

echo "[$(date '+%T')] === SMOKE v3${VARIANT} done ==="
