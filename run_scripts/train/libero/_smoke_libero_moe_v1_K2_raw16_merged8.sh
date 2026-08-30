#!/bin/bash
# Smoke test for libero K=2 {raw16, merged8} MoE training (both balance-on
# and balance-off variants). 5 steps, no wandb. Interactive GPU node only.
# Usage: bash _smoke_libero_moe_v1_K2_raw16_merged8.sh <BALANCE_WEIGHT>
# e.g.  bash _smoke_libero_moe_v1_K2_raw16_merged8.sh 0.05   # balance on
#       bash _smoke_libero_moe_v1_K2_raw16_merged8.sh 0.0    # balance off
BAL_W="${1:-0.0}"
LABEL=$( [ "$BAL_W" = "0.0" ] && echo "no_balance" || echo "balance$BAL_W" )

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
mkdir -p out
CKPT_DIR="$BASE_DIR/ckpt/libero/_smoke_moe_v1_K2_${LABEL}"
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"

# Scrub venv pollution then prepend gr00t env to PATH so any subprocess
# (notably torchrun launched from gr00t_finetune_fair_moe.py) resolves to
# the gr00t env's python instead of openpi/.venv/bin/python.
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
GR00T_PY="$CONDA_PATH/envs/gr00t/bin/python"
echo "[smoke] balance_weight=$BAL_W ckpt=$CKPT_DIR PATH-head=${PATH%%:*}"
PYTHONUNBUFFERED=1 "$GR00T_PY" $BASE_DIR/scripts/gr00t_finetune_fair_moe.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 8 \
    --data-config libero_multi_horizon \
    --embodiment-tag libero \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE_libero_moe_v1_K2_${LABEL} \
    --batch-size 16 --num-gpus 2 --max-steps 5 --save-steps 10 \
    --report-to tensorboard \
    --discrete-action-dims 6 \
    --use-merged-8-head --merged-8-weight 1.0 \
    --use-moe-routing \
    --moe-body-mode=per_expert_h \
    --moe-num-experts=2 \
    --moe-routing-mode=single_pick \
    --moe-expert-n-layers=2 \
    --moe-router-temp=0.5 --moe-target-temp=0.3 \
    --moe-balance-weight=$BAL_W --moe-supervise-weight=0.1 \
    --moe-router-warmup-steps=2 \
    --no-moe-shared-t \
    --moe-normalize-loss-for-kl --moe-loss-ema-momentum=0.99 \
    --moe-balance-type=mean_sq \
    --moe-length-cost-weight=0.0 \
    --save-total-limit=2
echo "[smoke] DONE balance_weight=$BAL_W"
