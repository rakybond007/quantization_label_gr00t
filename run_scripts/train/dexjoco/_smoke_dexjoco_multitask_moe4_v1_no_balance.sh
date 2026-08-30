#!/bin/bash
# Smoke DexJoCo MoE4 v1 no_balance multi-task via LeRobotMixtureDataset.
# 5 steps, 1 GPU.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
V20=/sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20
OUT="$BASE_DIR/ckpt/_smoke_dexjoco_multitask_moe4_v1_no_balance"
mkdir -p "$BASE_DIR/out"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
export WANDB_PROJECT=GR00T-dexjoco
cd "$BASE_DIR"

echo "[$(date '+%T')] === SMOKE DexJoCo multitask MoE4 v1 no_balance ==="
"$CONDA_PATH/envs/gr00t/bin/python" $BASE_DIR/scripts/gr00t_finetune_fair_moe.py \
    --dataset-path $V20/hammer_nail $V20/pinch_tongs \
    --output-dir "$OUT" \
    --dataloader-num-workers 4 \
    --data-config dexjoco_single_arm_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE-dexjoco-multitask-moe4-v1-no-balance \
    --batch-size 2 --num-gpus 1 --max-steps 5 --save-steps 5 \
    --report-to wandb \
    --action-mode abs \
    --use-merged-8-head --merged-8-weight 1.0 \
    --use-merged-4-head --merged-4-weight 1.0 \
    --use-native-8-head --native-8-weight 1.0 \
    --use-moe-routing \
    --moe-body-mode=per_expert_h \
    --moe-num-experts=4 \
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
