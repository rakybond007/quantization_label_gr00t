#!/bin/bash
# Smoke libero fair_moe v3a (K=4) — 5 step, 2 GPU. Verifies v3 head works with
# libero_multi_horizon data config (forward/backward).
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
OUT_DIR="$BASE_DIR/ckpt/_smoke_libero_fair_moe_v3a"
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"
mkdir -p "$BASE_DIR/out"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
export WANDB_PROJECT=GR00T-libero WANDB_MODE=disabled
cd "$BASE_DIR"
echo "[$(date '+%T')] === SMOKE libero fair_moe v3a K4 ==="
"$CONDA_PATH/envs/gr00t/bin/python" $BASE_DIR/scripts/gr00t_finetune_fair_moe_v3.py \
    --dataset-path $DATA_DIR \
    --output-dir $OUT_DIR \
    --dataloader-num-workers 8 \
    --data-config libero_multi_horizon \
    --embodiment-tag libero \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE-libero-v3a-K4 \
    --batch-size 2 --num-gpus 2 --max-steps 5 --save-steps 5 \
    --report-to wandb \
    --discrete-action-dims 6 \
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
    --save-total-limit 1
echo "[$(date '+%T')] === SMOKE done ==="
