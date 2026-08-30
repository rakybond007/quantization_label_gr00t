#!/bin/bash
# Smoke: dexjoco single-arm 6-task multitask + MoH (default horizons 4/8/12/16),
# 1 GPU per-gpu batch 32 (global 64), 10 steps. Verify MoH wiring on dexjoco.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT_DIR="$BASE_DIR/ckpt/_smoke_dexjoco_single_arm_multitask_moh_1gpu"
mkdir -p out

source $CONDA_PATH/bin/activate gr00t
export PYTHONPATH="$BASE_DIR:${PYTHONPATH:-}"
export WANDB_MODE=disabled  # smoke: no wandb upload

$CONDA_PATH/envs/gr00t/bin/python $BASE_DIR/scripts/gr00t_finetune_moh.py \
    --dataset-path /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/hammer_nail \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/click_mouse \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/pick_bucket \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/pinch_tongs \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/fold_glasses \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/water_plant \
    --output-dir $CKPT_DIR --dataloader-num-workers 4 \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE-dexjoco-single-arm-multitask-moh-h4-8-12-16 \
    --batch-size 32 --num-gpus 1 --max-steps 10 --save-steps 1000 \
    --report-to wandb \
    --horizons 4 8 12 16 \
    --aux-weight 1.0 \
    --balance-weight 0.001 \
    --use-gate-noise \
    --no-mean-fusion \
    --no-use-dynamic-replanning \
    --scale-ratio 1.0 \
    --min-replan-steps 5 \
    --min-active-horizons 1 \
    --load-action-head
echo "SMOKE_DEXJOCO_MOH_DONE rc=$?"
