#!/bin/bash
# Validate click_mouse (ego_right cam) works with the patched modality.json in a multitask mix.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
V20=/sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20
OUT="$BASE_DIR/ckpt/_smoke_dexjoco_multitask_clickmouse"
mkdir -p "$BASE_DIR/out"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
export WANDB_PROJECT=GR00T-dexjoco
cd "$BASE_DIR"
echo "[$(date '+%T')] === SMOKE click_mouse mix ==="
"$CONDA_PATH/envs/gr00t/bin/python" $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $V20/hammer_nail $V20/click_mouse \
    --output-dir "$OUT" --dataloader-num-workers 4 \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE-dexjoco-clickmouse-mix \
    --batch-size 4 --num-gpus 1 --max-steps 10 --save-steps 10 --report-to wandb
echo "[$(date '+%T')] === SMOKE done ==="
