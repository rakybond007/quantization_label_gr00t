#!/bin/bash
# Smoke DexJoCo multi-task baseline at --num-gpus 2 (matches full-train script).
# Run interactively on a 2-GPU node (tmux 0:0). 5 steps, batch 4 per GPU.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
V20=/sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20
OUT="$BASE_DIR/ckpt/_smoke_dexjoco_multitask_baseline_2gpu"
mkdir -p "$BASE_DIR/out"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
export WANDB_PROJECT=GR00T-dexjoco
cd "$BASE_DIR"
echo "[$(date '+%T')] === SMOKE multitask baseline 2-GPU on $(hostname) ==="
echo "torch.cuda.device_count: $($CONDA_PATH/envs/gr00t/bin/python -c 'import torch; print(torch.cuda.device_count())')"
"$CONDA_PATH/envs/gr00t/bin/python" $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $V20/hammer_nail $V20/click_mouse $V20/pick_bucket \
                   $V20/pinch_tongs $V20/fold_glasses $V20/water_plant \
    --output-dir "$OUT" --dataloader-num-workers 4 \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE-dexjoco-multitask-baseline-2gpu \
    --batch-size 4 --num-gpus 2 --max-steps 5 --save-steps 5 \
    --report-to wandb
echo "[$(date '+%T')] === SMOKE done ==="
