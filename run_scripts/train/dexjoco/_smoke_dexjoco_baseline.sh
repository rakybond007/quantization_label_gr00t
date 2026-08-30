#!/bin/bash
# Smoke DexJoCo baseline GR00T-N1.5 (hammer_nail single-arm). 5 steps, 1 GPU.
# Validates the dexjoco_single_arm_multi_horizon data-config + modality.json +
# DexJoCo LeRobot loading end-to-end.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
DATA=/sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/hammer_nail
OUT="$BASE_DIR/ckpt/_smoke_dexjoco_baseline"
mkdir -p "$BASE_DIR/out"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
export WANDB_PROJECT=GR00T-dexjoco
cd "$BASE_DIR"

echo "[$(date '+%T')] === SMOKE DexJoCo baseline (hammer_nail) ==="
"$CONDA_PATH/envs/gr00t/bin/python" $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path "$DATA" \
    --output-dir "$OUT" \
    --dataloader-num-workers 4 \
    --data-config dexjoco_single_arm_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE-dexjoco-baseline \
    --batch-size 2 --num-gpus 1 --max-steps 5 --save-steps 5 \
    --report-to wandb
echo "[$(date '+%T')] === SMOKE done ==="
