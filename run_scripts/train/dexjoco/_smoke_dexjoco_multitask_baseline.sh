#!/bin/bash
# Smoke DexJoCo baseline GR00T-N1.5 multi-task via LeRobotMixtureDataset.
# Validates that passing >1 --dataset-path assembles correctly with our
# dexjoco_single_arm_multi_horizon config. 5 steps, 1 GPU.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
V20=/sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20
OUT="$BASE_DIR/ckpt/_smoke_dexjoco_multitask_baseline"
mkdir -p "$BASE_DIR/out"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
export WANDB_PROJECT=GR00T-dexjoco
cd "$BASE_DIR"

echo "[$(date '+%T')] === SMOKE DexJoCo multitask baseline ==="
"$CONDA_PATH/envs/gr00t/bin/python" $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $V20/hammer_nail $V20/pinch_tongs \
    --output-dir "$OUT" \
    --dataloader-num-workers 4 \
    --data-config dexjoco_single_arm_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE-dexjoco-multitask-baseline \
    --batch-size 2 --num-gpus 1 --max-steps 5 --save-steps 5 \
    --report-to wandb
echo "[$(date '+%T')] === SMOKE done ==="
