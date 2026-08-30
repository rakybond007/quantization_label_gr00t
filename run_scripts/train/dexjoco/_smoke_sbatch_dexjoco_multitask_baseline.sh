#!/bin/bash
#SBATCH --job-name=_SMOKE_dexjoco_single_arm_multitask_6tasks_baseline_sbatch_2gpu
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="SMOKE sbatch (identical headers to full train) for dexjoco multitask baseline — 5 steps."
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-_smoke_sbatch_dexjoco_multitask_baseline.out
#SBATCH --error=out/%j-_smoke_sbatch_dexjoco_multitask_baseline.err

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"; CONDA_PATH="$HOME/miniconda3"
export WANDB_PROJECT=GR00T-dexjoco; mkdir -p out
CKPT_DIR="$BASE_DIR/ckpt/_smoke_sbatch_dexjoco_multitask_baseline"
source $CONDA_PATH/bin/activate gr00t
echo "[$(date '+%T')] nvidia-smi visible GPUs:"
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1 | head -4
echo "[$(date '+%T')] torch.cuda.device_count: $(python -c 'import torch; print(torch.cuda.device_count())')"
python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/hammer_nail \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/click_mouse \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/pick_bucket \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/pinch_tongs \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/fold_glasses \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/water_plant \
    --output-dir $CKPT_DIR --dataloader-num-workers 4 \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE-SBATCH-dexjoco-multitask-baseline \
    --batch-size 4 --num-gpus 2 --max-steps 5 --save-steps 5 \
    --report-to wandb
