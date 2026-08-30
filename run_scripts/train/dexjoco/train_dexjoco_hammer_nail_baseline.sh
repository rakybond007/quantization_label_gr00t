#!/bin/bash
#SBATCH --job-name=groot_n1_5_dexjoco_hammer_nail_baseline_gr00tn15_60k_bs64_2gpu
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="DexJoCo hammer_nail baseline GR00T-N1.5 (v2.0-converted LeRobot, single-arm dexterous hand, absolute actions)."
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-train_dexjoco_hammer_nail_baseline.out
#SBATCH --error=out/%j-train_dexjoco_hammer_nail_baseline.err
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"; CONDA_PATH="$HOME/miniconda3"
export WANDB_PROJECT=GR00T-dexjoco; mkdir -p out
DATA=/sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/hammer_nail
CKPT_DIR="$BASE_DIR/ckpt/dexjoco/groot/groot_n1_5_bs64_hammer_nail_baseline"
source $CONDA_PATH/bin/activate gr00t
python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path "$DATA" --output-dir $CKPT_DIR --dataloader-num-workers 16 \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-dexjoco-hammer-nail-baseline-bs64 \
    --batch-size 32 --num-gpus 2 --max-steps 60000 --save-steps 10000 \
    --report-to wandb --save-total-limit 2
