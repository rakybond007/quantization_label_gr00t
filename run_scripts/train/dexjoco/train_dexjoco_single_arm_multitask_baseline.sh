#!/bin/bash
#SBATCH --job-name=groot_n1_5_dexjoco_single_arm_multitask_6tasks_baseline_gr00tn15_60k_bs64_2gpu
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="DexJoCo single-arm multi-task (6 tasks via LeRobotMixtureDataset) baseline GR00T-N1.5. Absolute actions."
#SBATCH --partition=sjw_alinlab
#SBATCH --exclude=worker-node100
#SBATCH --output=out/%j-train_dexjoco_single_arm_multitask_baseline.out
#SBATCH --error=out/%j-train_dexjoco_single_arm_multitask_baseline.err

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"; CONDA_PATH="$HOME/miniconda3"
export WANDB_PROJECT=GR00T-dexjoco; mkdir -p out
CKPT_DIR="$BASE_DIR/ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_baseline"
source $CONDA_PATH/bin/activate gr00t
python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/hammer_nail \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/click_mouse \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/pick_bucket \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/pinch_tongs \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/fold_glasses \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/water_plant \
    --output-dir $CKPT_DIR --dataloader-num-workers 16 \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-dexjoco-single-arm-multitask-baseline-bs64 \
    --batch-size 32 --num-gpus 2 --max-steps 60000 --save-steps 10000 \
    --report-to wandb
