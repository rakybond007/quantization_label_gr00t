#!/bin/bash
#SBATCH --job-name=groot_n1_5_dexjoco_dual_arm_multitask_5tasks_baseline_60k_bs64_2gpu
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="DexJoCo dual-arm multi-task (5 bimanual tasks) baseline GR00T-N1.5."
#SBATCH --partition=sjw_alinlab
#SBATCH --exclude=worker-node100
#SBATCH --output=out/%j-train_dexjoco_dual_arm_multitask_baseline.out
#SBATCH --error=out/%j-train_dexjoco_dual_arm_multitask_baseline.err

set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"; CONDA_PATH="$HOME/miniconda3"
DSET_ROOT=/sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/bimanual
export WANDB_PROJECT=GR00T-dexjoco; mkdir -p out
CKPT_DIR="$BASE_DIR/ckpt/dexjoco/groot/groot_n1_5_bs64_dual_arm_multitask_baseline"

unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
cd "$BASE_DIR"
"$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/gr00t_finetune_dual_arm.py" \
    --dataset-path $DSET_ROOT/bimanual_assembly \
        $DSET_ROOT/bimanual_microwave_cook \
        $DSET_ROOT/bimanual_unlock_ipad \
        $DSET_ROOT/bimanual_hanoi \
        $DSET_ROOT/bimanual_photograph \
    --output-dir $CKPT_DIR --dataloader-num-workers 16 \
    --data-config dexjoco_dual_arm_multi_horizon --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-dexjoco-dual-arm-multitask-baseline-bs64 \
    --batch-size 32 --num-gpus 2 --max-steps 60000 --save-steps 10000 \
    --override-action-dim 64 \
    --resume \
    --report-to wandb
