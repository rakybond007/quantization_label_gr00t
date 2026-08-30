#!/bin/bash
#SBATCH --job-name=groot_n1_5_dexjoco_single_arm_multitask_moh_h4-8-12-16_2gpu_perGPU32_bs64_60k
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab_premium
#SBATCH --output=out/%j-groot_n1_5_dexjoco_single_arm_multitask_moh.out
#SBATCH --error=out/%j-groot_n1_5_dexjoco_single_arm_multitask_moh.err
#SBATCH --comment="GR00T N1.5 finetune on DexJoCo single-arm 6-task multitask with Mixture-of-Horizons (MoH) action head, horizons=[4,8,12,16], 60k steps, per-GPU 32 on 2 H100."

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

export WANDB_PROJECT=GR00T-dexjoco
mkdir -p out

CKPT_DIR="$BASE_DIR/ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_moh_h4-8-12-16"

source $CONDA_PATH/bin/activate gr00t
export PYTHONPATH="$BASE_DIR:${PYTHONPATH:-}"

$CONDA_PATH/envs/gr00t/bin/python $BASE_DIR/scripts/gr00t_finetune_moh.py \
    --dataset-path /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/hammer_nail \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/click_mouse \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/pick_bucket \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/pinch_tongs \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/fold_glasses \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/water_plant \
    --output-dir $CKPT_DIR --dataloader-num-workers 32 \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-dexjoco-single-arm-multitask-moh-h4-8-12-16-fromPT-bs64 \
    --batch-size 32 --num-gpus 2 --max-steps 60000 --save-steps 10000 \
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
    --load-action-head \
    --save-total-limit 6
