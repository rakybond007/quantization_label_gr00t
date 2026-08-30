#!/bin/bash
#SBATCH --job-name=groot_n1_5_dexjoco_single_arm_multitask_6tasks_moe4_v1_balance_30k_bs64_2gpu
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="DexJoCo single-arm multi-task (6 tasks via LeRobotMixtureDataset) MoE4 v1 with balance term (0.05). Otherwise identical to no_balance variant. Absolute actions -> action-mode abs."
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-train_dexjoco_single_arm_multitask_moe_balance.out
#SBATCH --error=out/%j-train_dexjoco_single_arm_multitask_moe_balance.err

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"; CONDA_PATH="$HOME/miniconda3"
export WANDB_PROJECT=GR00T-dexjoco; mkdir -p out
CKPT_DIR="$BASE_DIR/ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_moe4_v1_balance_30k"
source $CONDA_PATH/bin/activate gr00t
python $BASE_DIR/scripts/gr00t_finetune_fair_moe.py \
    --dataset-path /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/hammer_nail \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/click_mouse \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/pick_bucket \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/pinch_tongs \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/fold_glasses \
        /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/water_plant \
    --output-dir $CKPT_DIR --dataloader-num-workers 16 \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-dexjoco-single-arm-multitask-moe4-v1-balance-30k-bs64 \
    --batch-size 32 --num-gpus 2 --max-steps 30000 --save-steps 10000 \
    --report-to wandb \
    --action-mode abs \
    --use-merged-8-head --merged-8-weight 1.0 \
    --use-merged-4-head --merged-4-weight 1.0 \
    --use-native-8-head --native-8-weight 1.0 \
    --use-moe-routing --moe-body-mode=per_expert_h --moe-num-experts=4 \
    --moe-routing-mode=single_pick --moe-expert-n-layers=2 \
    --moe-router-temp=0.5 --moe-target-temp=0.3 \
    --moe-balance-weight=0.05 --moe-supervise-weight=0.1 --moe-router-warmup-steps=5000 \
    --no-moe-shared-t --moe-normalize-loss-for-kl --moe-loss-ema-momentum=0.99 \
    --moe-balance-type=mean_sq --moe-length-cost-weight=0.0 --save-total-limit=3
