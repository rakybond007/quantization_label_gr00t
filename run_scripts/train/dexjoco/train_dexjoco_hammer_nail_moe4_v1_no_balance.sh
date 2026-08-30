#!/bin/bash
#SBATCH --job-name=groot_n1_5_dexjoco_hammer_nail_moe4_v1_no_balance_60k_bs64
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="DexJoCo hammer_nail MoE4 v1 (no metaq, BALANCE OFF) GR00T-N1.5. Absolute actions -> action-mode abs (compression = block last-value)."
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-train_dexjoco_hammer_nail_moe4_v1_no_balance.out
#SBATCH --error=out/%j-train_dexjoco_hammer_nail_moe4_v1_no_balance.err
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"; CONDA_PATH="$HOME/miniconda3"
export WANDB_PROJECT=GR00T-dexjoco; mkdir -p out
DATA=/sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/hammer_nail
CKPT_DIR="$BASE_DIR/ckpt/dexjoco/groot/groot_n1_5_bs64_hammer_nail_moe4_v1_no_balance"
source $CONDA_PATH/bin/activate gr00t
python $BASE_DIR/scripts/gr00t_finetune_fair_moe.py \
    --dataset-path "$DATA" --output-dir $CKPT_DIR --dataloader-num-workers 16 \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-dexjoco-hammer-nail-moe4-v1-no-balance-bs64 \
    --batch-size 32 --num-gpus 2 --max-steps 60000 --save-steps 10000 \
    --report-to wandb \
    --action-mode abs \
    --use-merged-8-head --merged-8-weight 1.0 \
    --use-merged-4-head --merged-4-weight 1.0 \
    --use-native-8-head --native-8-weight 1.0 \
    --use-moe-routing --moe-body-mode=per_expert_h --moe-num-experts=4 \
    --moe-routing-mode=single_pick --moe-expert-n-layers=2 \
    --moe-router-temp=0.5 --moe-target-temp=0.3 \
    --moe-balance-weight=0.0 --moe-supervise-weight=0.1 --moe-router-warmup-steps=5000 \
    --no-moe-shared-t --moe-normalize-loss-for-kl --moe-loss-ema-momentum=0.99 \
    --moe-balance-type=mean_sq --moe-length-cost-weight=0.0 --save-total-limit=2
