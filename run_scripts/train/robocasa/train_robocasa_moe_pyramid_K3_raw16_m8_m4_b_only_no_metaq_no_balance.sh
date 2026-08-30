#!/bin/bash
#SBATCH --job-name=groot_n1_5_robocasa_moe_pyramid_K3_raw16_m8_m4_b_only_no_metaq_no_balance_60k_bs64
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="Robocasa MoE pyramid K=3 {raw16, merged8(2x), merged4(4x)} full-horizon factor 1/2/4/8 + [B] EMA-norm KL + no metaq + BALANCE OFF. fair_moe fork."
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-train_robocasa_moe_pyramid_K3_raw16_m8_m4_no_balance.out
#SBATCH --error=out/%j-train_robocasa_moe_pyramid_K3_raw16_m8_m4_no_balance.err

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
export WANDB_PROJECT=GR00T-robocasa
mkdir -p out
CKPT_DIR="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_moe_pyramid_K3_raw16_m8_m4_b_only_no_metaq_no_balance"

source $CONDA_PATH/bin/activate gr00t
python $BASE_DIR/scripts/gr00t_finetune_fair_moe.py \
    --dataset-path /sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config single_panda_gripper_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-robocasa-moe-pyramid-K3_raw16_m8_m4-b-only-no-metaq-no-balance-bs64 \
    --batch-size 32 --num-gpus 2 --max-steps 60000 --save-steps 10000 \
    --report-to wandb \
    --discrete-action-dims 6 11 \
    --use-merged-8-head --merged-8-weight 1.0 \
    --moe-pyramid \
    --use-moe-routing \
    --moe-body-mode=per_expert_h \
    --moe-num-experts=3 \
    --moe-routing-mode=single_pick \
    --moe-expert-n-layers=2 \
    --moe-router-temp=0.5 --moe-target-temp=0.3 \
    --moe-balance-weight=0.0 --moe-supervise-weight=0.1 \
    --moe-router-warmup-steps=5000 \
    --no-moe-shared-t \
    --moe-normalize-loss-for-kl --moe-loss-ema-momentum=0.99 \
    --moe-balance-type=mean_sq \
    --moe-length-cost-weight=0.0 \
    --save-total-limit=2
