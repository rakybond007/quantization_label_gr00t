#!/bin/bash
#SBATCH --job-name=groot_n1_5_libero_fair_moe_v3a_u32_c16_u16_c8_K4_B_only_no_metaq_60k_bs64
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="libero fair_moe v3a (K=4): decoders {u32, c16-from-32, u16, c8-from-16}. [B] EMA-norm KL only. No metaq."
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-train_libero_fair_moe_v3a_K4.out
#SBATCH --error=out/%j-train_libero_fair_moe_v3a_K4.err

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
export WANDB_PROJECT=GR00T-libero
mkdir -p out
CKPT_DIR="$BASE_DIR/ckpt/libero/groot/groot_n1_5_bs64_fair_moe_v3a_b_only"
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"

source $CONDA_PATH/bin/activate gr00t
python $BASE_DIR/scripts/gr00t_finetune_fair_moe_v3.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config libero_multi_horizon \
    --embodiment-tag libero \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-libero-fair-moe-v3a-K4-b-only-bs64 \
    --batch-size 32 --num-gpus 2 --max-steps 60000 --save-steps 10000 \
    --report-to wandb \
    --discrete-action-dims 6 \
    --use-merged-8-head --merged-8-weight 1.0 \
    --use-merged-4-head --merged-4-weight 1.0 \
    --use-native-8-head --native-8-weight 1.0 \
    --use-moe-routing \
    --moe-body-mode=per_expert_h \
    --moe-num-experts=4 \
    --moe-routing-mode=single_pick \
    --moe-expert-n-layers=2 \
    --moe-router-temp=0.5 --moe-target-temp=0.3 \
    --moe-balance-weight=0.05 --moe-supervise-weight=0.1 \
    --moe-router-warmup-steps=5000 \
    --moe-normalize-loss-for-kl --moe-loss-ema-momentum=0.99 \
    --moe-length-cost-weight=0.0 \
    --save-total-limit=2
