#!/bin/bash
#SBATCH --job-name=groot_n1_5_libero_moe4_v1_K4_b_only_no_metaq_warmup10k_ema0p95_rt1p0_60k_bs64
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="Libero MoE4 v1 (u16/c8/c4/n8) + [B] EMA-norm KL only + no metaq + no [D]. warmup=10k + EMA=0.95 + router_temp=1.0."
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-train_libero_moe4_v1_K4_b_only_no_metaq_warmup10k_ema0p95_rt1p0.out
#SBATCH --error=out/%j-train_libero_moe4_v1_K4_b_only_no_metaq_warmup10k_ema0p95_rt1p0.err

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
export WANDB_PROJECT=GR00T-libero
mkdir -p out
CKPT_DIR="$BASE_DIR/ckpt/libero/groot/groot_n1_5_bs64_fair_moe_v1_K4_b_only_no_metaq_warmup10k_ema0p95_rt1p0"
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"

source $CONDA_PATH/bin/activate gr00t
python $BASE_DIR/scripts/gr00t_finetune_fair_moe.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config libero_multi_horizon \
    --embodiment-tag libero \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-libero-moe4-v1-K4-b-only-no-metaq-warmup10k-ema0p95-rt1p0-bs64 \
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
    --moe-router-temp=1.0 --moe-target-temp=0.3 \
    --moe-balance-weight=0.05 --moe-supervise-weight=0.1 \
    --moe-router-warmup-steps=10000 \
    --no-moe-shared-t \
    --moe-normalize-loss-for-kl --moe-loss-ema-momentum=0.95 \
    --moe-balance-type=mean_sq \
    --moe-length-cost-weight=0.0 \
    --save-total-limit=2
