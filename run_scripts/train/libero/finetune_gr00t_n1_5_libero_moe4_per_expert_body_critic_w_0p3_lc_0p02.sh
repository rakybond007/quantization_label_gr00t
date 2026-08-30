#!/bin/bash
#SBATCH --job-name=groot_n1_5_libero_moe4_per_expert_body_critic_w_0p3_lc_0p02
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="GR00T N1.5 libero: MoE4 (main+m8+m4+n8) + compression-safety critic (w=0.3) + length cost (gamma=0.02)"
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-groot_n1_5_libero_moe4_critic.out
#SBATCH --error=out/%j-groot_n1_5_libero_moe4_critic.err

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
export WANDB_PROJECT=GR00T-libero
mkdir -p out
CKPT_DIR="$BASE_DIR/ckpt/libero/groot/groot_n1_5_bs64_moe4_per_expert_body_critic_w_0p3_lc_0p02"

source $CONDA_PATH/bin/activate gr00t
python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path /sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config libero_multi_horizon \
    --embodiment-tag libero \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-libero-moe4-critic-w-0p3-lc-0p02-bs64 \
    --batch-size 32 --num-gpus 2 --max-steps 60000 --save-steps 10000 \
    --report-to wandb \
    --discrete-action-dims 6 \
    --use-merged-8-head --merged-8-weight 1.0 \
    --use-merged-4-head --merged-4-weight 1.0 \
    --use-native-8-head --native-8-weight 1.0 \
    --use-moe-routing \
    --moe-body-mode=per_expert_h \
    --moe-num-experts=4 \
    --moe-expert-n-layers=2 \
    --moe-router-temp=0.5 \
    --moe-target-temp=0.3 \
    --moe-balance-weight=0.05 \
    --moe-supervise-weight=0.1 \
    --moe-length-cost-weight=0.02 \
    --moe-critic-weight=0.3 \
    --moe-critic-hidden=256 \
    --moe-router-warmup-steps=5000
