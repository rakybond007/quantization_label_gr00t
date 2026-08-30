#!/bin/bash
#SBATCH --job-name=groot_n1_5_robocasa_moe4_shared_body
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="GR00T N1.5 finetune: 4-expert MoE (Option A) — shared body h=16, x_t reconstruction at inference"
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-groot_n1_5_robocasa_moe4_shared_body.out
#SBATCH --error=out/%j-groot_n1_5_robocasa_moe4_shared_body.err

# Option A: 4-expert MoE on robocasa.
#   Experts: 1) main 16 raw, 2) m8 (sum-pair from 16), 3) m4 (sum-pair from 8), 4) n8 raw 8.
#   Body: single forward at action_horizon=16 (shared by all experts).
#   Inference: x_t reconstruction trick (analogous to pi0.5 m8) for experts 2/3/4
#              whose target_h != 16.
# Loss = soft_mixture(router_probs * per_sample_losses) + λ_balance * uniform_reg
#        + λ_supervise * KL(router_probs || softmax(-losses / τ_target))
# Inference: route by router argmax/sample, run only the chosen expert.
#
# IMPORTANT: depends on planned action_head MoE implementation:
#   --use-moe-routing                 (enable router + soft mixture)
#   --moe-body-mode=shared_h16        (single body forward at h=16)
#   --use-merged-4-head               (new: m4 = sum-pair of GT[:8])
#   --use-native-8-head               (new: n8 = raw GT[:8])
# Multi-horizon aux f2/f4 + ensemble-consistency are dropped — the 4 experts
# themselves provide diverse horizon supervision.

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

export WANDB_PROJECT=GR00T-robocasa
mkdir -p out

CKPT_DIR="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_moe4_shared_body"
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"

source $CONDA_PATH/bin/activate gr00t

python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config single_panda_gripper_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-robocasa-moe4-shared-body-fromPT-bs64 \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 60000 \
    --save-steps 10000 \
    --report-to wandb \
    --discrete-action-dims 6 11 \
    --use-merged-8-head \
    --merged-8-weight 1.0 \
    --use-merged-4-head \
    --merged-4-weight 1.0 \
    --use-native-8-head \
    --native-8-weight 1.0 \
    --use-moe-routing \
    --moe-body-mode=shared_h16 \
    --moe-num-experts=4 \
    --moe-expert-n-layers=2 \
    --moe-router-temp=0.5 \
    --moe-target-temp=0.3 \
    --moe-balance-weight=0.05 \
    --moe-supervise-weight=0.1 \
    --moe-router-warmup-steps=5000
