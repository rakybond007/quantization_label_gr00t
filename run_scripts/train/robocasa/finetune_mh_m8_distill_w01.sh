#!/bin/bash
#SBATCH --job-name=mh_m8_distill_w01
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="[Method 1] Fine-tune mh_m8 from 60k ckpt with strong f2->m8 distillation"
#SBATCH --partition=batch
#SBATCH --output=out/%j-mh_m8_distill_w01.out
#SBATCH --error=out/%j-mh_m8_distill_w01.err

# Resumes m8 head + body from existing 60k checkpoint (mh_m8_econsist) and adds
# strong distillation:  loss += 1.0 * MSE(v_m8, v_f2[:8].detach())
# Goal: m8-only inference quality matches/exceeds ensemble->8.
# Backward compat: nothing in head structure changes; only loss term added.
# 10k extra steps, aux warmup OFF (model already trained).
# Global batch = 2 * 32 = 64

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
export WANDB_PROJECT=GR00T-robocasa
mkdir -p out

# Start from the previously-trained mh_m8+econsist checkpoint
SRC_CKPT="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_mh_m8_econsist_discfix/checkpoint-60000"
CKPT_DIR="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_mh_m8_econsist_discfix_distill_w0_1"
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"

source $CONDA_PATH/bin/activate gr00t

python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config single_panda_gripper_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path "$SRC_CKPT" \
    --run-name GR00T-N1.5-mh_m8_econsist-distill-from-f2-w0.1 \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 30000 \
    --save-steps 10000 \
    --report-to wandb \
    --use-multi-horizon-loss \
    --multi-horizon-factors 2 4 \
    --multi-horizon-loss-weights 1.0 1.0 \
    --multi-horizon-main-weight 1.0 \
    --aux-grad-scale-to-body 0.1 \
    --aux-loss-warmup-steps 0 \
    --discrete-action-dims 6 11 \
    --use-merged-8-head \
    --merged-8-weight 1.0 \
    --use-ensemble-consistency-loss \
    --ensemble-consistency-weight 0.1 \
    --ensemble-consistency-warmup-steps 0 \
    --m8-distill-from-f2-weight 0.1
