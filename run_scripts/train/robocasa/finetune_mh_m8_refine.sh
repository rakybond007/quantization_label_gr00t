#!/bin/bash
#SBATCH --job-name=mh_m8_refinement
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="[Method 2] Fine-tune mh_m8 from 60k ckpt + add post-hoc m8 refinement MLP"
#SBATCH --partition=batch
#SBATCH --output=out/%j-mh_m8_refine.out
#SBATCH --error=out/%j-mh_m8_refine.err

# Adds a small refinement MLP that maps (vl_pooled, m8_pred_8x32) -> residual
# delta. Inference: head='m8_refined' returns m8_pred + delta. Backward compat:
# head='m8' still returns raw m8 (no refine). Refinement supervised by GT
# clean_m8 with a perturbed-GT proxy as the input (cheap training surrogate).
# 10k extra steps, base + body fully trainable but main-loss warmups disabled.

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
export WANDB_PROJECT=GR00T-robocasa
mkdir -p out

SRC_CKPT="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_mh_m8_econsist_discfix/checkpoint-60000"
CKPT_DIR="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_mh_m8_econsist_discfix_refine"
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"

source $CONDA_PATH/bin/activate gr00t

python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config single_panda_gripper_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path "$SRC_CKPT" \
    --run-name GR00T-N1.5-mh_m8_econsist-refine \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 10000 \
    --save-steps 2000 \
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
    --use-m8-refinement \
    --m8-refinement-weight 1.0 \
    --m8-refinement-hidden 512
