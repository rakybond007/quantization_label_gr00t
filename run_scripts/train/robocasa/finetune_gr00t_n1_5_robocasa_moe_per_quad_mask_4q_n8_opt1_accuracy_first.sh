#!/bin/bash
#SBATCH --job-name=groot_n1_5_robocasa_moe_per_quad_mask_4q_n8_OPT1_accuracy_first_sharp_supervise_low_balance
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="per_quad_mask+n8 OPT1: τ_target=0.15 / supervise=0.20 / balance=0.02 / compaware=0 / uniform_warmup=2000 / min_prob=0.03 — push accuracy via decisive low-loss routing"
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-groot_n1_5_robocasa_pqm_n8_opt1.out
#SBATCH --error=out/%j-groot_n1_5_robocasa_pqm_n8_opt1.err

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
export WANDB_PROJECT=GR00T-robocasa
mkdir -p out
CKPT_DIR="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_moe_pqm_4q_n8_opt1_accuracy_first"

source $CONDA_PATH/bin/activate gr00t
python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path /sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config single_panda_gripper_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-robocasa-pqm-4q-n8-opt1-accuracy-bs64 \
    --batch-size 32 --num-gpus 2 --max-steps 60000 --save-steps 10000 \
    --report-to wandb \
    --discrete-action-dims 6 11 \
    --use-merged-8-head --merged-8-weight 1.0 \
    --use-merged-4-head --merged-4-weight 1.0 \
    --use-native-8-head --native-8-weight 1.0 \
    --use-moe-routing \
    --moe-body-mode=per_expert_h \
    --moe-num-experts=4 \
    --moe-routing-mode=per_quad_mask \
    --moe-per-quad-use-n8 \
    --moe-expert-n-layers=2 \
    --moe-router-temp=0.5 \
    --moe-target-temp=0.15 \
    --moe-supervise-weight=0.20 \
    --moe-balance-weight=0.02 \
    --moe-compression-weight=0.0 \
    --moe-min-prob=0.03 \
    --moe-uniform-warmup-steps=2000 \
    --moe-router-warmup-steps=5000
