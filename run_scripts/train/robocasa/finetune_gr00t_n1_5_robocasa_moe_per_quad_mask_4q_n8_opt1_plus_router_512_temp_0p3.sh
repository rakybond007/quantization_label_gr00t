#!/bin/bash
#SBATCH --job-name=groot_n1_5_robocasa_moe_per_quad_mask_4q_n8_OPT1plus_router_hidden_512_router_temp_0p3
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="per_quad_mask+n8 OPT1+: OPT1 base + router_temp=0.3 (sharper training-time routing) + router_hidden=512 (2x capacity)"
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-groot_n1_5_robocasa_pqm_n8_opt1plus.out
#SBATCH --error=out/%j-groot_n1_5_robocasa_pqm_n8_opt1plus.err

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
export WANDB_PROJECT=GR00T-robocasa
mkdir -p out
CKPT_DIR="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_moe_pqm_4q_n8_opt1plus_router512_temp0p3"

source $CONDA_PATH/bin/activate gr00t
python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path /sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config single_panda_gripper_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-robocasa-pqm-4q-n8-opt1plus-router512-temp0p3-bs64 \
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
    --moe-router-temp=0.3 \
    --moe-router-hidden=512 \
    --moe-target-temp=0.15 \
    --moe-supervise-weight=0.20 \
    --moe-balance-weight=0.02 \
    --moe-compression-weight=0.0 \
    --moe-min-prob=0.03 \
    --moe-uniform-warmup-steps=2000 \
    --moe-router-warmup-steps=5000
