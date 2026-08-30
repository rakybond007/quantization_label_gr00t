#!/bin/bash
#SBATCH --job-name=groot_n1_5_libero_moe4_per_expert_body_main_m8_m4_n8_compression_aware_lambda_c_0p1_router_prior
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="GR00T N1.5 finetune on libero: 4-expert MoE (main+m8+m4+n8) + compression-aware router prior (lambda_c=0.1)"
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-groot_n1_5_libero_moe4_compaware.out
#SBATCH --error=out/%j-groot_n1_5_libero_moe4_compaware.err

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
export WANDB_PROJECT=GR00T-libero
mkdir -p out
CKPT_DIR="$BASE_DIR/ckpt/libero/groot/groot_n1_5_bs64_moe4_per_expert_body_compression_aware_lambda_c_0p1"

source $CONDA_PATH/bin/activate gr00t
python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path /sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config libero_multi_horizon \
    --embodiment-tag libero \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-libero-moe4-compression-aware-lambda-c-0p1-bs64 \
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
    --moe-compression-weight=0.1 \
    --moe-router-warmup-steps=5000
