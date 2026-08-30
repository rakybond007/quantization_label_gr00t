#!/bin/bash
#SBATCH --job-name=groot_n1_5_robotwin_clean50_fair_moe_v1_K4_B_only_abs_action_no_metaq_60k_bs64
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="RoboTwin Clean (50 tasks) — fair_moe v1 K=4 {u16,c8,c4,n8} + B only + abs action."
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-train_robotwin_clean_fair_moe_b_only_abs.out
#SBATCH --error=out/%j-train_robotwin_clean_fair_moe_b_only_abs.err

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
export WANDB_PROJECT=GR00T-robotwin
mkdir -p out
CKPT_DIR="$BASE_DIR/ckpt/robotwin/groot_n1_5_bs64_fair_moe_clean50_b_only_abs"

# Read 50 dataset paths from helper file (one per line) into a bash array
mapfile -t DATASETS < "$BASE_DIR/run_scripts/train/robotwin_clean/_dataset_paths_clean50.txt"
echo "[$(date '+%T')] passing ${#DATASETS[@]} dataset paths"

source $CONDA_PATH/bin/activate gr00t
python $BASE_DIR/scripts/gr00t_finetune_fair_moe.py \
    --dataset-path "${DATASETS[@]}" \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config robotwin_agilex \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-robotwin-clean50-fair-moe-v1-K4-b-only-abs-bs64 \
    --batch-size 32 --num-gpus 2 --max-steps 60000 --save-steps 10000 \
    --report-to wandb \
    --video-backend torchvision_av \
    --action-mode abs \
    --discrete-action-dims 6 13 \
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
