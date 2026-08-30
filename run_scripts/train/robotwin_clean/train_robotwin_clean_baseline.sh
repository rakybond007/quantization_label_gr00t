#!/bin/bash
#SBATCH --job-name=groot_n1_5_robotwin_clean50_baseline_vanilla_no_MoE_no_compression_abs_60k_bs64
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="RoboTwin Clean (50 tasks) — baseline GR00T-N1.5 (no MoE / no compression / standard flow-matching head)."
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-train_robotwin_clean_baseline.out
#SBATCH --error=out/%j-train_robotwin_clean_baseline.err

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
export WANDB_PROJECT=GR00T-robotwin
mkdir -p out
CKPT_DIR="$BASE_DIR/ckpt/robotwin/groot_n1_5_bs64_baseline_clean50"

mapfile -t DATASETS < "$BASE_DIR/run_scripts/train/robotwin_clean/_dataset_paths_clean50.txt"
echo "[$(date '+%T')] passing ${#DATASETS[@]} dataset paths"

source $CONDA_PATH/bin/activate gr00t
python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path "${DATASETS[@]}" \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config robotwin_agilex \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-robotwin-clean50-baseline-bs64 \
    --batch-size 32 --num-gpus 2 --max-steps 60000 --save-steps 10000 \
    --report-to wandb \
    --video-backend torchvision_av
