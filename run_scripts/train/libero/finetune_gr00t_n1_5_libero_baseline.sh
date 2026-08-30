#!/bin/bash
#SBATCH --job-name=groot_n1_5_libero_baseline
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="GR00T N1.5 finetune on Libero (baseline, no multi-horizon)"
#SBATCH --partition=batch
#SBATCH --output=out/%j-groot_n1_5_libero_baseline.out
#SBATCH --error=out/%j-groot_n1_5_libero_baseline.err

# Baseline libero training: matches the reference libero scripts but with our
# 2-GPU bs=32-per-GPU setup (effective bs=64) for fair comparison with the
# m8+econsist run.

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

export WANDB_PROJECT=GR00T-libero
mkdir -p out

CKPT_DIR="$BASE_DIR/ckpt/libero/groot/groot_n1_5_bs64_baseline"
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"

source $CONDA_PATH/bin/activate gr00t

python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config libero \
    --embodiment-tag libero \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-libero-baseline-fromPT-bs64 \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 60000 \
    --save-steps 10000 \
    --report-to wandb
