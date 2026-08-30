#!/bin/bash
#SBATCH --job-name=train_gateB_mlp_head_on_frozen_gr00t_backbone_features_full_robocasa_gemma_labels
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --exclude=worker-node100,worker-node1
#SBATCH --time=12:00:00
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --output=out/%j-train_gate_head.out
#SBATCH --error=out/%j-train_gate_head.err
set -u
BASE="$HOME/quantization_agent_workspace/vlm_gate"
cd "$BASE"
CK="${MODEL_OUTPUT_DIR:-/rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_gemma_v1}"
# Features depend only on (episode,frame,task), identical across gemma/cosmos labels:
# extract once (gemma dir) and share.
FEAT="${FEATURES_DIR:-/rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_gemma_v1/features_backboneB}"
"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u scripts/train_gate_head.py \
  --features-dir "$FEAT" \
  --labels "$CK/labels/full_merged.parquet" \
  --out-dir "$CK/module_B_full" \
  --epochs 30 --bs 512 --lr 1e-3 --num-workers 4 \
  --wandb "${WANDB_RUN:-gateB_gemma_full}"
