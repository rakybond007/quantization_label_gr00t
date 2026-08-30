#!/bin/bash
#SBATCH --job-name=extract_frozen_gr00t_backbone_features_full_robocasa_labels_for_gateB_head_distillation
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --exclude=worker-node100,worker-node1
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --output=out/%j-extract_feat.out
#SBATCH --error=out/%j-extract_feat.err
set -u
BASE="$HOME/quantization_agent_workspace/vlm_gate"
cd "$BASE"
CK="${MODEL_OUTPUT_DIR:-/rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_gemma_v1}"
"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u scripts/extract_gate_backbone_features.py \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels "$CK/labels/full_merged.parquet" \
  --model-path "$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000" \
  --out-dir "$CK/features_backboneB" \
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards 10 --batch-size 16
