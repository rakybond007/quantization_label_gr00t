#!/bin/bash
#SBATCH --job-name=build_gate_frame_cache_robocasa_full_for_gate_module_distillation_training
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --exclude=worker-node100,worker-node1
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --output=out/%j-build_cache.out
#SBATCH --error=out/%j-build_cache.err
set -u
BASE="$HOME/quantization_agent_workspace/vlm_gate"
cd "$BASE"
"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u scripts/build_gate_frame_cache.py \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels /rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_gemma_v1/labels/full_merged.parquet \
  --out-dir "$CACHE_DIR" --shard "$SLURM_ARRAY_TASK_ID" --num-shards 10
