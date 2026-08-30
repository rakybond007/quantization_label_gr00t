#!/bin/bash
#SBATCH --job-name=train_small_gate_module_full_dataset_distillation_from_vlm_labels_robocasa
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --exclude=worker-node100,worker-node1
#SBATCH --time=24:00:00
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --output=out/%j-train_gate.out
#SBATCH --error=out/%j-train_gate.err
set -u
BASE="$HOME/quantization_agent_workspace/vlm_gate"
cd "$BASE"
CK=/rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_${JUDGE_BACKEND}_v1
"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u scripts/train_gate_module.py \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels "$CK/labels/full_merged.parquet" \
  --cache-dir /rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_frame_cache_robocasa \
  --out-dir "$CK/module_A_tc_full" --epochs 30 --bs 256 --lr 3e-4 --num-workers 8 \
  --wandb "gateA_tc_${JUDGE_BACKEND}_full" \
  --task-emb /rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_frame_cache_robocasa/task_embeddings.npz
