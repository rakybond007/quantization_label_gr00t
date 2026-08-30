#!/bin/bash
#SBATCH --job-name=train_small_gate_module_A_9k_teacher_comparison_frontier_action_vs_local_cosmos_900ep
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p sjw_alinlab
#SBATCH --gpus=1
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%j-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%j-%x.err
set -eu
: "${MODEL_OUTPUT_DIR:?}"
: "${LABELS:?}"
BASE=$HOME/quantization_agent_workspace/vlm_gate; cd $BASE
CACHE=/rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_frame_cache_robocasa
$HOME/miniconda3/envs/quant_gate_eval/bin/python -u scripts/train_gate_module.py \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels "$LABELS" --cache-dir "$CACHE" --out-dir "$MODEL_OUTPUT_DIR" \
  --epochs 37 --bs 256 --lr 3e-4 --num-workers 8 \
  --wandb "$(basename $MODEL_OUTPUT_DIR)" --task-emb "$CACHE/task_embeddings.npz"
