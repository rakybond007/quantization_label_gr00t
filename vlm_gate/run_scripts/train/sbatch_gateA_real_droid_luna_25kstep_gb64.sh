#!/bin/bash
#SBATCH --job-name=train_gateA_real_droid_pnp_luna_labels_25kstep_gb64_stepmatch_vs_gateC
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p sjw_alinlab
#SBATCH --gpus=2
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%j-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%j-%x.err
set -eu
: "${MODEL_OUTPUT_DIR:?set via --export=ALL,MODEL_OUTPUT_DIR=...}"
BASE=$HOME/quantization_agent_workspace/vlm_gate
CK=/rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_real_droid_pnp_v1
CACHE=$BASE/output/_gate_distill/cache_real_luna
cd $BASE
# step-match to gateC 25k steps @ global batch 64: 2609 train frames / 64 = 41 steps/epoch -> 610 epochs
$HOME/miniconda3/envs/quant_gate_eval/bin/python -u scripts/train_gate_module.py \
  --dataset-path /sjw_alinlab/home/hojin2/taekwan/Isaac-GR00T/Data/human_data/MoSS/lerobot/pnp_objects \
  --labels $CK/labels/real_luna_full.parquet \
  --cache-dir "$CACHE" \
  --out-dir "$MODEL_OUTPUT_DIR" \
  --epochs 610 --bs 64 --lr 3e-4 --num-workers 8 \
  --wandb "gateA_real_luna_25kstep_gb64" \
  --task-emb "$CACHE/task_embeddings.npz"
